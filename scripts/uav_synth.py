# -*- coding: utf-8 -*-
"""斜拍 UAV 照片的對位：以 DTM + 底圖「合成斜視圖」再比對（LoFTR 直接配正射底圖在斜拍照上失敗率高，因為視角差太大）。

做法（無 EXIF，相機姿態完全未知）
  1. 以 API 座標為中心，把 20 m DTM（重採樣 2 m）+ Esri 底圖建成場景；用 GPU 沿相機光線對高度場做 ray-marching，
     從任意相機位置渲染「合成斜視圖」與逐像素 3D 位置圖（x 東, y 北, z 高）。
  2. 粗搜：方位角 × 距離 × 離地高 × 焦距，共約 144 個假設姿態，每個渲染 512 px 影像，與真實照片做 LoFTR，
     以基礎矩陣 MAGSAC 內點數評分（斜視圖對斜視圖，視角差小，匹配遠比正射底圖容易）。
  3. 取前 3 名精修：用合成圖的 3D 位置圖把匹配點變成 2D–3D 對應 → 既有的 solve_camera（PnP + 穩健最小二乘）
     → 以解出的相機重新渲染（影像對幾乎對齊）再匹配，共 3 輪。
  4. 留出 30% 對應點量地面誤差（RMSE）；通過門檻（預設 10 m）才輸出，沿用 uav_ortho_dtm.ortho_stage 做 DTM 正射。
限制：場景半徑 2 km（相機與目標需在其內）；合成圖與照片的外觀差異（季節、崩塌前後）會降低匹配；
  主點固定在影像中心、k1 限 ±0.2、roll 假設接近 0（粗搜時）；3D 點精度受 20 m DTM 與底圖定位誤差限制。

用法：python scripts/uav_synth.py <event_id> [--lat L --lon L] [--publish] [--hold-max 10] [--search-only]
輸出：scripts/uav_register/work_synth/<id>/（search.json、best_synth.jpg、best_match.jpg、report.json、ortho.png…）
"""
import argparse, concurrent.futures as cf, json, math, os, sys, time
from pathlib import Path
from types import SimpleNamespace

import cv2, numpy as np, torch
import torch.nn.functional as F

HERE = Path(__file__).parent / "uav_register"
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.append(str(HERE))
import register as R      # noqa: E402
import dtm20 as DTM       # noqa: E402
import thermal_guard as TG  # noqa: E402
import uav_ortho_dtm as U   # noqa: E402

RE = U.RE
DEV = os.environ.get("UAV_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
_MATCHER = None


def matcher():
    global _MATCHER
    if _MATCHER is None:
        import kornia.feature as KF
        _MATCHER = KF.LoFTR(pretrained="outdoor").to(DEV).eval()
    return _MATCHER


_CLAHE = cv2.createCLAHE(2.5, (8, 8))


def gray(im):
    return _CLAHE.apply(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))


def loftr(a_bgr, b_bgr, thr=0.5):
    """兩張同尺寸（8 的倍數）影像做 LoFTR；回傳 (pts_a, pts_b)（各自像素座標）。"""
    TG.wait_cool("(synth match)", min_interval=0)
    t0 = time.time()
    t = lambda g: torch.from_numpy(np.ascontiguousarray(g))[None, None].float().to(DEV) / 255.0
    with torch.no_grad():
        out = matcher()({"image0": t(gray(a_bgr)), "image1": t(gray(b_bgr))})
    if DEV.startswith("cuda"):
        torch.cuda.synchronize()
    keep = out["confidence"].cpu().numpy() > thr
    pa, pb = out["keypoints0"].cpu().numpy()[keep], out["keypoints1"].cpu().numpy()[keep]
    TG.duty_sleep(time.time() - t0)
    return pa, pb


class Scene:
    """以 (lat0, lon0) 為原點的局部公尺座標場景：x 東、y 北、z 高（減去中心高程）。與 uav_ortho_dtm 的 (ox, oy, zmean) 約定一致。"""

    def __init__(self, lat0, lon0, cache: Path, half=2000.0, res=2.0, z=17):
        self.lat0, self.lon0, self.half, self.res, self.z = lat0, lon0, half, res, z
        k = self.k = math.cos(math.radians(lat0))
        mx0, my0 = R.merc(lon0, lat0)
        self.ox, self.oy = mx0 * k, my0 * k
        n = self.n = int(2 * half / res)
        xs = -half + (np.arange(n) + 0.5) * res
        ys = half - (np.arange(n) + 0.5) * res
        X, Y = np.meshgrid(xs, ys)
        MX, MY = (X + self.ox) / k, (Y + self.oy) / k
        lon = np.degrees(MX / RE)
        lat = np.degrees(2 * np.arctan(np.exp(MY / RE)) - math.pi / 2)
        zc = cache / f"scene_{z}_{int(half)}_{lat0:.5f}_{lon0:.5f}.npz"
        if zc.exists():
            d = np.load(zc)
            Zabs, ref, self.x0, self.y0 = d["Z"], d["ref"], int(d["x0"]), int(d["y0"])
        else:
            Zabs = DTM.Dtm20Source().sample_points(lon.ravel(), lat.ravel(), max_px=4096).reshape(n, n)
            ref, self.x0, self.y0 = self._mosaic(MX, MY)
            np.savez_compressed(zc, Z=Zabs, ref=ref, x0=self.x0, y0=self.y0)
        self.world = 2 * math.pi * RE
        self.ts = self.world / 2 ** z / 256
        self.ref = ref
        self.zmean = float(Zabs[n // 2, n // 2])
        self.Z = (Zabs - self.zmean).astype(np.float32)
        px = ((MX + self.world / 2) / self.ts - self.x0 * 256).astype(np.float32)
        py = ((self.world / 2 - MY) / self.ts - self.y0 * 256).astype(np.float32)
        tex = cv2.remap(ref, px, py, cv2.INTER_AREA if self.ts / k > res * 1.5 else cv2.INTER_LINEAR)
        self.Ht = torch.from_numpy(self.Z)[None, None].to(DEV)
        self.Tt = torch.from_numpy(tex.copy()).permute(2, 0, 1)[None].float().to(DEV) / 255.0

    def _mosaic(self, MX, MY):
        world = 2 * math.pi * RE
        ts = world / 2 ** self.z / 256
        tx0, tx1 = int((MX.min() + world / 2) // (ts * 256)), int((MX.max() + world / 2) // (ts * 256))
        ty0, ty1 = int((world / 2 - MY.max()) // (ts * 256)), int((world / 2 - MY.min()) // (ts * 256))
        ref = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256, 3), np.uint8)

        def one(ij):
            i, j = ij
            for _ in range(3):
                try:
                    im = cv2.imdecode(np.frombuffer(R.fetch(R.SRC["esri"].format(z=self.z, x=tx0 + i, y=ty0 + j)), np.uint8), 1)
                    if im is not None and im.shape[:2] == (256, 256):
                        return i, j, im
                except Exception:  # noqa: BLE001
                    time.sleep(1)
            return i, j, None
        jobs = [(i, j) for j in range(ty1 - ty0 + 1) for i in range(tx1 - tx0 + 1)]
        with cf.ThreadPoolExecutor(8) as ex:
            for i, j, im in ex.map(one, jobs):
                if im is not None:
                    ref[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256] = im
        print("mosaic", ref.shape, len(jobs), "tiles", flush=True)
        return ref, tx0, ty0

    def terrain_z(self, x, y):
        c = int(np.clip((x + self.half) / self.res, 0, self.n - 1)); r = int(np.clip((self.half - y) / self.res, 0, self.n - 1))
        return float(self.Z[r, c])

    @torch.no_grad()
    def render(self, C, Rm, f, w, h, tmax=4500.0):
        """從相機中心 C、旋轉 Rm（Xc = Rm (X − C)）、焦距 f(px) 渲染 w×h 影像。回傳 (BGR uint8, pos(h,w,3) float32, valid(h,w) bool)。"""
        TG.wait_cool("(render)", min_interval=0)
        t0 = time.time()
        u = torch.arange(w, device=DEV).float() + 0.5; v = torch.arange(h, device=DEV).float() + 0.5
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        dc = torch.stack([(uu - w / 2) / f, (vv - h / 2) / f, torch.ones_like(uu)], -1).reshape(-1, 3)
        Rt = torch.tensor(Rm, dtype=torch.float32, device=DEV)
        d = dc @ Rt                                    # = (Rm^T dc^T)^T
        d = d / d.norm(dim=1, keepdim=True)
        Ct = torch.tensor(C, dtype=torch.float32, device=DEV)
        N = d.shape[0]
        hit = torch.zeros(N, dtype=torch.bool, device=DEV)
        thit = torch.zeros(N, device=DEV)
        prev_d = prev_t = None
        t = 4.0
        hh = self.half
        while t < tmax:
            P = Ct + t * d
            gx = (P[:, 0] + hh) / (2 * hh) * 2 - 1; gy = (hh - P[:, 1]) / (2 * hh) * 2 - 1
            zt = F.grid_sample(self.Ht, torch.stack([gx, gy], -1).view(1, N, 1, 2), mode="bilinear",
                               padding_mode="border", align_corners=False).view(N)
            dif = P[:, 2] - zt
            if prev_d is not None:
                inside = (gx.abs() < 1) & (gy.abs() < 1)
                cross = (~hit) & (dif < 0) & (prev_d >= 0) & inside
                frac = prev_d / (prev_d - dif).clamp(min=1e-6)
                thit = torch.where(cross, prev_t + (t - prev_t) * frac, thit)
                hit |= cross
            prev_d, prev_t = dif, t
            t += max(3.0, 0.006 * t)
        X = Ct + thit[:, None] * d
        gx = (X[:, 0] + hh) / (2 * hh) * 2 - 1; gy = (hh - X[:, 1]) / (2 * hh) * 2 - 1
        col = F.grid_sample(self.Tt, torch.stack([gx, gy], -1).view(1, N, 1, 2), mode="bilinear",
                            padding_mode="border", align_corners=False)[0, :, :, 0].T
        col = torch.where(hit[:, None], col, torch.zeros_like(col))
        img = (col.reshape(h, w, 3).clamp(0, 1) * 255).byte().cpu().numpy()
        pos = X.reshape(h, w, 3).cpu().numpy()
        valid = hit.reshape(h, w).cpu().numpy()
        if DEV.startswith("cuda"):
            torch.cuda.synchronize()
        TG.duty_sleep(time.time() - t0)
        return img, pos, valid


def look_at(C, T):
    f = T - C; f = f / np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(f, right)
    return np.stack([right, down, f])


def pose_from_p(p):
    Rm, _ = cv2.Rodrigues(p[:3])
    return -Rm.T @ p[3:6], Rm


def fund_inliers(pa, pb, thr=1.5):
    if len(pa) < 12:
        return 0, None
    F_, m = cv2.findFundamentalMat(pa, pb, cv2.USAC_MAGSAC, thr, 0.999, 10000)
    return (int(m.sum()) if m is not None else 0), m


def dir_to_R(az_deg, pitch_deg):
    """相機朝向：方位角（北起順時針）、俯仰角（向下為負），roll = 0。回傳 Rm（Xc = Rm (X − C)）。"""
    a, p = math.radians(az_deg), math.radians(pitch_deg)
    f = np.array([math.sin(a) * math.cos(p), math.cos(a) * math.cos(p), math.sin(p)])
    right = np.cross(f, [0, 0, 1.0]); right /= np.linalg.norm(right)
    return np.stack([right, np.cross(f, right), f])


def pnp_score(pa, pb, pos, valid, f, ws, hs):
    """匹配點 → (真實照片像素, 合成圖 3D 位置) → 固定焦距的 PnP-RANSAC 內點數。比基礎矩陣更不容易被一堆平行誤配騙過
    （合成圖的 3D 是剛性場景，隨機誤配很難湊出一致的姿態）。"""
    X, ok = lookup_pos(pos, valid, pb)
    if ok.sum() < 8:
        return 0
    K = np.array([[f, 0, ws / 2], [0, f, hs / 2], [0, 0, 1.0]])
    try:
        okp, _rv, _tv, inl = cv2.solvePnPRansac(X[ok], pa[ok].astype(np.float64), K, None, iterationsCount=1500,
                                                reprojectionError=6.0, flags=cv2.SOLVEPNP_EPNP)
    except cv2.error:
        return 0
    return int(len(inl)) if okp and inl is not None else 0


def search(sc, photo_s, ws, hs, log=print, prior="camera"):
    """粗搜相機姿態。prior="camera"：API 座標＝拍攝位置（以石公溪實測：相機距 API 座標約 110 m）；
    "target"：API 座標＝拍攝目標，相機在其周圍。回傳依 PnP 內點數排序的 [(score, desc, C, R, f)]。"""
    res = []
    hyps = []
    if prior == "camera":
        for pitch in (-25, -45, -65):
            for H in (150, 400):
                for az in range(0, 360, 20):
                    hyps.append((f"az{az} pitch{pitch} H{H}", np.array([0.0, 0.0, sc.terrain_z(0, 0) + H]), dir_to_R(az, pitch), 0.85 * ws))
    else:
        for D in (500, 1000, 1700):
            for H in (150, 400):
                for az in range(0, 360, 30):
                    cx, cy = D * math.sin(math.radians(az)), D * math.cos(math.radians(az))
                    if max(abs(cx), abs(cy)) > sc.half - 100:
                        continue
                    C = np.array([cx, cy, sc.terrain_z(cx, cy) + H])
                    for fr in (0.65, 0.95):
                        hyps.append((f"tgt az{az} D{D} H{H} f{fr}", C, look_at(C, np.zeros(3)), fr * ws))
    for i, (desc, C, Rm, f) in enumerate(hyps):
        img, pos, valid = sc.render(C, Rm, f, ws, hs)
        if valid.mean() < 0.3:
            res.append((0, desc, C, Rm, f)); continue
        pa, pb = loftr(photo_s, img)
        res.append((pnp_score(pa, pb, pos, valid, f, ws, hs), desc, C, Rm, f))
        if (i + 1) % 36 == 0:
            log(f"  {i + 1}/{len(hyps)} best so far {max(r[0] for r in res)}")
    res.sort(key=lambda r: -r[0])
    return res


def lookup_pos(pos, valid, pts):
    """合成圖像素 → 3D 位置（最近像素，且 3×3 鄰域皆有效，避開遮擋邊緣）。回傳 (X, ok)。"""
    h, w = valid.shape
    ix = np.clip(np.round(pts[:, 0] - 0.5).astype(int), 1, w - 2); iy = np.clip(np.round(pts[:, 1] - 0.5).astype(int), 1, h - 2)
    ok = np.ones(len(pts), bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ok &= valid[iy + dy, ix + dx]
    # 鄰域位置差過大 = 跨越深度不連續
    ok &= np.linalg.norm(pos[iy, ix] - pos[iy + 1, ix + 1], axis=1) < 30
    return pos[iy, ix].astype(np.float64), ok


def refine(sc, uav, C, Rm, f_r, W, log=print, rounds=3):
    """以 W 寬度渲染 → 匹配 → 3D–2D → solve_camera；以解出相機再渲染，共 rounds 輪。"""
    h, w = uav.shape[:2]
    Hh = int(round(W * h / w / 8)) * 8
    photo = cv2.resize(uav, (W, Hh), interpolation=cv2.INTER_AREA)
    cam = U.Cam(w, h)
    best = None
    for r in range(rounds):
        img, pos, valid = sc.render(C, Rm, f_r, W, Hh)
        pa, pb = loftr(photo, img, 0.4)
        X, ok = lookup_pos(pos, valid, pb)
        uv = pa[ok] * (w / W); X = X[ok]
        log(f"  round {r + 1}: matches {len(pa)}, with 3D {len(X)}")
        if len(X) < 15:
            return best
        sol = U.solve_camera(cam, X, uv, w, h)
        if sol is None:
            return best
        p, keep = sol
        n_in = int(keep.sum())
        log(f"    camera f={p[6]:.0f}px ({p[6] / w:.2f}w) k1={p[7]:.3f} inliers {n_in}/{len(X)}")
        if best is None or n_in > best["n"]:
            best = {"p": p, "X": X, "uv": uv, "keep": keep, "n": n_in, "photo": photo, "synth": img, "pa": pa, "pb": pb}
        C, Rm = pose_from_p(p)
        f_r = p[6] * (W / w)
    return best


def holdout(cam, X, uv, clean, w, h, seed=7):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(clean); nt = int(len(idx) * 0.3)
    te, tr = idx[:nt], idx[nt:]
    sol = U.solve_camera(cam, X[tr], uv[tr], w, h)
    if sol is None:
        return None
    p, _ = sol
    C, _R = pose_from_p(p)
    uvp, _z = cam.project(p, X[te])
    dist = np.linalg.norm(X[te] - C, axis=1)
    e_c = np.linalg.norm(uvp - uv[te], axis=1) * dist / p[6]
    Hh, _ = cv2.findHomography(uv[tr], X[tr, :2], cv2.USAC_MAGSAC, 4.0, maxIters=20000, confidence=0.9999)
    e_h = np.linalg.norm(cv2.perspectiveTransform(uv[te][None].astype(np.float32), Hh)[0] - X[te, :2], axis=1) if Hh is not None else np.full(len(te), np.nan)

    def stat(e):
        e = np.sort(e[np.isfinite(e)])
        return {"rmse_m": round(float(np.sqrt((e ** 2).mean())), 2), "median_m": round(float(np.median(e)), 2),
                "rmse_p90_m": round(float(np.sqrt((e[: max(1, int(len(e) * 0.9))] ** 2).mean())), 2)}
    return {"camera_dtm": stat(e_c), "homography": stat(e_h), "n_holdout": int(nt), "p_train": p}


def find_rec(eid):
    for f in ("uav_cands.json", "uav_cands_2025.json"):
        for r in json.loads((HERE / f).read_text(encoding="utf-8")):
            if r["EventID"] == eid:
                return r
    return None


def make_meta(eid, rec, sc, X, keep):
    date = ((rec or {}).get("PhotoDate") or "")[:10]
    rec = rec or {}
    pts = X[keep]
    lo, hi = pts[:, :2].min(0), pts[:, :2].max(0)
    pad = 0.35 * (hi - lo) + 40
    lo, hi = lo - pad, hi + pad
    k = sc.k
    mx0, my0 = (lo[0] + sc.ox) / k, (lo[1] + sc.oy) / k
    mx1, my1 = (hi[0] + sc.ox) / k, (hi[1] + sc.oy) / k
    return {
        "id": eid,
        "title": f"{rec.get('County', '')}{rec.get('Town', '')}{rec.get('Vill') or ''}（UAV 空拍，{date}）"
                 + (f"　{rec['DisasterName']}" if rec.get("DisasterName") else ""),
        "description": (rec.get("Description") or "").strip(),
        "source_page": f"https://photo.ardswc.gov.tw/Repository/ViewEvent/{eid}",
        "attribution": f"{rec.get('Source') or '水保署歷史影像平台'}（照片授權方式：姓名標示，CC-BY）；農業部農村發展及水土保持署歷史影像平台",
        "bounds_merc": {"minx": mx0, "miny": my0, "maxx": mx1, "maxy": my1},
        "view": {"heading": 0, "pitch": -35, "range": 1800},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eid")
    ap.add_argument("--lat", type=float); ap.add_argument("--lon", type=float)
    ap.add_argument("--cache", default=str(HERE / "work_synth"))
    ap.add_argument("--hold-max", type=float, default=float(os.environ.get("UAV_HOLD_MAX", "10")))
    ap.add_argument("--min-inliers", type=int, default=40)
    ap.add_argument("--gsd", type=float, default=0.5)
    ap.add_argument("--half", type=float, default=2000.0)
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--search-only", action="store_true")
    ap.add_argument("--refine-top", type=int, default=3)
    ap.add_argument("--prior", default="camera", choices=["camera", "target"])
    a = ap.parse_args()
    out = Path(a.cache) / a.eid
    out.mkdir(parents=True, exist_ok=True)
    rec = find_rec(a.eid)
    lat = a.lat if a.lat is not None else rec["Lat"]; lon = a.lon if a.lon is not None else rec["Lng"]
    up = out / "uav.jpg"
    if not up.exists():
        cv2.imwrite(str(up), U.fetch_uav(a.eid), [cv2.IMWRITE_JPEG_QUALITY, 97])
    uav = cv2.imread(str(up)); h, w = uav.shape[:2]
    print("uav", w, h, "center", lat, lon, flush=True)
    t0 = time.time()
    sc = Scene(lat, lon, out, half=a.half)
    print(f"scene {sc.n}² @ {sc.res} m, DTM relief {sc.Z.min():.0f}..{sc.Z.max():.0f} m  ({time.time() - t0:.0f}s)", flush=True)
    ws = 512; hs = int(round(ws * h / w / 8)) * 8
    photo_s = cv2.resize(uav, (ws, hs), interpolation=cv2.INTER_AREA)
    cand = search(sc, photo_s, ws, hs, log=lambda m: print(m, flush=True), prior=a.prior)
    (out / "search.json").write_text(json.dumps([{"score": r[0], "hyp": r[1]} for r in cand[:12]], indent=1))
    print("search top:", [(r[0], r[1]) for r in cand[:5]], f"({time.time() - t0:.0f}s)", flush=True)
    if a.search_only or cand[0][0] < 10:
        print("粗搜失敗（最佳 PnP 內點 < 10）" if cand[0][0] < 10 else "search only")
        return 2 if cand[0][0] < 10 else 0
    best = None
    for r in cand[:a.refine_top]:
        print(f"refine {r[1]} (search score {r[0]})", flush=True)
        b = refine(sc, uav, r[2], r[3], r[4] * (640 / ws), 640, log=lambda m: print(m, flush=True))
        if b is not None and (best is None or b["n"] > best["n"]):
            best = b
    if best is None or best["n"] < 12:
        print("精修失敗"); return 3
    X, uv, keep, p = best["X"], best["uv"], best["keep"], best["p"]
    clean = np.where(keep)[0]
    cam = U.Cam(w, h)
    ho = holdout(cam, X, uv, clean, w, h)
    if ho is None:
        print("留出檢核失敗"); return 3
    pf = U.solve_camera(cam, X[clean], uv[clean], w, h)[0]
    C, _R = pose_from_p(pf)
    hull = cv2.convexHull(uv[clean].astype(np.float32)); cover = float(cv2.contourArea(hull) / (w * h))
    rep = {"n_match": int(len(X)), "n_clean": int(len(clean)), "n_holdout": ho["n_holdout"], "homography": ho["homography"],
           "camera_dtm": ho["camera_dtm"], "f_px": float(pf[6]), "k1": float(pf[7]), "cam_height_agl_m": float(C[2] - sc.terrain_z(C[0], C[1])),
           "support_cover": round(cover, 2), "method": "synthetic-oblique"}
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps(rep, ensure_ascii=False, indent=1), flush=True)
    both = np.hstack([best["photo"], best["synth"]])
    cv2.imwrite(str(out / "best_match.jpg"), both, [cv2.IMWRITE_JPEG_QUALITY, 85])
    ok = len(clean) >= a.min_inliers and ho["camera_dtm"]["rmse_m"] <= a.hold_max and cover >= 0.2
    print(("通過" if ok else "未通過") + f"：內點 {len(clean)}（≥{a.min_inliers}）、留出 RMSE {ho['camera_dtm']['rmse_m']} m（≤{a.hold_max}）、覆蓋 {cover:.0%}（≥20%）", flush=True)
    meta = make_meta(a.eid, rec, sc, X, clean)
    args = SimpleNamespace(eid=a.eid, gsd=a.gsd, publish=False, z=sc.z)
    U.ortho_stage(args, out, uav, cam, pf, meta, sc.ox, sc.oy, sc.zmean, lat, DTM.Dtm20Source(), sc.ref, sc.x0, sc.y0, sc.ts, sc.world)
    if ok and a.publish:
        d = REPO / "webapp/change_detect_viewer/static/uav" / a.eid
        d.mkdir(parents=True, exist_ok=True)
        ortho = cv2.imread(str(out / "ortho.png"), cv2.IMREAD_UNCHANGED)
        v = ortho[..., 3] > 0
        bm = meta["bounds_merc"]; pix = a.gsd / math.cos(math.radians(lat))
        oh, ow = ortho.shape[:2]
        U.publish(args, out, ortho[..., :3], v, meta, ow, oh, bm, pix, lat, pf)
        mf = d / "meta.json"
        m = json.loads(mf.read_text(encoding="utf-8"))
        m["method"] = ("DTM 合成斜視圖匹配：以 20 m DTM＋Esri 底圖渲染合成斜視圖（相機姿態粗搜），LoFTR 匹配真實照片與合成圖 → "
                       "合成圖的 3D 位置 → 相機姿態解算（PnP + 穩健最小二乘）→ 沿相機光線與 DTM 求交的正射重採樣；UAV 無 EXIF，純影像內容對位")
        m["quality"]["note"] = "（斜拍照，經合成斜視圖匹配）" + m["quality"]["note"]
        m["quality"]["support_cover"] = rep["support_cover"]
        mf.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
