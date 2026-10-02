"""以 DTM 做「真正的」UAV 糾正：用 LoFTR 對應點 + DTM 高程解相機姿態（焦距、旋轉、位置、k1），
再沿相機光線與 DTM 求交做正射重採樣；與單應矩陣（平面假設）在同一批留出點上比較精度。

用法：python scripts/uav_ortho_dtm.py <event_id> [--cache DIR] [--z 18] [--device cuda]
前置：static/uav/<id>/meta.json 已有單應矩陣對位成果（只用它的四角當初始位置，省掉旋轉/尺度搜尋）。
輸出（DIR 下）：uav.jpg、ref.jpg、matches.npz、camera.json、ortho.png、report.json
"""
import argparse, json, math, sys
from pathlib import Path

import cv2, numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).parent / "uav_register"          # 不放進 uav_register/：該目錄的 select.py 會遮蔽標準庫 select
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.append(str(HERE))
import register as R           # noqa: E402
import dtm20 as DTM            # noqa: E402

RE = 6378137.0


def fetch_uav(eid):
    import urllib.request
    req = urllib.request.Request(f"https://photo.ardswc.gov.tw/api/Download/{eid}", headers={"User-Agent": "Mozilla/5.0 ardswc-uav-register"})
    return cv2.imdecode(np.frombuffer(urllib.request.urlopen(req, timeout=60).read(), np.uint8), cv2.IMREAD_COLOR)


def merc_to_local(mx, my, lat0):
    """Web Mercator → 近似地面公尺（乘 cos φ0，等角所以 x/y 同尺度）。"""
    k = math.cos(math.radians(lat0))
    return mx * k, my * k


def match_prewarped(pre, mask, ref, dev, maxd):
    """對「已預先 warp 到底圖像素格」的 UAV 與底圖分塊 LoFTR。回傳 (P_pre, P_ref)，兩者都是底圖像素格座標。
    maxd：預對位後匹配點允許的最大像素距離（第 1 輪用單應矩陣預對位殘差大，第 2 輪用相機+DTM 預對位可收緊）。"""
    import torch, kornia.feature as KF
    global _MATCHER
    if "_MATCHER" not in globals():
        _MATCHER = KF.LoFTR(pretrained="outdoor").to(dev).eval()
    matcher = _MATCHER
    clahe = cv2.createCLAHE(2.5, (8, 8))
    prep = lambda im: clahe.apply(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    t = lambda g: torch.from_numpy(np.ascontiguousarray(g))[None, None].float().to(dev) / 255.0
    rh, rw = ref.shape[:2]
    pg, rg = prep(pre), prep(ref)
    P_ref, P_pre = [], []
    WIN, STRIDE = 480, 240
    for y in range(0, rh - 63, STRIDE):
        for x in range(0, rw - 63, STRIDE):
            m = mask[y:y + WIN, x:x + WIN]
            if m.mean() < 0.4:
                continue
            a, b = pg[y:y + WIN, x:x + WIN], rg[y:y + WIN, x:x + WIN]
            a, b = a[:(a.shape[0] // 8) * 8, :(a.shape[1] // 8) * 8], b[:(b.shape[0] // 8) * 8, :(b.shape[1] // 8) * 8]
            with torch.no_grad():
                out = matcher({"image0": t(a), "image1": t(b)})
            keep = out["confidence"].cpu().numpy() > 0.3
            if keep.sum() < 8:
                continue
            P_pre.append(out["keypoints0"].cpu().numpy()[keep] + [x, y])
            P_ref.append(out["keypoints1"].cpu().numpy()[keep] + [x, y])
    P_pre, P_ref = np.vstack(P_pre), np.vstack(P_ref)
    _, ui = np.unique(np.round(np.c_[P_pre, P_ref] / 2).astype(int), axis=0, return_index=True)
    P_pre, P_ref = P_pre[ui], P_ref[ui]
    keep = np.linalg.norm(P_pre - P_ref, axis=1) < maxd
    return P_pre[keep], P_ref[keep]


def match_stage(uav, ref, H0, dev):
    """第 1 輪：用既有單應矩陣預對位（山區殘差大，maxd 放寬到 80 px；精確剔除交給含 DTM 的相機模型）。"""
    h, w = uav.shape[:2]
    um = uav.copy(); um[int(h * 0.93):, :int(w * 0.46)] = 0
    rh, rw = ref.shape[:2]
    pre = cv2.warpPerspective(um, H0, (rw, rh), flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H0, (rw, rh), flags=cv2.INTER_NEAREST) > 0
    P_pre, P_ref = match_prewarped(pre, mask, ref, dev, 80)
    P_uav = cv2.perspectiveTransform(P_pre[None].astype(np.float32), np.linalg.inv(H0))[0]
    return P_uav.astype(np.float64), P_ref.astype(np.float64)


def project_grid(cam, p, Xg, shape_hw, grid_hw):
    """把地面格點 Xg（N×3）投影到 UAV；回傳 mapx, mapy（grid 形狀）與可用遮罩
    （在畫面內、未被地形遮擋（z-buffer）、局部縮放 ≥ 0.6 UAV px/格、不在授權文字區）。"""
    h, w = shape_hw; gh, gw = grid_hw
    uv, depth = cam.project(p, Xg)
    inside = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
    zb = np.full((h + 1, w + 1), np.inf)
    ui, vi = uv[inside].astype(int).T
    np.minimum.at(zb, (vi, ui), depth[inside])
    vis = np.zeros(len(Xg), bool)
    zi = zb[uv[inside, 1].astype(int), uv[inside, 0].astype(int)]
    vis[inside] = depth[inside] <= zi * 1.01 + 2.0
    mapx = uv[:, 0].reshape(gh, gw).astype(np.float32); mapy = uv[:, 1].reshape(gh, gw).astype(np.float32)
    jx, jy = np.gradient(mapx, axis=1), np.gradient(mapy, axis=0)
    jx2, jy2 = np.gradient(mapx, axis=0), np.gradient(mapy, axis=1)
    scale = np.sqrt(np.abs(jx * jy - jx2 * jy2))
    txt = (mapy > h * 0.93) & (mapx < w * 0.46)
    v = vis.reshape(gh, gw) & (scale > 0.6) & ~txt
    v = cv2.morphologyEx(v.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _c = cv2.connectedComponentsWithStats(v)          # 只留主要連通塊（丟掉遠景的零星細條）
    if n > 1:
        big = st[1:, cv2.CC_STAT_AREA].max()
        v = np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 0.1 * big])
    return mapx, mapy, v > 0


class Cam:
    """針孔 + 徑向 k1。參數 p = [rvec(3), tvec(3), f, k1]；主點固定在影像中心。"""
    def __init__(self, w, h):
        self.cx, self.cy = w / 2, h / 2

    def project(self, p, X):
        R_, _ = cv2.Rodrigues(p[:3])
        Xc = X @ R_.T + p[3:6]
        z = Xc[:, 2]
        x, y = Xc[:, 0] / z, Xc[:, 1] / z
        r2 = x * x + y * y
        s = 1 + p[7] * r2
        return np.c_[p[6] * x * s + self.cx, p[6] * y * s + self.cy], z


def solve_camera(cam, X, uv, w, h):
    """相機參數：f 限制在 0.45–1.0×影像寬（一般空拍機 FOV 60–95°），k1 限 ±0.2；
    以不同 f 起點各做 PnP-RANSAC＋穩健精修，取「殘差 < 5 px 內點最多」者，再放開 f 精修，最後迭代剔除離群。
    （自由 k1／f 在地形起伏小時會互相補償，出現 f 過大 + k1 −1 之類不合物理的解，故加物理範圍。）"""
    lo = np.r_[-np.inf * np.ones(6), 0.45 * w, -0.2]; hi = np.r_[np.inf * np.ones(6), 1.0 * w, 0.2]
    xs = np.r_[np.ones(3), 10 * np.ones(3), w, 0.1]
    best = None
    for f in np.geomspace(0.45 * w, 1.0 * w, 12):
        K = np.array([[f, 0, cam.cx], [0, f, cam.cy], [0, 0, 1.0]])
        ok, rv, tv, inl = cv2.solvePnPRansac(X, uv, K, None, iterationsCount=4000, reprojectionError=8.0,
                                             flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < 8:
            continue
        p = np.r_[rv.ravel(), tv.ravel(), f, 0.0]
        fun = lambda q: (cam.project(q, X)[0] - uv).ravel()
        sol = least_squares(fun, p, loss="soft_l1", f_scale=3.0, x_scale=xs, bounds=(lo, hi))
        res = np.linalg.norm(cam.project(sol.x, X)[0] - uv, axis=1)
        n5 = int((res < 5).sum())
        if best is None or n5 > best[0]:
            best = (n5, sol.x)
    if best is None:
        return None
    p = best[1]
    keep = np.ones(len(X), bool)
    for it in range(4):
        fun = lambda q: (cam.project(q, X[keep])[0] - uv[keep]).ravel()
        p = least_squares(fun, p, loss="soft_l1", f_scale=3.0, x_scale=xs, bounds=(lo, hi)).x
        res = np.linalg.norm(cam.project(p, X)[0] - uv, axis=1)
        keep = (res < max(4.0, 2.5 * np.median(res[keep]))) & (cam.project(p, X)[1] > 0)
    return p, keep


def ground_to_px(cam, p, X):
    return cam.project(p, X)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eid")
    ap.add_argument("--cache", default=str(HERE / "work"))
    ap.add_argument("--z", type=int, default=18)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rounds", type=int, default=2, help="匹配/解算輪數（第 2 輪起用相機+DTM 預對位）")
    ap.add_argument("--publish", action="store_true", help="把正射成果與 meta 寫進 webapp/.../static/uav/<id>/")
    ap.add_argument("--gsd", type=float, default=0.5, help="輸出正射 GSD（公尺）")
    args = ap.parse_args()
    out = Path(args.cache) / args.eid
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((REPO / "webapp/change_detect_viewer/static/uav" / args.eid / "meta.json").read_text(encoding="utf-8"))

    up = out / "uav.jpg"
    if not up.exists():
        img = fetch_uav(args.eid)
        cv2.imwrite(str(up), img, [cv2.IMWRITE_JPEG_QUALITY, 97])
    uav = cv2.imread(str(up))
    h, w = uav.shape[:2]
    print("uav", w, h)

    # 初始位置（來自既有單應矩陣對位的四角）
    cl = np.array(meta.get("corners_lonlat") or meta["corners_lonlat_homography"])          # 左上, 右上, 右下, 左下 對應 UAV 四角（register.rectify 的順序）
    lat0, lon0 = meta["center_lonlat"][1], meta["center_lonlat"][0]
    z = args.z
    ref_path = out / "ref.npz"
    pad_m = 150
    mx = np.array([R.merc(a, b)[0] for a, b in cl]); my = np.array([R.merc(a, b)[1] for a, b in cl])
    n = 2 ** z; world = 2 * math.pi * RE; ts = world / n / 256
    tx0 = int((mx.min() - pad_m / math.cos(math.radians(lat0)) + world / 2) // (ts * 256))
    tx1 = int((mx.max() + pad_m / math.cos(math.radians(lat0)) + world / 2) // (ts * 256))
    ty0 = int((world / 2 - (my.max() + pad_m / math.cos(math.radians(lat0)))) // (ts * 256))
    ty1 = int((world / 2 - (my.min() - pad_m / math.cos(math.radians(lat0)))) // (ts * 256))
    if ref_path.exists():
        d = np.load(ref_path); ref, x0, y0 = d["ref"], int(d["x0"]), int(d["y0"])
    else:
        import time
        ref = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256, 3), np.uint8)
        for j in range(ty1 - ty0 + 1):
            for i in range(tx1 - tx0 + 1):
                im = cv2.imdecode(np.frombuffer(R.fetch(R.SRC["esri"].format(z=z, x=tx0 + i, y=ty0 + j)), np.uint8), 1)
                if im is not None and im.shape[:2] == (256, 256):
                    ref[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256] = im
        x0, y0 = tx0, ty0
        np.savez_compressed(ref_path, ref=ref, x0=x0, y0=y0)
    print("ref", ref.shape, "tiles", x0, y0)

    def merc2px(mx_, my_):
        return (mx_ + world / 2) / ts - x0 * 256, (world / 2 - my_) / ts - y0 * 256
    # 初始單應矩陣：UAV 四角 → ref 像素
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([merc2px(*R.merc(a, b)) for a, b in cl]).reshape(-1, 2)
    H0 = cv2.getPerspectiveTransform(src, dst)

    mpath = out / "matches.npz"
    if mpath.exists():
        d = np.load(mpath); P_uav, P_ref = d["P_uav"], d["P_ref"]
    else:
        P_uav, P_ref = match_stage(uav, ref, H0, args.device)
        np.savez(mpath, P_uav=P_uav, P_ref=P_ref)
    print("matches", len(P_uav))

    dtm = DTM.Dtm20Source()
    ox = oy = zmean = None

    def ground(P):
        """底圖像素 → 區域座標（公尺）＋ DTM 高程。第一次呼叫時定下原點（之後各輪沿用，座標系一致）。"""
        nonlocal ox, oy, zmean
        pm_x = (P[:, 0] + x0 * 256) * ts - world / 2
        pm_y = world / 2 - (P[:, 1] + y0 * 256) * ts
        lon = np.degrees(pm_x / RE); lat = np.degrees(2 * np.arctan(np.exp(pm_y / RE)) - math.pi / 2)
        zg = dtm.sample_points(lon, lat)
        gx, gy = merc_to_local(pm_x, pm_y, lat0)
        if ox is None:
            ox, oy, zmean = gx.mean(), gy.mean(), zg.mean()
        return np.c_[gx - ox, gy - oy, zg - zmean], zg

    X, zg = ground(P_ref)
    print(f"DTM relief at matches: {zg.min():.0f}-{zg.max():.0f} m")
    cam = Cam(w, h)
    # 先用全部點解一次相機，以其殘差清掉誤配，再在乾淨集合內做留出檢核。
    # 偏差說明：清理用相機模型，對相機略有利；但單應矩陣若在同一乾淨集合仍明顯較差，才算真實差距。
    sol0 = solve_camera(cam, X, P_uav, w, h)
    if sol0 is None:
        sys.exit("camera solve failed")
    clean = np.where(sol0[1])[0]
    print("round 1: clean %d / %d" % (len(clean), len(X)))
    # 第 2 輪起：用上一輪的相機+DTM 把 UAV 重新 warp 到底圖像素格（殘差只剩數像素），LoFTR 在更接近的影像對上匹配，
    # maxd 收緊到 20 px，得到更密、更準的對應點。
    rh, rw = ref.shape[:2]
    gj, gi = np.meshgrid(np.arange(rw) + 0.5, np.arange(rh) + 0.5)
    Xgrid, _z = ground(np.c_[gj.ravel(), gi.ravel()])
    um = uav.copy()
    for rnd in range(2, args.rounds + 1):
        pc = solve_camera(cam, X[clean], P_uav[clean], w, h)[0]
        mapx, mapy, vmask = project_grid(cam, pc, Xgrid, (h, w), (rh, rw))
        pre = cv2.remap(um, mapx, mapy, cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
        P_pre, P_ref2 = match_prewarped(pre, vmask, ref, args.device, 20)
        keepv = vmask[np.clip(P_pre[:, 1].astype(int), 0, rh - 1), np.clip(P_pre[:, 0].astype(int), 0, rw - 1)]
        P_pre, P_ref2 = P_pre[keepv], P_ref2[keepv]
        from scipy.ndimage import map_coordinates
        P_uav = np.c_[map_coordinates(mapx, [P_pre[:, 1], P_pre[:, 0]], order=1), map_coordinates(mapy, [P_pre[:, 1], P_pre[:, 0]], order=1)].astype(np.float64)
        P_ref = P_ref2
        X, zg = ground(P_ref)
        sol0 = solve_camera(cam, X, P_uav, w, h)
        clean = np.where(sol0[1])[0]
        print("round %d: matches %d, clean %d" % (rnd, len(X), len(clean)))
    np.savez(out / "matches_final.npz", P_uav=P_uav, P_ref=P_ref, clean=clean)
    rng = np.random.default_rng(7)
    idx = rng.permutation(clean); ntest = int(len(idx) * 0.3)
    te, tr = idx[:ntest], idx[ntest:]
    print("clean %d / %d; train %d, holdout %d" % (len(clean), len(X), len(tr), len(te)))
    sol = solve_camera(cam, X[tr], P_uav[tr], w, h)
    if sol is None:
        sys.exit("相機姿態解算失敗")
    p, keep = sol
    print("camera f=%.0f px (%.2f×width)  k1=%.3f  inliers %d/%d" % (p[6], p[6] / w, p[7], keep.sum(), len(tr)))

    # 以 UAV 像素的地面尺度把像素誤差換成公尺：中心處 1 px ≈ GSD
    Rm, _ = cv2.Rodrigues(p[:3]); C = -Rm.T @ p[3:6]
    print("camera pos (local m)", np.round(C, 1), "AGL~%.0f m" % (C[2] - np.mean(X[:, 2])))

    def err_cam(ix):
        uvp, _z = cam.project(p, X[ix])
        # 地面誤差：在地面把 UAV 觀測像素射線求交 → 與 X 的水平差。簡化：像素誤差 × 該點的地面 GSD
        # （GSD = 該點距離 / f / cos 傾角近似）
        dist = np.linalg.norm(X[ix] - C, axis=1)
        return np.linalg.norm(uvp - P_uav[ix], axis=1) * dist / p[6]

    # 基準：單應矩陣（只用訓練點擬合），在同一批留出點上量測地面誤差
    Hh, _ = cv2.findHomography(P_uav[tr], P_ref[tr], cv2.USAC_MAGSAC, 4.0, maxIters=20000, confidence=0.9999)
    pr = cv2.perspectiveTransform(P_uav[te][None].astype(np.float32), Hh)[0]
    e_h = np.linalg.norm(pr - P_ref[te], axis=1) * ts * math.cos(math.radians(lat0))
    e_c = err_cam(te)
    # 兩者比較需排除對應點本身是誤配的離群：用 80 百分位以內
    def stat(e):
        e = np.sort(e); q = e[: int(len(e) * 0.9)]
        return {"rmse_m": round(float(np.sqrt((e ** 2).mean())), 2), "median_m": round(float(np.median(e)), 2),
                "rmse_p90_m": round(float(np.sqrt((q ** 2).mean())), 2)}
    rep = {"n_match": int(len(X)), "n_holdout": int(len(te)), "homography": stat(e_h), "camera_dtm": stat(e_c),
           "f_px": float(p[6]), "k1": float(p[7]), "cam_height_agl_m": float(C[2] - np.mean(X[:, 2])),
           "dtm_relief_m": [float(zg.min()), float(zg.max())], "rounds": args.rounds, "n_clean": int(len(clean))}
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    # 正式相機：用全部乾淨點重解
    pf, kf = solve_camera(cam, X[clean], P_uav[clean], w, h)
    np.savez(out / "camera.npz", p=pf, ox=ox, oy=oy, zmean=zg.mean(), lat0=lat0)
    ortho_stage(args, out, uav, cam, pf, meta, ox, oy, zmean, lat0, dtm, ref, x0, y0, ts, world)


def ortho_stage(args, out, uav, cam, p, meta, ox, oy, zmean, lat0, dtm, ref, x0, y0, ts, world):
    """沿相機光線與 DTM 求交：對輸出網格每格取 DTM 高程 → 投影回 UAV 影像取色；z-buffer 排除被地形遮擋的格。"""
    h, w = uav.shape[:2]
    bm = meta["bounds_merc"]
    pix = args.gsd / math.cos(math.radians(lat0))                       # 輸出像素的 Mercator 公尺
    ow, oh = int((bm["maxx"] - bm["minx"]) / pix), int((bm["maxy"] - bm["miny"]) / pix)
    mx = bm["minx"] + (np.arange(ow) + 0.5) * pix
    my = bm["maxy"] - (np.arange(oh) + 0.5) * pix
    MX, MY = np.meshgrid(mx, my)
    lon = np.degrees(MX / RE); lat = np.degrees(2 * np.arctan(np.exp(MY / RE)) - math.pi / 2)
    Z = dtm.sample_points(lon, lat)
    Z = np.where(np.isnan(Z), np.nanmedian(Z), Z)
    gx, gy = merc_to_local(MX, MY, lat0)
    Xg = np.stack([gx - ox, gy - oy, Z - zmean], -1).reshape(-1, 3)
    mapx, mapy, v = project_grid(cam, p, Xg, (h, w), (oh, ow))
    warped = cv2.remap(uav, mapx, mapy, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    print("ortho %dx%d, visible %.0f%%" % (ow, oh, 100 * v.mean()))
    rgba = np.dstack([warped, (v * 255).astype(np.uint8)])
    cv2.imwrite(str(out / "ortho.png"), rgba)
    if args.publish:
        publish(args, out, warped, v, meta, ow, oh, bm, pix, lat0, p)
    # 與底圖疊合檢視（底圖重採樣到同網格）
    rx = ((MX + world / 2) / ts - x0 * 256).astype(np.float32); ry = ((world / 2 - MY) / ts - y0 * 256).astype(np.float32)
    rr = cv2.remap(ref, rx, ry, cv2.INTER_LINEAR)
    ov = rr.copy(); ov[v] = (0.5 * warped[v] + 0.5 * rr[v]).astype(np.uint8)
    cv2.imwrite(str(out / "overlay.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(str(out / "ref_same_grid.jpg"), rr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    # 與舊單應矩陣成果比較：在兩者都有值的區域，量「與底圖的梯度相關」（不依賴匹配點的獨立指標）
    od = REPO / "webapp/change_detect_viewer/static/uav" / args.eid
    old_fp = next(od.glob("rect.*"), None)
    if old_fp is not None:
        old = cv2.imread(str(old_fp), cv2.IMREAD_UNCHANGED)
        om = old[..., 3] > 0 if old.ndim == 3 and old.shape[2] == 4 else (old[..., :3].sum(2) > 0)
        old = cv2.resize(old[..., :3], (ow, oh), interpolation=cv2.INTER_AREA)
        om = cv2.resize(om.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST) > 0
        both = v & om
        def gcorr(img):
            g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 2.0).astype(np.float32)
            gm = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
            return gm
        gr = gcorr(rr)
        res = {}
        for name, im in (("camera_dtm", warped), ("homography_old", old)):
            gi = gcorr(im)
            res[name] = round(float(np.corrcoef(gi[both], gr[both])[0, 1]), 4)
        res["overlap_px"] = int(both.sum())
        print("gradient corr vs basemap (higher = better aligned):", res)
        cv2.imwrite(str(out / "overlay_old.jpg"), np.where(om[..., None], (0.5 * old + 0.5 * rr).astype(np.uint8), rr), [cv2.IMWRITE_JPEG_QUALITY, 88])
        rep = json.loads((out / "report.json").read_text()); rep["gradient_corr"] = res
        (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


def publish(args, out, warped, v, meta, ow, oh, bm, pix, lat0, p):
    """裁到有效範圍，存 RGBA WebP，更新 meta.json（方法、品質、範圍）。"""
    ys, xs = np.where(v)
    x0_, x1_, y0_, y1_ = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    rgba = np.dstack([warped, (v * 255).astype(np.uint8)])[y0_:y1_, x0_:x1_]
    d = REPO / "webapp/change_detect_viewer/static/uav" / args.eid
    cv2.imwrite(str(d / "rect.webp"), rgba, [cv2.IMWRITE_WEBP_QUALITY, 90])
    old = d / "rect.png"
    if old.exists():
        old.unlink()
    minx, maxy = bm["minx"] + x0_ * pix, bm["maxy"] - y0_ * pix
    maxx, miny = bm["minx"] + x1_ * pix, bm["maxy"] - y1_ * pix
    un = lambda mx, my: (math.degrees(mx / RE), math.degrees(2 * math.atan(math.exp(my / RE)) - math.pi / 2))
    (w_, s_), (e_, n_) = un(minx, miny), un(maxx, maxy)
    rep = json.loads((out / "report.json").read_text())
    c, hg = rep["camera_dtm"], rep["homography"]
    meta["corners_lonlat_homography"] = meta.pop("corners_lonlat", None) or meta.get("corners_lonlat_homography")   # 初始單應矩陣的四角，重跑時當初始位置
    meta.update({
        "image": "rect.webp",
        "bounds_merc": {"minx": minx, "maxy": maxy, "maxx": maxx, "miny": miny},
        "bounds_lonlat": {"west": w_, "south": s_, "east": e_, "north": n_},
        "center_lonlat": [round((w_ + e_) / 2, 6), round((s_ + n_) / 2, 6)],
        "method": "LoFTR（學習式匹配，兩輪）→ 相機姿態解算（PnP + 穩健最小二乘，焦距／k1 限物理範圍）→ 沿相機光線與本地 20 m DTM 求交的正射重採樣；"
                  "底圖 ESRI World Imagery z%d；UAV 無 EXIF，純影像內容對位" % args.z,
        "quality": {
            "joint_inliers": int(rep["n_clean"]),
            "fit_rmse_m": None,
            "holdout_rmse_m": c["rmse_m"],
            "holdout_median_m": c["median_m"],
            "homography_holdout_median_m": hg["median_m"],
            "gsd_m_per_px": args.gsd,
            "camera": {"f_px": round(rep["f_px"], 1), "k1": round(rep["k1"], 3), "agl_m": round(rep["cam_height_agl_m"])},
            "note": "留出點（佔 30%%）地面誤差 RMSE %.1f m、中位數 %.1f m（同批點上單應矩陣平面假設的中位數為 %.1f m；該批點由相機模型篩選，對單應矩陣偏不利，僅供參考）。"
                    "已用 DTM 與相機姿態處理地形視差，但限制仍在：DTM 為 20 m 解析度、遮擋以 DTM 近似（樹冠、建物未建模）、"
                    "底圖本身有數公尺定位誤差、焦距與相機高度在地形起伏小時互相補償（無 EXIF）；遠景、被山遮擋與拉伸過度的區域已裁掉（透明）。"
                    "屬『較精確的地形糾正』，非測量級正射。" % (c["rmse_m"], c["median_m"], hg["median_m"]),
        },
    })
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("published ->", d, "bounds", meta["bounds_lonlat"])


if __name__ == "__main__":
    main()
