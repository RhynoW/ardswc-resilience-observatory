"""批次 UAV 對位：對候選事件逐一嘗試 LoFTR 粗對位（掃旋轉角）→ 精對位（多窗口聯合單應）
→ 留出檢核，只有通過品質門檻的才輸出成果。未通過一律如實記錄失敗原因，不降標準湊數。

品質門檻（accept 條件，全部要滿足）：
  joint_inliers >= 30、holdout_rmse <= 8 m、四角凸、足跡面積 2–400 ha、
  內點在 UAV 畫面的覆蓋跨度 >= 35%（避免全部集中在一小塊）、與已接受樣本至少相距 1.5 km。
"""
import json, math, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2, numpy as np, torch, kornia.feature as KF
import register as R
import thermal_guard as TG

HERE = Path(__file__).parent
DEST = Path("F:/GitHub/ardswc-resilience-observatory/webapp/change_detect_viewer/static/uav")
WANT = int(sys.argv[1]) if len(sys.argv) > 1 else 9
MIN_SEP_KM = float(os.environ.get("UAV_MIN_SEP_KM", "1.5"))
HOLD_MAX = float(os.environ.get("UAV_HOLD_MAX", "8"))      # 留出檢核 RMSE 上限（m）；2026 擴充時放寬到 10

dev = os.environ.get("UAV_DEVICE", "cpu")          # 本機 GPU 曾當機兩次，預設 CPU
if dev == "cpu":
    torch.set_num_threads(6)
print("device", dev, flush=True)
matcher = KF.LoFTR(pretrained="outdoor").to(dev).eval()
clahe = cv2.createCLAHE(2.5, (8, 8))
prep = lambda im: clahe.apply(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
to_t = lambda g: torch.from_numpy(np.ascontiguousarray(g))[None, None].float().to(dev) / 255.0
ESRI = R.SRC[os.environ.get("UAV_REF", "esri")]          # 參考底圖：esri（預設）或 nlsc（PHOTO2025，較新，災後地貌變化大的 2025–2026 樣本可試）


def fetch_uav(eid):
    url = f"https://photo.ardswc.gov.tw/api/Download/{eid}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 ardswc-uav-register"})
    body = urllib.request.urlopen(req, timeout=60).read()
    img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    if img is None or min(img.shape[:2]) < 500:
        return None
    return img


def mosaic_par(lat, lon, z, half):
    """平行抓 ESRI 圖磚拼接；回傳 (BGR, x0, y0)。"""
    cx, cy = R.tile_xy(lat, lon, z)
    x0, y0 = int(cx) - half, int(cy) - half
    n = 2 * half + 1
    canvas = np.zeros((n * 256, n * 256, 3), np.uint8)

    def one(ij):
        i, j = ij
        for _ in range(3):
            try:
                im = cv2.imdecode(np.frombuffer(R.fetch(ESRI.format(z=z, x=x0 + i, y=y0 + j)), np.uint8), 1)
                if im is not None and im.shape[:2] == (256, 256):
                    return i, j, im
            except Exception:
                time.sleep(0.4)
        return i, j, None

    with ThreadPoolExecutor(12) as ex:
        for i, j, im in ex.map(one, [(i, j) for j in range(n) for i in range(n)]):
            if im is not None:
                canvas[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256] = im
    return canvas, x0, y0


def rot_img(im, deg):
    h, w = im.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(im, M, (nw, nh), flags=cv2.INTER_AREA), M


def crop8(g, y, x, win):
    sub = g[y:y + win, x:x + win]
    return sub[:(sub.shape[0] // 8) * 8, :(sub.shape[1] // 8) * 8]


def match(uav_g, ref_g, thr=0.5):
    TG.wait_cool("(match)", min_interval=0)
    t0 = time.time()
    with torch.no_grad():
        out = matcher({"image0": to_t(uav_g), "image1": to_t(ref_g)})
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    TG.duty_sleep(time.time() - t0)                      # 工作週期限流：每次推論後休息，避免溫度瞬間衝高
    keep = out["confidence"].cpu().numpy() > thr
    if keep.sum() < 8:
        return None, None
    return out["keypoints0"].cpu().numpy()[keep], out["keypoints1"].cpu().numpy()[keep]


def try_one(rec, log):
    eid = rec["EventID"]
    lat0, lon0 = rec["Lat"], rec["Lng"]
    uav = fetch_uav(eid)
    if uav is None:
        return {"id": eid, "ok": False, "reason": "影像下載失敗或尺寸過小"}
    uav_m = uav.copy()
    hh, ww = uav.shape[:2]
    uav_m[int(hh * 0.93):, :int(ww * 0.46)] = 0          # 左下角授權文字
    uav_pg = prep(uav_m)

    Z, HALF = 17, 2                                      # 1280 px ≈ 1.4 km（API 座標為事後推估，實測偏移 ~250 m）
    ref, x0, y0 = mosaic_par(lat0, lon0, Z, HALF)
    if ref.mean() < 8:
        return {"id": eid, "ok": False, "reason": "底圖無資料"}
    ts_merc = R.px_to_merc(0, 0, x0, y0, Z)[2]
    ts = ts_merc * math.cos(math.radians(lat0))          # 底圖地面 m/px
    ref_g = prep(ref)

    # ── 粗對位：整張底圖降到 832，掃 12 個旋轉角，找大致方位
    COARSE = 640
    fc = COARSE / ref_g.shape[0]
    ref_c = cv2.resize(ref_g, (COARSE, COARSE), interpolation=cv2.INTER_AREA)
    ref_c = ref_c[:(COARSE // 8) * 8, :(COARSE // 8) * 8]
    gsd_c = ts / fc
    best_rot, best_n, best_g = None, 0, None
    for g in (0.12, 0.25, 0.5):                          # UAV 飛行高度未知：粗掃三個地面解析度
        small = cv2.resize(uav_pg, None, fx=g / gsd_c, fy=g / gsd_c, interpolation=cv2.INTER_AREA)
        if min(small.shape) < 64:
            continue
        f0 = min(1.0, 640 / max(small.shape))            # UAV 端上限 640 px（CPU 時間）
        if f0 < 1.0:
            continue                                      # 比粗底圖還大 → 此尺度不合理，跳過
        for deg in range(0, 360, 30):
            rimg, _ = rot_img(small, deg)
            rimg = rimg[:(rimg.shape[0] // 8) * 8, :(rimg.shape[1] // 8) * 8]
            p0, p1 = match(rimg, ref_c)
            if p0 is None:
                continue
            Hc, inl = cv2.findHomography(p0, p1, cv2.USAC_MAGSAC, 4.0, maxIters=4000, confidence=0.999)
            n = 0 if Hc is None else int(inl.sum())
            if n > best_n:
                best_n, best_rot, best_g = n, deg, g
    log(f"  粗對位 best rot {best_rot} gsd {best_g} inliers {best_n}")
    if best_n < 12:
        return {"id": eid, "ok": False, "reason": f"粗對位失敗（最佳內點 {best_n} < 12）"}

    # ── 精對位：最佳角 ±10°，兩個尺度，2×2 窗口聯合
    WIN = 832
    offs = [0, ref_g.shape[0] - WIN]
    best = None
    for gsd in (best_g * 0.8, best_g * 1.25):
        s = gsd / ts
        small = cv2.resize(uav_pg, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        for deg in (best_rot - 10, best_rot, best_rot + 10):
            rimg, M = rot_img(small, deg)
            f = min(1.0, WIN / max(rimg.shape))
            if f < 1.0:
                rimg = cv2.resize(rimg, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            rimg = rimg[:(rimg.shape[0] // 8) * 8, :(rimg.shape[1] // 8) * 8]
            Minv = cv2.invertAffineTransform(M)
            P_u, P_r, WID = [], [], []
            for wi, (wy, wx) in enumerate([(a, b) for a in offs for b in offs]):
                p0, p1 = match(rimg, crop8(ref_g, wy, wx, WIN))
                if p0 is None:
                    continue
                q = (Minv @ np.c_[p0 / f, np.ones(len(p0))].T).T / s     # → UAV 全解析像素
                P_u.append(q); P_r.append(p1 + np.array([wx, wy])); WID += [wi] * len(p0)
            if not P_u:
                continue
            P_u = np.vstack(P_u).astype(np.float32); P_r = np.vstack(P_r).astype(np.float32)
            WID = np.array(WID)
            _, ui = np.unique(np.round(np.c_[P_u, P_r] / 2).astype(int), axis=0, return_index=True)
            P_u, P_r, WID = P_u[ui], P_r[ui], WID[ui]
            if len(P_u) < 20:
                continue
            H, inl = cv2.findHomography(P_u, P_r, cv2.USAC_MAGSAC, 3.0, maxIters=20000, confidence=0.9999)
            if H is None:
                continue
            inl = inl.ravel().astype(bool)
            n = int(inl.sum())
            if best is None or n > best["n"]:
                best = {"n": n, "H": H, "P_u": P_u, "P_r": P_r, "inl": inl, "gsd": gsd, "rot": deg}
    if best is None or best["n"] < 30:
        return {"id": eid, "ok": False, "reason": f"精對位內點不足（{0 if best is None else best['n']} < 30）"}

    H, P_u, P_r, inl = best["H"], best["P_u"], best["P_r"], best["inl"]
    proj = cv2.perspectiveTransform(P_u[inl][None], H)[0]
    fit = float(np.sqrt((np.linalg.norm(proj - P_r[inl], axis=1) ** 2).mean()) * ts)
    rng = np.random.default_rng(1)
    idx = np.where(inl)[0]; rng.shuffle(idx)
    k = max(4, int(len(idx) * 0.7)); tr, te = idx[:k], idx[k:]
    if len(te) < 4:
        return {"id": eid, "ok": False, "reason": "內點太少無法留出檢核"}
    H2, _ = cv2.findHomography(P_u[tr], P_r[tr], 0)
    if H2 is None:
        return {"id": eid, "ok": False, "reason": "留出檢核擬合失敗"}
    e2 = np.linalg.norm(cv2.perspectiveTransform(P_u[te][None], H2)[0] - P_r[te], axis=1) * ts
    hold = float(np.sqrt((e2 ** 2).mean()))

    # 只保留「有匹配點支撐」的區域：內點凸包外擴 6% 影像對角線。斜拍影像的遠景常沒有內點，
    # 單應矩陣會把它外推拉長到很遠的地方（無法驗證），所以不顯示。
    pts = P_u[inl]
    hull = cv2.convexHull(pts.astype(np.float32))
    buf = 0.06 * math.hypot(ww, hh)
    hmask = np.zeros((hh, ww), np.uint8)
    cv2.fillConvexPoly(hmask, hull.astype(np.int32), 255)
    hmask = cv2.dilate(hmask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2 * buf) | 1, int(2 * buf) | 1)))
    cover = float((hmask > 0).mean())
    cnts, _ = cv2.findContours(hmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    poly = max(cnts, key=cv2.contourArea).astype(np.float32)
    cr = cv2.perspectiveTransform(poly.reshape(1, -1, 2), H)[0]          # 支撐區在底圖的輪廓
    area_ha = float(abs(cv2.contourArea(cr)) * ts * ts / 1e4)
    # 幾何合理性只檢查「支撐區」：斜拍時整張畫面的外推角點可能跨過地平線而翻折（那部分已裁除），
    # 真正要求的是支撐區內每一點的投影分母 > 0（沒跨地平線），且映射後的外廓仍是凸的。
    hp = np.c_[poly.reshape(-1, 2), np.ones(len(poly.reshape(-1, 2)))] @ H.T
    hull_ref = cv2.perspectiveTransform(hull.reshape(1, -1, 2).astype(np.float32), H)[0]
    convex = bool((hp[:, 2] > 0).all() == (hp[:, 2] > 0).any() and cv2.isContourConvex(hull_ref.astype(np.float32)))
    c = cv2.perspectiveTransform(np.float32([[pts.mean(0)]]), H)[0][0]
    clon, clat = R.unmerc(*R.px_to_merc(c[0], c[1], x0, y0, Z)[:2])
    st = {"id": eid, "n": best["n"], "fit_rmse_m": round(fit, 2), "holdout_rmse_m": round(hold, 2),
          "cover": round(cover, 2), "area_ha": round(area_ha, 1), "convex": convex,
          "rot": best["rot"], "gsd_guess": round(best["gsd"], 3), "center": [round(clon, 6), round(clat, 6)],
          "dist_from_api_m": round(R.hav(clon, clat, lon0, lat0))}
    log(f"  精對位 {json.dumps(st, ensure_ascii=False)}")
    if hold > HOLD_MAX:
        return {**st, "ok": False, "reason": f"留出檢核 RMSE {hold:.1f} m > {HOLD_MAX:g} m"}
    if not convex or not (1.0 <= area_ha <= 150.0):
        return {**st, "ok": False, "reason": f"足跡不合理（凸={convex}、支撐區面積 {area_ha:.1f} ha）"}
    if cover < 0.15:
        return {**st, "ok": False, "reason": f"有匹配點支撐的區域太小（{cover:.0%} < 15% 畫面）"}

    # ── 糾正輸出（RGBA，範圍外透明）
    UP = 4
    x_min, y_min = np.floor(cr.min(0)).astype(int); x_max, y_max = np.ceil(cr.max(0)).astype(int)
    S = np.array([[UP, 0, -x_min * UP], [0, UP, -y_min * UP], [0, 0, 1]], np.float64)
    ow, oh = int((x_max - x_min) * UP), int((y_max - y_min) * UP)
    if max(ow, oh) > 6000:
        sc = 6000 / max(ow, oh)
        S = np.array([[UP * sc, 0, -x_min * UP * sc], [0, UP * sc, -y_min * UP * sc], [0, 0, 1]], np.float64)
        ow, oh = int(ow * sc), int(oh * sc)
    warped = cv2.warpPerspective(uav, S @ H, (ow, oh), flags=cv2.INTER_LANCZOS4)
    valid = cv2.warpPerspective(hmask, S @ H, (ow, oh), flags=cv2.INTER_NEAREST)
    valid = cv2.erode(valid, np.ones((5, 5), np.uint8))
    rgba = np.dstack([warped, valid])
    while rgba.shape[1] > 2400 or rgba.shape[0] > 2400:
        rgba = cv2.resize(rgba, (rgba.shape[1] // 2, rgba.shape[0] // 2), interpolation=cv2.INTER_AREA)
    d = DEST / eid
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "rect.png"), rgba, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    mx0, my0, _ = R.px_to_merc(x_min, y_min, x0, y0, Z)
    mx1, my1, _ = R.px_to_merc(x_max, y_max, x0, y0, Z)
    west, north = R.unmerc(mx0, my0); east, south = R.unmerc(mx1, my1)
    out_gsd = round((x_max - x_min) * ts / rgba.shape[1], 2)
    date = (rec.get("PhotoDate") or "")[:10]
    meta = {
        "id": eid,
        "title": f"{rec['County']}{rec['Town']}{rec.get('Vill') or ''}（UAV 空拍，{date}）"
                 + (f"　{rec['DisasterName']}" if rec.get("DisasterName") else ""),
        "description": (rec.get("Description") or "").strip(),
        "source_page": f"https://photo.ardswc.gov.tw/Repository/ViewEvent/{eid}",
        "attribution": f"{rec.get('Source') or '水保署歷史影像平台'}（照片授權方式：姓名標示，CC-BY）；"
                       "農業部農村發展及水土保持署歷史影像平台",
        "image": "rect.png",
        "bounds_lonlat": {"west": west, "south": south, "east": east, "north": north},
        "support_polygon_lonlat": [list(R.unmerc(*R.px_to_merc(px, py, x0, y0, Z)[:2])) for px, py in cr[::max(1, len(cr) // 60)]],
        "center_lonlat": [round(clon, 6), round(clat, 6)],
        "method": "LoFTR（學習式匹配）+ MAGSAC 單應矩陣；底圖 Esri World Imagery z17；"
                  "UAV 無 EXIF，純影像內容對位（粗掃旋轉角 → 多窗口聯合平差）",
        "quality": {
            "joint_inliers": best["n"], "fit_rmse_m": round(fit, 2), "holdout_rmse_m": round(hold, 2),
            "support_cover": round(cover, 2), "footprint_ha": round(area_ha, 1),
            "gsd_m_per_px": out_gsd, "dist_from_api_coord_m": st["dist_from_api_m"],
            "note": f"留出檢核點 RMSE {hold:.1f} m（底圖解析度 {ts:.2f} m/px）。只顯示有匹配點支撐的區域"
                    f"（原畫面 {cover:.0%}），斜拍遠景等外推部分已裁除。單應矩陣假設地面近似平面，"
                    "起伏地形處會有數公尺重影；非嚴格正射，真正正射需 DSM + 相機姿態。",
        },
    }
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return {**st, "ok": True, "png_mb": round((d / "rect.png").stat().st_size / 1e6, 2), "title": meta["title"]}


def main():
    cands = json.load(open(HERE / "batch_cands.json", encoding="utf-8"))
    if len(sys.argv) > 2:                                # 目視挑選的近垂直候選順序（挑選≠對位；對位全自動＋品質門檻）
        cands = [cands[int(i)] for i in sys.argv[2].split(",")]
    logf = open(HERE / "batch.log", "a", encoding="utf-8")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    done, results = [], []
    accepted_pts = []
    for mf in DEST.glob("*/meta.json"):                 # 已完成的樣本：新樣本不與之重複
        accepted_pts.append(tuple(json.loads(mf.read_text(encoding="utf-8"))["center_lonlat"]))
    for i, rec in enumerate(cands, 1):
        if len(done) >= WANT:
            break
        if any(R.hav(rec["Lng"], rec["Lat"], *p) < MIN_SEP_KM * 1000 for p in accepted_pts):
            continue
        log(f"[{i}/{len(cands)}] {rec['EventID']} {rec['County']}{rec['Town']} "
            f"{rec['Lat']:.5f},{rec['Lng']:.5f} {rec.get('Source')}")
        t = time.time()
        try:
            r = try_one(rec, log)
        except Exception as e:                       # 單一候選失敗不中斷整批
            r = {"id": rec["EventID"], "ok": False, "reason": f"{type(e).__name__}: {e}"}
        r["secs"] = round(time.time() - t)
        results.append(r)
        if r.get("ok"):
            done.append(r["id"]); accepted_pts.append(tuple(r["center"]))
            log(f"  ✔ 接受（{len(done)}/{WANT}）{r.get('title')} {r['secs']}s")
        else:
            log(f"  ✘ {r.get('reason')} {r['secs']}s")
        json.dump({"accepted": done, "results": results}, open(HERE / "batch_results.json", "w"),
                  ensure_ascii=False, indent=1)
    log(f"完成：接受 {len(done)} / 嘗試 {len(results)}")


if __name__ == "__main__":
    main()
