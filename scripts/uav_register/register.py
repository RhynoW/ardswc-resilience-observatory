"""UAV 影像自動對位（SIFT + RANSAC 單應矩陣）與糾正（warp 到北在上的 EPSG:3857 網格）。
以兩個獨立底圖（NLSC PHOTO2025、ESRI World Imagery）各對位一次，互相驗證。"""
import io, json, math, sys, urllib.request
from pathlib import Path
import cv2, numpy as np
from PIL import Image

OUT = Path(__file__).parent
R = 6378137.0
LAT, LON = 24.20942, 121.6717          # 頁面標示的（事後推估）座標，只當搜尋中心
SRC = {
    "nlsc": "https://wmts.nlsc.gov.tw/wmts/PHOTO2025/default/GoogleMapsCompatible/{z}/{y}/{x}",
    "esri": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
}

def tile_xy(lat, lon, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y

def merc(lon, lat):
    return R * math.radians(lon), R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

def unmerc(x, y):
    return math.degrees(x / R), math.degrees(2 * math.atan(math.exp(y / R)) - math.pi / 2)

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 uav-register-test"})
    return urllib.request.urlopen(req, timeout=40).read()

def mosaic(src, z, half_tiles):
    cx, cy = tile_xy(LAT, LON, z)
    x0, y0 = int(cx) - half_tiles, int(cy) - half_tiles
    n = 2 * half_tiles + 1
    canvas = np.zeros((n * 256, n * 256, 3), np.uint8)
    for j in range(n):
        for i in range(n):
            t = fetch(SRC[src].format(z=z, x=x0 + i, y=y0 + j))
            im = cv2.imdecode(np.frombuffer(t, np.uint8), 1)
            if im is not None and im.shape[:2] == (256, 256):
                canvas[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256] = im
    return canvas, x0, y0

def px_to_merc(px, py, x0, y0, z):
    """mosaic 像素 → EPSG:3857（像素左上角座標系）。"""
    world = 2 * math.pi * R
    ts = world / 2 ** z / 256          # 每像素公尺
    return (x0 * 256 + px) * ts - world / 2, world / 2 - (y0 * 256 + py) * ts, ts

def register(uav_bgr, ref_bgr, uav_scale):
    """回傳 H（UAV 全解析像素 → 底圖 mosaic 像素）與統計。"""
    small = cv2.resize(uav_bgr, None, fx=uav_scale, fy=uav_scale, interpolation=cv2.INTER_AREA)
    mask = np.full(small.shape[:2], 255, np.uint8)
    h, w = small.shape[:2]
    mask[int(h * 0.93):, :int(w * 0.46)] = 0          # 左下角授權文字不當特徵
    sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.02)
    g = lambda im: cv2.createCLAHE(3.0, (8, 8)).apply(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    k1, d1 = sift.detectAndCompute(g(small), mask)
    k2, d2 = sift.detectAndCompute(g(ref_bgr), None)
    if d1 is None or d2 is None:
        return None, {"error": "no features"}
    m = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [a for a, b in (p for p in m if len(p) == 2) if a.distance < 0.8 * b.distance]
    st = {"kp_uav": len(k1), "kp_ref": len(k2), "good": len(good)}
    if len(good) < 12:
        return None, st
    p1 = np.float32([k1[a.queryIdx].pt for a in good]) / uav_scale     # 回到全解析
    p2 = np.float32([k2[a.trainIdx].pt for a in good])
    H, inl = cv2.findHomography(p1, p2, cv2.USAC_MAGSAC, 6.0, maxIters=10000, confidence=0.999)
    if H is None:
        return None, st
    inl = inl.ravel().astype(bool)
    proj = cv2.perspectiveTransform(p1[inl][None], H)[0]
    err = np.linalg.norm(proj - p2[inl], axis=1)
    hs, ws = uav_bgr.shape[:2]
    pts = p1[inl]
    spread = float(np.ptp(pts[:, 0]) / ws * np.ptp(pts[:, 1]) / hs)      # 內點覆蓋畫面比例（0–1）
    st.update(inliers=int(inl.sum()), rmse_px=float(np.sqrt((err ** 2).mean())), median_px=float(np.median(err)),
              spread=round(spread, 3))
    return H, st

def rectify(uav_bgr, H, z, x0, y0, ref_bgr, tag, out_ppm=None):
    """把 UAV 影像 warp 進底圖 mosaic 座標（北在上），存 PNG + 世界檔 + 疊圖。"""
    h, w = uav_bgr.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    c_ref = cv2.perspectiveTransform(corners[None], H)[0]
    x_min, y_min = np.floor(c_ref.min(0)).astype(int); x_max, y_max = np.ceil(c_ref.max(0)).astype(int)
    pad = 0
    ow, oh = int(x_max - x_min), int(y_max - y_min)
    T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], np.float64)
    warped = cv2.warpPerspective(uav_bgr, T @ H, (ow, oh), flags=cv2.INTER_LANCZOS4,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    valid = cv2.warpPerspective(np.full((h, w), 255, np.uint8), T @ H, (ow, oh), flags=cv2.INTER_NEAREST)
    cv2.imwrite(str(OUT / f"rect_{tag}.png"), warped)
    mx, my, ts = px_to_merc(x_min, y_min, x0, y0, z)
    Path(OUT / f"rect_{tag}.jgw").write_text("\n".join(f"{v:.8f}" for v in [ts, 0, 0, -ts, mx + ts / 2, my - ts / 2]))
    # 疊圖：底圖裁同範圍，UAV 半透明
    crop = ref_bgr[max(0, y_min):y_max, max(0, x_min):x_max].copy()
    ov = crop.copy()
    wx0, wy0 = max(0, x_min) - x_min, max(0, y_min) - y_min
    wv = warped[wy0:wy0 + crop.shape[0], wx0:wx0 + crop.shape[1]]
    vm = valid[wy0:wy0 + crop.shape[0], wx0:wx0 + crop.shape[1]] > 0
    ov[vm] = (0.55 * wv[vm] + 0.45 * crop[vm]).astype(np.uint8)
    cv2.imwrite(str(OUT / f"overlay_{tag}.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(OUT / f"ref_crop_{tag}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    ll = [unmerc(*px_to_merc(px, py, x0, y0, z)[:2]) for px, py in c_ref]
    ctr = cv2.perspectiveTransform(np.float32([[[w / 2, h / 2]]]), H)[0][0]
    cll = unmerc(*px_to_merc(ctr[0], ctr[1], x0, y0, z)[:2])
    # 地面解析度：UAV 每像素對應公尺（取影像中心處的局部尺度）
    d = cv2.perspectiveTransform(np.float32([[[w / 2, h / 2], [w / 2 + 1, h / 2]]]), H)[0]
    gsd = float(np.linalg.norm(d[1] - d[0])) * ts * math.cos(math.radians(LAT))
    return {"corners_lonlat": [[round(a, 6), round(b, 6)] for a, b in ll], "center_lonlat": [round(cll[0], 6), round(cll[1], 6)],
            "gsd_m_per_px_center": round(gsd, 3), "out_size_px": [ow, oh], "pixel_size_merc_m": round(ts, 4)}

def hav(lon1, lat1, lon2, lat2):
    p = math.pi / 180
    a = math.sin((lat2 - lat1) * p / 2) ** 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2
    return 12742000 * math.asin(math.sqrt(a))

if __name__ == "__main__":
    uav = cv2.imread(str(OUT / "uav.jpg"))
    results = {}
    for src in ("nlsc", "esri"):
        for z, half in ((18, 3), (17, 3)):
            ref, x0, y0 = mosaic(src, z, half)
            ts = px_to_merc(0, 0, x0, y0, z)[2] * math.cos(math.radians(LAT))     # 底圖 m/px（地面）
            best = None
            for gsd_guess in (0.20, 0.30, 0.45):                                     # UAV 地面解析度未知：多個尺度試
                s = min(1.0, gsd_guess / ts)
                H, st = register(uav, ref, s)
                if H is not None and (best is None or st["inliers"] > best[1]["inliers"]):
                    best = (H, {**st, "uav_gsd_guess": gsd_guess})
            tag = f"{src}_z{z}"
            if best is None:
                results[tag] = {"ok": False}; print(tag, "FAILED"); continue
            H, st = best
            info = rectify(uav, H, z, x0, y0, ref, tag)
            info["dist_from_page_gps_m"] = round(hav(*info["center_lonlat"], LON, LAT))
            results[tag] = {"ok": True, **st, **info}
            print(tag, json.dumps(results[tag], ensure_ascii=False))
    # 兩個底圖獨立對位的一致性：中心點距離
    ok = {k: v for k, v in results.items() if v.get("ok")}
    ks = list(ok)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = ok[ks[i]]["center_lonlat"], ok[ks[j]]["center_lonlat"]
            print("center diff", ks[i], ks[j], round(hav(*a, *b), 1), "m")
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
