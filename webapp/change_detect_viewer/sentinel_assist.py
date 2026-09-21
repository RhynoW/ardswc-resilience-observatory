"""
Sentinel-2 輔助來源：補足 Google Earth Web 歷史影像時間解析度不足的問題。

Google Earth Web 的歷史影像每個地點通常只有零星幾期（數月到數年一張）；Sentinel-2 每 5 天
重訪、2015 年起連續存在，只是解析度為 10 m/像素，看得到崩塌、裸露地、河道改道、大面積
開發這類「面積級」變化，看不到單棟建物。因此本模組是**補充時間軸**，不取代 GE 影像。

取像方式：Copernicus Data Space 的 Sentinel Hub **WMTS 512 像素圖磚**（PopularWebMercator512，
與 Rust_WMTS_Server 的 static/layers.js 相同來源與同一個 Instance ID）。tilematrix 14 每張約
4.9 km、9.55 m/像素，剛好對應 10 m 原生解析度；5 km 範圍約 4 張圖磚，拼接後裁成目標範圍。
伺服器會在**每張圖磚**左下角蓋一塊 Copernicus 浮水印，拼接後會落在影像內部，所以每張圖磚
先把該區塊填成固定灰色再拼——各期同位置同色，SSIM 於該處恆為 1，不產生假變化
（代價：每張圖磚約 6% 的面積永久看不到，不是所有變化都看得見）。
找有哪些日期可用則走 STAC 目錄（免金鑰），並用場景雲量先濾掉明顯被雲蓋住的日期。

輸出格式刻意與 GE 擷取一致（`<site>_gmap_<YYYYMMDD>.png` + `.jgw` EPSG:3857 世界檔），
這樣既有的 `ge_change_detect.detect_change()`、`/api/pair`、`/api/timeline`、詳情面板
全部不必改就能吃。

Instance ID 等同存取憑證且額度綁在申請人帳號（免費 10,000 請求/月、300/分鐘），因此只從環境變數
`SENTINEL_INSTANCE_ID`（HF Space secret）讀取，絕不寫進原始碼或 repo。
"""
import json
import math
import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import cv2
import numpy as np

R_MERC = 6378137.0
STAC_SEARCH = "https://stac.dataspace.copernicus.eu/v1/search"
STAC_COLLECTION = "sentinel-2-l2a"
WMTS_TEMPLATE = "https://sh.dataspace.copernicus.eu/ogc/wmts/{instance}"
NATIVE_M_PER_PX = 10.0
TILE_PX = 512
TILE_MATRIX = 14                  # PopularWebMercator512：matrix N 每軸 2^(N-1) 張；14 → 9.55 m/px（≈10 m 原生）
MAX_TILES = 16                    # 單次取像圖磚數上限（額度保護；±5 km 最多 3×3）
UPSCALE = 2                       # 存檔時放大倍數（面板顯示用；世界檔像素大小同步縮小）
WATERMARK_BOX_PX = (190, 75)      # 每張圖磚左下角的 Copernicus 浮水印（實測約 x15–175、y435–500，含 logo+文字）
LAYERS = ("TRUE_COLOR", "FALSE_COLOR")
UA = "ardswc-resilience-observatory/1.0 (sentinel-assist)"


def instance_id():
    return os.environ.get("SENTINEL_INSTANCE_ID", "").strip()


def is_configured():
    return bool(instance_id())


# ── 座標換算 ────────────────────────────────────────────────────────────
def lonlat_to_merc(lon, lat):
    return (R_MERC * math.radians(lon),
            R_MERC * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))


def bbox_3857(lat, lon, half_m):
    """以 lat/lon 為中心、邊長 2*half_m 的 EPSG:3857 範圍。Web Mercator 在緯度 φ 的地面尺度為
    1/cos φ，所以範圍要乘 1/cos φ 才會是實際的 half_m 公尺（否則台灣緯度約 23° 會少 8%）。"""
    x, y = lonlat_to_merc(lon, lat)
    h = half_m / math.cos(math.radians(lat))
    return (x - h, y - h, x + h, y + h)


def bbox_wgs84(lat, lon, half_m):
    dlat = half_m / 111_320.0
    dlon = half_m / (111_320.0 * math.cos(math.radians(lat)))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def _http_get(url, timeout=60):
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req, timeout=timeout) as r:
        return r.read(), r.headers.get("Content-Type", "")


# ── 找出可用日期（STAC，免金鑰）───────────────────────────────────────────
def search_scenes(lat, lon, start, end, max_cloud=40.0, half_m=2500.0):
    """回傳 [{date:'YYYYMMDD', cloud:float}]，同一天多個場景取雲量最低者，依日期由舊到新。
    start/end 為 date。雲量是整個 100×100 km 場景的值，只是粗篩；之後還會逐張算 AOI 內的局部雲。"""
    params = {
        "collections": STAC_COLLECTION,
        "bbox": ",".join(f"{v:.6f}" for v in bbox_wgs84(lat, lon, half_m)),
        "datetime": f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z",
        "limit": 100,  # 此 collection 回應很大；limit>100 必須搭配 fields 擴充
        "fields": "id,properties.datetime,properties.eo:cloud_cover",
    }
    url = STAC_SEARCH + "?" + urlencode(params)
    best = {}
    for _ in range(15):  # 分頁上限（每頁 100，3 年區間單一 tile 約 200 筆）
        body, _ct = _http_get(url, timeout=45)
        doc = json.loads(body)
        for f in doc.get("features", []):
            p = f.get("properties", {})
            cc = p.get("eo:cloud_cover")
            dt = p.get("datetime") or ""
            if cc is None or len(dt) < 10:
                continue
            d = dt[:10].replace("-", "")
            if cc <= max_cloud and (d not in best or cc < best[d]):
                best[d] = float(cc)
        nxt = next((l["href"] for l in doc.get("links", []) if l.get("rel") == "next"), None)
        if not nxt:
            break
        url = nxt
    return [{"date": d, "cloud": round(c, 1)} for d, c in sorted(best.items())]


def select_dates(scenes, n):
    """從候選日期中均勻挑 n 個（含頭尾），讓時間軸涵蓋整段區間而不是擠在一起。"""
    if n >= len(scenes):
        return list(scenes)
    if n <= 1:
        return scenes[:1]
    idx = sorted({round(i * (len(scenes) - 1) / (n - 1)) for i in range(n)})
    return [scenes[i] for i in idx]


# ── 取像與品質檢查 ────────────────────────────────────────────────────────
def _fetch_tile(iid, ymd, layer, row, col):
    d = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
    q = {"service": "WMTS", "request": "GetTile", "version": "1.0.0", "style": "default",
         "format": "image/jpeg", "tilematrixset": "PopularWebMercator512", "tilematrix": TILE_MATRIX,
         "tilerow": row, "tilecol": col, "time": f"{d}/{d}", "layer": layer,
         # 不帶 maxcc 時此服務套用預設雲量上限，超過的日期整張回黑圖（實測 2025-10-11）；
         # 雲量篩選改由 STAC 場景雲量 + AOI 內亮雲/離群檢查負責
         "maxcc": 100}
    url = WMTS_TEMPLATE.format(instance=iid) + "?" + urlencode(q)
    last = None
    for attempt in range(2):
        try:
            body, ct = _http_get(url, timeout=60)
        except HTTPError as e:
            # 不把含 Instance ID 的 URL 帶進錯誤訊息（會被寫進 job log 並經 /api/job 回傳給前端）
            last = RuntimeError(f"Sentinel Hub HTTP {e.code}（{ymd} tile {row}/{col}）")
            continue
        except URLError as e:
            last = RuntimeError(f"Sentinel Hub 連線失敗（{ymd}）：{e.reason}")
            continue
        if "image" not in ct:
            raise RuntimeError(f"Sentinel Hub 回傳非影像（{ymd}）：{body[:160]!r}")
        img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.shape[:2] != (TILE_PX, TILE_PX):
            raise RuntimeError(f"Sentinel Hub 圖磚解碼失敗或尺寸異常（{ymd}）")
        return img
    raise last


def mask_watermark(tile):
    """把圖磚左下角浮水印區塊填成固定灰色（各期同位置同色，SSIM 於此恆為 1）。"""
    h, w = tile.shape[:2]
    bw, bh = WATERMARK_BOX_PX
    tile[max(0, h - bh):, :min(w, bw)] = 128
    return tile


def fetch_image(lat, lon, half_m, ymd, layer="TRUE_COLOR"):
    """取回單日影像：拼接 WMTS 512 圖磚並裁成目標範圍。回傳 (BGR uint8, bounds_3857)，
    bounds 為實際裁切後的 (minx, miny, maxx, maxy)——以整數像素對齊，供世界檔使用。
    同一座標的各期日期結果範圍完全相同（只取決於 lat/lon/half_m）。"""
    if layer not in LAYERS:
        raise ValueError(f"layer 必須是 {LAYERS}")
    iid = instance_id()
    if not iid:
        raise RuntimeError("未設定 SENTINEL_INSTANCE_ID")
    world = 2 * math.pi * R_MERC
    p = world / (TILE_PX * 2 ** (TILE_MATRIX - 1))          # 每像素 Web Mercator 公尺
    minx, miny, maxx, maxy = bbox_3857(lat, lon, half_m)
    px0, px1 = (minx + world / 2) / p, (maxx + world / 2) / p        # 全球像素座標（x 向右、y 向下）
    py0, py1 = (world / 2 - maxy) / p, (world / 2 - miny) / p
    ix0, ix1, iy0, iy1 = int(px0), math.ceil(px1), int(py0), math.ceil(py1)
    c0, c1, r0, r1 = ix0 // TILE_PX, (ix1 - 1) // TILE_PX, iy0 // TILE_PX, (iy1 - 1) // TILE_PX
    if (c1 - c0 + 1) * (r1 - r0 + 1) > MAX_TILES:
        raise ValueError("範圍過大（圖磚數超過上限）")
    mosaic = np.zeros(((r1 - r0 + 1) * TILE_PX, (c1 - c0 + 1) * TILE_PX, 3), np.uint8)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            t = mask_watermark(_fetch_tile(iid, ymd, layer, r, c))
            mosaic[(r - r0) * TILE_PX:(r - r0 + 1) * TILE_PX, (c - c0) * TILE_PX:(c - c0 + 1) * TILE_PX] = t
    crop = mosaic[iy0 - r0 * TILE_PX:iy1 - r0 * TILE_PX, ix0 - c0 * TILE_PX:ix1 - c0 * TILE_PX]
    bounds = (ix0 * p - world / 2, world / 2 - iy1 * p, ix1 * p - world / 2, world / 2 - iy0 * p)
    return crop.copy(), bounds


def cloud_fraction(img):
    """粗估 AOI 內雲/雪/霧比例：高亮度且低飽和。用於逐張剔除被雲蓋住的日期。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(((hsv[..., 2] > 170) & (hsv[..., 1] < 45)).mean())


def outlier_scores(imgs, diff_thresh=28):
    """薄雲/霧不夠亮，`cloud_fraction()` 抓不到（實測：整片薄雲只算出 1%，卻讓 SSIM 幾乎全片判成變化）。
    改用「與所有期別逐像素中位數的偏離度」：雲、霧、雲影每期位置都不同，會成為離群值；
    真正的地表變化（季節、農耕）多半是一致的趨勢，不會只在單一期出現大片偏離。
    回傳每張影像偏離中位數的像素比例（0–1），需 ≥3 張才有意義，否則全部回 0。"""
    if len(imgs) < 3:
        return [0.0] * len(imgs)
    grays = [cv2.GaussianBlur(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), (0, 0), 2).astype(np.int16) for i in imgs]
    med = np.median(np.stack(grays), axis=0)
    # 先各自扣掉整體亮度差（不同日期的曝光/大氣差異），只比較空間分布
    return [float((np.abs((g - g.mean()) - (med - med.mean())) > diff_thresh).mean()) for g in grays]


def blank_fraction(img):
    """無資料（該日 AOI 不在幅寬內）時 Sentinel Hub 回全黑；回傳近黑像素比例。"""
    return float((img.max(axis=2) < 5).mean())


def write_world_file(path, bounds, width_px, height_px):
    minx, miny, maxx, maxy = bounds
    a = (maxx - minx) / width_px
    e = -(maxy - miny) / height_px
    # 檔案行序 A, D, B, E, C, F（與 ge_change_detect._read_jgw 相同），C/F 為左上像素「中心」座標
    vals = [a, 0.0, 0.0, e, minx + a / 2, maxy + e / 2]
    Path(path).write_text("\n".join(f"{v:.10f}" for v in vals) + "\n", encoding="ascii")


_YMD = re.compile(r"^\d{8}$")


def save_frame(site_dir, site, ymd, img, bounds):
    if not _YMD.fullmatch(ymd):
        raise ValueError("日期格式錯誤")
    big = cv2.resize(img, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    png = Path(site_dir) / f"{site}_gmap_{ymd}.png"
    cv2.imwrite(str(png), big)
    write_world_file(png.with_suffix(".jgw"), bounds, big.shape[1], big.shape[0])
    return png


def default_range(years=3):
    end = datetime.now(timezone.utc).date()
    return end - timedelta(days=365 * years), end
