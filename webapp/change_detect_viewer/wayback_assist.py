"""
Esri World Imagery Wayback：「線上分析（任意座標）」的歷史影像來源，取代 Google Earth Web 即時擷取。

Wayback 是 Esri 對 World Imagery 底圖逐次發布的存檔（2014 起、約 190 個版本），每個版本可用
WMTS 圖磚（256 px）直接取得，不需要瀏覽器自動化、不吃 GPU、也沒有 UI 邊條要裁。解析度通常
略低於 GE：多數地區 z17（約 1.2 m/px）有影像，z18 只有部分地點，z19 幾乎沒有。

**版本 ≠ 影像期別**：大部分相鄰版本在同一地點的圖磚完全相同（Esri 只更新有變動的區域）。
所以不能把 190 個版本當成 190 期。做法是 Wayback 網站自己用的 tilemap 端點：對某版本查
某張圖磚，回應的 `select` 就是「這張圖磚目前這個樣子是在哪個版本引入的」；從最新版本開始，
跳到該版本更舊的一個版本再查，就能只走過「這個地點真正改過的」版本，通常每地點只有數期到十餘期。

**影像日期**：版本日期是 Esri 發布日，不是拍攝日。拍攝日讀各版本的 metadata 圖層
（`World_Imagery_Metadata_*`，圖層編號依解析度分，z17≈1.2 m 為圖層 6），欄位 SRC_DATE2。
存檔檔名用拍攝日；查不到才退回發布日並在時間軸 JSON 標註（`date_kind`）。
**同一拍攝日只留一期**：實測南港例子，tilemap 認定有變動的 7 個版本中，只有 4 個是不同拍攝日——
其餘是同一批影像重新調色／重新編碼（例：版本 10842 與 51127 拍攝日同為 2025-04-11）。
把這種「換版不換影像」當成一期，會讓 SSIM 比對到純色調差異而產生偽陽性。

輸出格式刻意與 GE／Sentinel 擷取一致（`<site>_gmap_<YYYYMMDD>.png` + `.jgw` EPSG:3857 世界檔），
既有的 `ge_change_detect.detect_change()`、`/api/pair`、`/api/timeline`、詳情面板不必改。

限制（照實）：Wayback 是各家供應商（Maxar／Vivid／航拍）的混合鑲嵌，相鄰期別可能來自不同感測器、
不同季節、不同定位誤差，SSIM 的偽陽性風險比同感測器的 Sentinel-2 序列高；結果一律是候選訊號。
授權：Esri World Imagery 條款，僅作概念驗證與輔助；正式導入應改用授權明確的政府航照。
"""
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np

R_MERC = 6378137.0
CONFIG_URL = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
TILEMAP_URL = ("https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/"
               "tilemap/{release}/{z}/{row}/{col}")
TILE_URL = ("https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/"
            "default028mm/MapServer/tile/{release}/{z}/{row}/{col}")
TILE_PX = 256
MAX_TILES = 49                  # 單期圖磚數上限（±750 m、z17 台灣緯度約 6×6，跨格時 7×7）
MIN_ZOOM, MAX_ZOOM, DEFAULT_ZOOM = 15, 18, 17
MAX_WALK = 40                   # tilemap 走訪次數上限（版本鏈長度保護）
UA = "ardswc-resilience-observatory/1.0 (wayback-assist)"
_CFG_TTL_S = 6 * 3600
_cfg_cache = {"t": 0.0, "releases": None}
_YMD = re.compile(r"^\d{8}$")
_TITLE_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _http_get(url, timeout=30, retries=2):
    last = None
    for _ in range(retries + 1):
        try:
            with urlopen(Request(url, headers={"User-Agent": UA}), timeout=timeout) as r:
                return r.read()
        except HTTPError as e:
            if e.code == 404:
                raise
            last = e
        except (URLError, TimeoutError) as e:
            last = e
        time.sleep(0.5)
    raise last


def _get_json(url, timeout=30, retries=2):
    return json.loads(_http_get(url, timeout, retries))


def releases():
    """全部 Wayback 版本，由新到舊：[{num:int, date:'YYYYMMDD', title, meta_url}]。記憶體快取 6 小時。"""
    if _cfg_cache["releases"] and time.time() - _cfg_cache["t"] < _CFG_TTL_S:
        return _cfg_cache["releases"]
    cfg = _get_json(CONFIG_URL)
    out = []
    for num, v in cfg.items():
        m = _TITLE_DATE.search(v.get("itemTitle", ""))
        if not m:
            continue
        out.append({"num": int(num), "date": m.group(1).replace("-", ""),
                    "title": v["itemTitle"], "meta_url": v.get("metadataLayerUrl")})
    out.sort(key=lambda r: r["date"], reverse=True)
    _cfg_cache.update(t=time.time(), releases=out)
    return out


def is_available():
    """設定檔取得得到就算可用（免金鑰，只需對外網路）。"""
    try:
        return bool(releases())
    except Exception:  # noqa: BLE001
        return False


# ── 座標換算 ────────────────────────────────────────────────────────────
def lonlat_to_merc(lon, lat):
    return (R_MERC * math.radians(lon),
            R_MERC * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))


def bbox_3857(lat, lon, half_m):
    """邊長 2*half_m（實際公尺）的 EPSG:3857 範圍；Web Mercator 地面尺度 1/cos φ，需放大。"""
    x, y = lonlat_to_merc(lon, lat)
    h = half_m / math.cos(math.radians(lat))
    return (x - h, y - h, x + h, y + h)


def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


# ── 找出這個地點真正有變動的版本 ─────────────────────────────────────────
def distinct_versions(lat, lon, z=DEFAULT_ZOOM, max_versions=None):
    """回傳 [(release_dict, tile_available)]，由新到舊，每項是「中心圖磚在該版本才有的新樣貌」。
    用 tilemap 的 select 欄位跳版本；沒有圖磚（data=0）的版本鏈在此中止。"""
    rels = releases()
    idx_by_num = {r["num"]: i for i, r in enumerate(rels)}
    col, row = lonlat_to_tile(lon, lat, z)
    out, i, walks = [], 0, 0
    while i < len(rels) and walks < MAX_WALK:
        walks += 1
        r = rels[i]
        tm = _get_json(TILEMAP_URL.format(release=r["num"], z=z, row=row, col=col))
        data, sel = tm.get("data") or [0], tm.get("select") or []
        if not data[0]:
            break
        # 沒有 select = 這個版本自己就是此圖磚樣貌的引進版本（實測：版本內容有變動時省略該欄位）
        introduced = sel[0] if sel else r["num"]
        if introduced not in idx_by_num:
            break
        out.append(rels[idx_by_num[introduced]])
        if max_versions and len(out) >= max_versions:
            break
        i = idx_by_num[introduced] + 1      # 清單由新到舊：下一個就是更舊的版本
    return out


META_TIMEOUT_S = 20            # metadata 服務延遲不穩（實測單次 1 秒到 120 秒），逾時就放棄、退回發布日
_meta_cache = {}


def acquisition_info(release, lat, lon, z=DEFAULT_ZOOM):
    """該版本在此座標的影像拍攝日與供應來源。回傳 {'date': 'YYYYMMDD'|None, 'res_m', 'name'}。
    metadata 圖層依解析度編號（0=1.9 cm … 6=1.2 m … 13=150 m）；z17 對應圖層 6，
    該層沒有就往較粗的圖層找（影像在較粗層級才有覆蓋）。"""
    base = release.get("meta_url")
    if not base:
        return {"date": None, "res_m": None, "name": None}
    key = (release["num"], z, *lonlat_to_tile(lon, lat, z))
    if key in _meta_cache:
        return _meta_cache[key]
    none = {"date": None, "res_m": None, "name": None}
    start = max(0, 23 - z)
    for layer in range(start, min(start + 3, 14)):
        q = (f"{base}/{layer}/query?f=json&geometry={lon:.6f},{lat:.6f}&geometryType=esriGeometryPoint"
             "&inSR=4326&spatialRel=esriSpatialRelIntersects&outFields=SRC_DATE2,SRC_RES,NICE_NAME"
             "&returnGeometry=false")
        try:
            feats = _get_json(q, META_TIMEOUT_S, retries=0).get("features") or []
        except Exception:  # noqa: BLE001
            continue
        for f in feats:
            a = f.get("attributes") or {}
            ms = a.get("SRC_DATE2")
            if ms:
                d = datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y%m%d")
                res = a.get("SRC_RES")
                info = {"date": d, "res_m": res if res and res > 0 else None, "name": a.get("NICE_NAME")}
                _meta_cache[key] = info
                return info
    return none            # 逾時／查無：不快取，下次可重試


def list_frames(lat, lon, z=DEFAULT_ZOOM, log=None):
    """此座標所有「不同拍攝日」的影像期別，由舊到新：
    [{release, release_date, date, date_kind('acquired'|'published'), res_m, source}]。"""
    versions = distinct_versions(lat, lon, z)
    if log:
        log(f"tilemap 判定此地點有 {len(versions)} 個內容有變動的 Wayback 版本，查詢拍攝日…")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=12) as ex:           # metadata 查詢彼此獨立，平行以免逐版本串行等待
        infos = list(ex.map(lambda v: acquisition_info(v, lat, lon, z), versions))
    seen, frames = set(), []
    for v, a in zip(versions, infos):                       # 新→舊；同拍攝日保留較新的版本
        ymd, kind = (a["date"], "acquired") if a["date"] else (v["date"], "published")
        if ymd in seen:
            continue
        seen.add(ymd)
        frames.append({"release": v["num"], "release_date": v["date"], "date": ymd, "date_kind": kind,
                       "res_m": a["res_m"], "source": a["name"]})
    frames.sort(key=lambda f: f["date"])
    return frames


# ── 取像 ────────────────────────────────────────────────────────────────
def _fetch_tile(release_num, z, row, col):
    body = _http_get(TILE_URL.format(release=release_num, z=z, row=row, col=col), timeout=30)
    img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    if img is None or img.shape[:2] != (TILE_PX, TILE_PX):
        raise RuntimeError(f"Wayback 圖磚解碼失敗或尺寸異常（release {release_num} {z}/{row}/{col}）")
    return img


def fetch_image(lat, lon, half_m, release_num, z=DEFAULT_ZOOM):
    """拼接指定版本的 256 px 圖磚並裁成目標範圍。回傳 (BGR uint8, bounds_3857)；
    bounds 以整數像素對齊（供世界檔）。同一座標、同一 z 的各期範圍完全相同。"""
    world = 2 * math.pi * R_MERC
    p = world / (TILE_PX * 2 ** z)                                      # 每像素 Web Mercator 公尺
    minx, miny, maxx, maxy = bbox_3857(lat, lon, half_m)
    px0, px1 = (minx + world / 2) / p, (maxx + world / 2) / p
    py0, py1 = (world / 2 - maxy) / p, (world / 2 - miny) / p
    ix0, ix1, iy0, iy1 = int(px0), math.ceil(px1), int(py0), math.ceil(py1)
    c0, c1, r0, r1 = ix0 // TILE_PX, (ix1 - 1) // TILE_PX, iy0 // TILE_PX, (iy1 - 1) // TILE_PX
    if (c1 - c0 + 1) * (r1 - r0 + 1) > MAX_TILES:
        raise ValueError("範圍過大（圖磚數超過上限）")
    mosaic = np.zeros(((r1 - r0 + 1) * TILE_PX, (c1 - c0 + 1) * TILE_PX, 3), np.uint8)
    coords = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as ex:            # 圖磚彼此獨立；串行取像實測約 1.2 秒/張
        tiles = list(ex.map(lambda rc: _fetch_tile(release_num, z, *rc), coords))
    for (r, c), t in zip(coords, tiles):
        mosaic[(r - r0) * TILE_PX:(r - r0 + 1) * TILE_PX, (c - c0) * TILE_PX:(c - c0 + 1) * TILE_PX] = t
    crop = mosaic[iy0 - r0 * TILE_PX:iy1 - r0 * TILE_PX, ix0 - c0 * TILE_PX:ix1 - c0 * TILE_PX]
    bounds = (ix0 * p - world / 2, world / 2 - iy1 * p, ix1 * p - world / 2, world / 2 - iy0 * p)
    return crop.copy(), bounds


def write_world_file(path, bounds, width_px, height_px):
    minx, miny, maxx, maxy = bounds
    a = (maxx - minx) / width_px
    e = -(maxy - miny) / height_px
    # 行序 A, D, B, E, C, F（同 ge_change_detect._read_jgw），C/F 為左上像素「中心」座標
    vals = [a, 0.0, 0.0, e, minx + a / 2, maxy + e / 2]
    Path(path).write_text("\n".join(f"{v:.10f}" for v in vals) + "\n", encoding="ascii")


def save_frame(site_dir, site, ymd, img, bounds):
    if not _YMD.fullmatch(ymd):
        raise ValueError("日期格式錯誤")
    png = Path(site_dir) / f"{site}_gmap_{ymd}.png"
    cv2.imwrite(str(png), img)
    write_world_file(png.with_suffix(".jgw"), bounds, img.shape[1], img.shape[0])
    return png


def m_per_px(lat, z):
    return 2 * math.pi * R_MERC * math.cos(math.radians(lat)) / (TILE_PX * 2 ** z)


def viewer_url(lat, lon, release_num=None, zoom=19):
    """Esri Wayback 網站的直連（取代 GE Web 的 /api/ge_trace）；release_num 指定時開在該版本。"""
    u = f"https://livingatlas.arcgis.com/wayback/#mapCenter={lon:.5f}%2C{lat:.5f}%2C{zoom}&mode=explore"
    return u + (f"&active={release_num}" if release_num else "")
