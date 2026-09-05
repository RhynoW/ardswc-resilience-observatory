"""
Google Maps 衛星 tile 下載 primitive（獨立、自足，不依賴任何姊妹 repo）。

從 SOM_analyst_workstation/Training/collect_gmaps_tiles.py 抽出這支 demo 只需要的
最小子集：座標↔tile 索引換算 + 單張 tile 下載（含磁碟快取、429 退避、重試、UA 輪替）。
刻意不帶原始檔案裡的 duckdb/ultralytics 兩段式蒐集邏輯——那些是本地資料蒐集 pipeline
的一部分，不是這支公開偵測 demo 需要的東西。
"""
import json
import math
import os
import random
import re
import threading
import time
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

GOOGLE_TILE_PX = 256

TILE_CACHE_DIR = Path(os.getenv("GMAPS_DEMO_TILE_CACHE_DIR",
                                str(Path(__file__).parent / "tile_cache")))

GMAPS_SERVERS = ["mt0", "mt1", "mt2", "mt3"]
GMAPS_URL = "https://{server}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"

# 可選衛星圖磚來源（顯示與偵測共用）。
#  google：標準 XYZ、無 CORS（display + server 偵測 OK，client 端推論不行）。
#  esri  ：z/y/x、送 CORS header（可供未來 client 端讀 canvas 像素推論）；全球僅到 z19。
#  bing  ：quadkey 索引、無 CORS（同 google，display + server 偵測 OK，client 端推論不行）；~z20。
#  apple ：MapKit 衛星圖磚，與前三者本質不同——公開、無需認證的 XYZ 端點不存在，tile 需要
#          v= + accessKey= 這組短效期（實測 expiresInSeconds=1800，即 30 分鐘）token。
#          取得方式（同 webapp/gmaps_bbox_demo 的既有驗證作法，見該 app README「Apple 來源」）：
#          純 HTTP 兩段請求鏈，不需要瀏覽器/headless——`duckduckgo.com/local.js?get_mk_token=1`
#          換一個短效 JWT，拿去打 `cdn.apple-mapkit.com/ma/bootstrap` 換到含 accessKey 的 JSON
#          （這正是 DuckDuckGo 自己的網頁載入 MapKit JS 時做的事，只是我們直接重放這兩個
#          request，不用真的把整個瀏覽器/DOM 跑起來）。過期前會自動背景重新換發、只存記憶體。
#          本 app（`change_detect_viewer`，純本機互動地圖，非公開 HF Space 部署）預設**開啟**
#          自動換發（APPLE_AUTO 預設 True，與 gmaps_bbox_demo 的「公開部署預設關閉」取捨不同——
#          那邊考量的是「公開 Space 不該無條件依賴第三方網站自己的工作階段機制」，本 app 只在
#          使用者自己機器上執行，不受那個顧慮限制）；仍可用 GMAPS_DEMO_APPLE_APPLE_AUTO=0 關閉，
#          或用 GMAPS_DEMO_APPLE_ACCESSKEY/_V 手動指定 token 覆蓋。
APPLE_ACCESS_KEY = os.getenv("GMAPS_DEMO_APPLE_ACCESSKEY", "").strip()
APPLE_V = os.getenv("GMAPS_DEMO_APPLE_V", "").strip()
APPLE_AUTO = os.getenv("GMAPS_DEMO_APPLE_AUTO", "1").strip() == "1"

_DDG_TOKEN_URL = "https://duckduckgo.com/local.js?get_mk_token=1"
_APPLE_BOOTSTRAP_URL = "https://cdn.apple-mapkit.com/ma/bootstrap?apiVersion=2&mkjsVersion=5.78.158&poi=1"
_APPLE_REFRESH_MARGIN_S = 300   # 到期前 5 分鐘就提早換發，避免請求途中剛好過期
_apple_lock = threading.Lock()
_apple_cache = {"v": None, "access_key": None, "expires_at": 0.0}


def _apple_bootstrap_fetch():
    """純 HTTP 重放 DuckDuckGo 自己載入 MapKit JS 時做的兩段請求，換一組新鮮的
    (v, accessKey, expires_in_seconds)。"""
    ua = {"User-Agent": random.choice(USER_AGENTS)}
    req1 = Request(_DDG_TOKEN_URL, headers=ua)
    with urlopen(req1, timeout=TIMEOUT) as r:
        token = r.read().decode("utf-8").strip()
    req2 = Request(_APPLE_BOOTSTRAP_URL, headers={**ua, "Authorization": f"Bearer {token}"})
    with urlopen(req2, timeout=TIMEOUT) as r:
        data = json.loads(r.read().decode("utf-8"))
    sat = next((t for t in data.get("tileSources", []) if t.get("tileSource") == "satellite"), None)
    if sat is None:
        raise RuntimeError("bootstrap 回應缺少 satellite tileSource")
    path = sat["path"]
    v = re.search(r"[?&]v=([^&]+)", path)
    key = re.search(r"[?&]accessKey=([^&]+)", path)
    if not (v and key):
        raise RuntimeError("satellite path 缺少 v/accessKey 參數")
    expires_in = float(data.get("expiresInSeconds") or 1800)
    return v.group(1), key.group(1), expires_in


def get_apple_auth():
    """回傳 (v, access_key) 給 download_tile 用。手動環境變數優先（若兩者都設定）；否則若
    APPLE_AUTO 就自動 bootstrap+快取+到期自動重新換發；都沒有就回 (None, None)。"""
    if APPLE_ACCESS_KEY and APPLE_V:
        return APPLE_V, APPLE_ACCESS_KEY
    if not APPLE_AUTO:
        return None, None
    with _apple_lock:
        now = time.time()
        if _apple_cache["access_key"] and now < _apple_cache["expires_at"] - _APPLE_REFRESH_MARGIN_S:
            return _apple_cache["v"], _apple_cache["access_key"]
        try:
            v, key, expires_in = _apple_bootstrap_fetch()
            _apple_cache.update(v=v, access_key=key, expires_at=now + expires_in)
            print(f"[apple bootstrap] 換發成功，{int(expires_in)}s 後到期")
            return v, key
        except Exception as exc:  # noqa: BLE001 — 換發失敗不致命，退回舊快取或 None（下游 fail-closed）
            print(f"[apple bootstrap] 換發失敗：{type(exc).__name__}: {exc}")
            if _apple_cache["access_key"]:
                return _apple_cache["v"], _apple_cache["access_key"]
            return None, None


def apple_is_available():
    v, key = get_apple_auth()
    return bool(v and key)


APPLE_CONFIGURED = bool(APPLE_ACCESS_KEY and APPLE_V)   # 供舊呼叫端相容；新程式碼請用 apple_is_available()


def set_apple_token(v, access_key):
    """手動覆蓋 Apple token（供未來若改用瀏覽器側錄手法時呼叫）。設定後 get_apple_auth() 的手動
    環境變數優先分支不會用到這個，故直接寫 _apple_cache 讓自動快取路徑取用。"""
    with _apple_lock:
        _apple_cache.update(v=v, access_key=access_key, expires_at=time.time() + 1800)
    return bool(v and access_key)

TILE_SOURCES = {
    "google": {"url": GMAPS_URL, "servers": GMAPS_SERVERS,
               "referer": "https://maps.google.com/", "max_zoom": 21},
    "esri":   {"url": "https://server.arcgisonline.com/ArcGIS/rest/services/"
                      "World_Imagery/MapServer/tile/{z}/{y}/{x}", "servers": None,
               "referer": "https://www.arcgis.com/", "max_zoom": 19},
    "bing":   {"url": "https://ecn.t{server}.tiles.virtualearth.net/tiles/a{quadkey}.jpeg?g=1",
               "servers": ["0", "1", "2", "3"], "referer": "https://www.bing.com/maps/",
               "max_zoom": 20, "quadkey": True},
    "apple":  {"url": "https://sat-cdn.apple-mapkit.com/tile?style=7&size=1&scale=1"
                      "&z={z}&x={x}&y={y}&v={v}&accessKey={access_key}", "servers": None,
               "referer": "https://duckduckgo.com/", "max_zoom": 20, "needs_apple_auth": True},
    # 百度衛星：BD09MC 投影 + GCJ02/BD09 座標偏移，與其他來源不同座標系 → 由後端反投影成標準 EPSG:3857
    # tile（get_baidu_3857_tile），故對前端 Leaflet 與後端偵測而言就是一般 3857 來源、免改管線。
    "baidu":  {"url": None, "servers": None, "referer": "https://map.baidu.com/",
               "max_zoom": int(os.getenv("GMAPS_DEMO_BAIDU_MAXZ", "19")), "reproject_baidu": True},
    # 騰訊衛星：GCJ02 + 標準墨卡托 + TMS y 翻轉 → 同樣反投影成標準 3857 tile（get_tencent_3857_tile）。
    "tencent": {"url": None, "servers": None, "referer": "https://map.qq.com/",
                "max_zoom": int(os.getenv("GMAPS_DEMO_TENCENT_MAXZ", "18")), "reproject_tencent": True},
    # 內政部國土測繪中心 WMTS（https://wmts.nlsc.gov.tw/wmts，公開、免認證、標準 GoogleMapsCompatible
    # z/y/x 索引，已用真實座標實測驗證 url 順序與內容正確——見 CLAUDE.md §17.6）。
    #  nlsc_topo  ：圖層 B50000＝1/50000 地形圖（使用者慣稱「經建版地圖」），z17 仍為真實內容
    #               （非原生解析度、伺服器端放大），故 max_zoom 保守設 17。
    #  nlsc_photo ：圖層 PHOTO2＝正射影像圖(通用)，即最新期航照/衛星正射影像，實測 z20 仍銳利
    #               （停車格/車輛可辨），max_zoom 設 20。
    #  兩者皆帶 insecure_ssl=True——wmts.nlsc.gov.tw 的 TLS 憑證缺少 Subject Key Identifier
    #  擴充欄位（伺服器端憑證鏈設定缺陷，非本專案可控），Python ssl 模組預設嚴格驗證會直接
    #  拒絕（curl/瀏覽器多半較寬容而不會發現）；已用同一份 URL 實測確認純粹是驗證嚴格度問題、
    #  非位址錯誤（不驗證時 200 OK、驗證時 SSLCertVerificationError），故僅對此兩個政府圖磚
    #  來源關閉憑證驗證，其餘來源（google/esri/bing/apple/baidu/tencent）不受影響。
    "nlsc_topo": {"url": "https://wmts.nlsc.gov.tw/wmts/B50000/default/GoogleMapsCompatible/{z}/{y}/{x}",
                  "servers": None, "referer": "https://maps.nlsc.gov.tw/", "max_zoom": 17,
                  "insecure_ssl": True},
    "nlsc_photo": {"url": "https://wmts.nlsc.gov.tw/wmts/PHOTO2/default/GoogleMapsCompatible/{z}/{y}/{x}",
                   "servers": None, "referer": "https://maps.nlsc.gov.tw/", "max_zoom": 20,
                   "insecure_ssl": True},
}

_INSECURE_SSL_CTX = None


def _insecure_ssl_context():
    global _INSECURE_SSL_CTX
    if _INSECURE_SSL_CTX is None:
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        _INSECURE_SSL_CTX = ctx
    return _INSECURE_SSL_CTX


def quadkey(tx, ty, zoom):
    """Bing/Virtual Earth 的 quadkey 索引（由 z/x/y 換算）。"""
    q = []
    for i in range(zoom, 0, -1):
        digit, mask = 0, 1 << (i - 1)
        if tx & mask:
            digit += 1
        if ty & mask:
            digit += 2
        q.append(str(digit))
    return "".join(q)


def tile_cache_path(tx, ty, zoom, source="google"):
    """來源分開快取，避免不同來源同 (z,x,y) 互相覆蓋。"""
    return TILE_CACHE_DIR / source / str(zoom) / f"{tx}_{ty}.jpg"

REQUEST_DELAY = float(os.getenv("GMAPS_DEMO_REQUEST_DELAY", "1.5"))
MAX_RETRY = 3
TIMEOUT = 15
RATE_LIMIT_BACKOFF = 45

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
]


def latlon_to_tile(lat, lon, zoom):
    """WGS84 → Google Maps 整數 tile 索引 (x, y)。"""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def _normalize_tile(img):
    """畫布拼接假設每張 tile 都是 GOOGLE_TILE_PX 見方——google/esri/bing 皆天生如此，
    但 apple 的 size/scale 語意是從第三方逆向工程文件抄來的、未必每個 z 都精準對應 256px，
    保險起見不符就強制縮放，避免拼接錯位。"""
    if img.size != (GOOGLE_TILE_PX, GOOGLE_TILE_PX):
        img = img.resize((GOOGLE_TILE_PX, GOOGLE_TILE_PX), Image.LANCZOS)
    return img


def download_tile(tx, ty, zoom, source="google"):
    """下載單張衛星 tile（source: 'google' | 'esri' | 'bing' | 'apple'），回傳 PIL Image 或 None。先查磁碟快取。"""
    src = TILE_SOURCES.get(source, TILE_SOURCES["google"])
    cache_path = tile_cache_path(tx, ty, zoom, source)
    if cache_path.exists():
        try:
            return _normalize_tile(Image.open(cache_path).convert("RGB"))
        except Exception:
            cache_path.unlink(missing_ok=True)

    if src.get("reproject_baidu") or src.get("reproject_tencent"):   # 百度/騰訊：反投影成 3857 tile（座標偏移校正）
        img = (get_baidu_3857_tile(zoom, tx, ty) if src.get("reproject_baidu")
               else get_tencent_3857_tile(zoom, tx, ty))
        if img is None:
            return None
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(cache_path, "JPEG", quality=90)
        except Exception:
            pass
        return _normalize_tile(img)

    if src.get("needs_apple_auth"):
        v, access_key = get_apple_auth()
        if not (v and access_key):   # fail-closed：拿不到 token 就不發出注定失敗的請求
            return None
        url = src["url"].format(x=tx, y=ty, z=zoom, v=v, access_key=access_key)
    elif src.get("quadkey"):
        qk = quadkey(tx, ty, zoom)
        url = (src["url"].format(server=random.choice(src["servers"]), quadkey=qk)
               if src["servers"] else src["url"].format(quadkey=qk))
    elif src["servers"]:
        url = src["url"].format(server=random.choice(src["servers"]), x=tx, y=ty, z=zoom)
    else:
        url = src["url"].format(x=tx, y=ty, z=zoom)
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": src["referer"],
        "Cache-Control": "no-cache",
    }

    urlopen_kwargs = {"timeout": TIMEOUT}
    if src.get("insecure_ssl"):
        urlopen_kwargs["context"] = _insecure_ssl_context()

    for attempt in range(MAX_RETRY):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, **urlopen_kwargs) as resp:
                data = resp.read()
            img = Image.open(BytesIO(data)).convert("RGB")
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(data)
            except Exception:
                pass
            return _normalize_tile(img)
        except HTTPError as e:
            if e.code in (429, 403):
                wait = RATE_LIMIT_BACKOFF * (attempt + 1) + random.uniform(0, 15)
                time.sleep(wait)
            elif attempt < MAX_RETRY - 1:
                time.sleep(2.0 * (attempt + 1))
            else:
                return None
        except URLError:
            if attempt < MAX_RETRY - 1:
                time.sleep(2.0 * (attempt + 1))
            else:
                return None
        except Exception:
            return None
    return None


# ═══════════════════════════════════════════════════════════════════════════
# 百度衛星（BD09MC）→ 標準 EPSG:3857 tile 反投影代理
# 百度用 BD09（GCJ02 之上再加一層偏移）＋ BD09MC，與其他來源的 WGS84 WebMercator 不同座標系。
# 後端把每張被請求的 3857 tile，逐像素 WGS84→GCJ02→BD09→BD09MC 反查百度原生 tile 重採樣 →
# 輸出對齊的 3857 tile：前端 Leaflet 與後端偵測皆視為一般 3857 來源、座標正確、管線免改。
# 座標公式為公開標準（eviltransform + 百度 convertor.js 分段多項式）。
# ═══════════════════════════════════════════════════════════════════════════
_BD_XPI = math.pi * 3000.0 / 180.0
_BD_A = 6378245.0
_BD_EE = 0.00669342162296594323
BAIDU_MAX_Z = int(os.getenv("GMAPS_DEMO_BAIDU_MAXZ", "19"))
_BAIDU_UDT = os.getenv("GMAPS_DEMO_BAIDU_UDT", "20240101")

_LLBAND = [75, 60, 45, 30, 15, 0]
_LL2MC = [
    [-0.0015702102444, 111320.7020616939, 1704480524535203.0, -10338987376042340.0, 26112667856603880.0, -35149669176653700.0, 26595700718403920.0, -10725012454188240.0, 1800819912950474.0, 82.5],
    [0.0008277824516172526, 111320.7020463578, 647795574.6671607, -4082003173.641316, 10774905663.51142, -15171875531.51559, 12053065338.62167, -5124939663.577472, 913311935.9512032, 67.5],
    [0.00337398766765, 111320.7020202162, 4481351.045890365, -23393751.19931662, 79682215.47186455, -115964993.2797253, 97236711.15602145, -43661946.33752821, 8477230.501135234, 52.5],
    [0.00220636496208, 111320.7020209128, 51751.86112841131, 3796837.749470245, 992013.7397791013, -1221952.21711287, 1340652.697009075, -620943.6990984312, 144416.9293806241, 37.5],
    [-0.0003441963504368392, 111320.7020576856, 278.2353980772752, 2485758.690035394, 6070.750963243378, 54821.18345352118, 9540.606633304236, -2710.55326746645, 1405.483844121726, 22.5],
    [-0.0003218135878613132, 111320.7020701615, 0.00369383431289, 823725.6402795718, 0.46104986909093, 2351.343141331292, 1.58060784298199, 8.77738589078284, 0.37238884252424, 7.45],
]


def _transform_lat(x, y):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * np.sqrt(np.abs(x))
    ret += (20.0 * np.sin(6.0 * x * math.pi) + 20.0 * np.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * np.sin(y * math.pi) + 40.0 * np.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * np.sin(y / 12.0 * math.pi) + 320.0 * np.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lng(x, y):
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * np.sqrt(np.abs(x))
    ret += (20.0 * np.sin(6.0 * x * math.pi) + 20.0 * np.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * np.sin(x * math.pi) + 40.0 * np.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * np.sin(x / 12.0 * math.pi) + 300.0 * np.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return ret


def _wgs84_to_gcj02(lng, lat):
    dlat = _transform_lat(lng - 105.0, lat - 35.0)
    dlng = _transform_lng(lng - 105.0, lat - 35.0)
    radlat = np.radians(lat)
    magic = np.sin(radlat)
    magic = 1 - _BD_EE * magic * magic
    sqmagic = np.sqrt(magic)
    dlat = (dlat * 180.0) / ((_BD_A * (1 - _BD_EE)) / (magic * sqmagic) * math.pi)
    dlng = (dlng * 180.0) / (_BD_A / sqmagic * np.cos(radlat) * math.pi)
    return lng + dlng, lat + dlat


def _gcj02_to_bd09(lng, lat):
    z = np.sqrt(lng * lng + lat * lat) + 0.00002 * np.sin(lat * _BD_XPI)
    theta = np.arctan2(lat, lng) + 0.000003 * np.cos(lng * _BD_XPI)
    return z * np.cos(theta) + 0.0065, z * np.sin(theta) + 0.006


def _bd09_to_bd09mc(lng, lat):
    """BD09 經緯度 → BD09MC 公尺。單一 tile 緯度範圍極小 → 用中心緯度選 band。"""
    latm = float(np.mean(lat))
    cD = _LL2MC[-1]
    for i, b in enumerate(_LLBAND):
        if abs(latm) >= b:
            cD = _LL2MC[i]
            break
    lat_c = np.clip(lat, -74.0, 74.0)
    mcx = cD[0] + cD[1] * np.abs(lng)
    i_ = np.abs(lat_c) / cD[9]
    mcy = (cD[2] + cD[3] * i_ + cD[4] * i_**2 + cD[5] * i_**3
           + cD[6] * i_**4 + cD[7] * i_**5 + cD[8] * i_**6)
    return mcx * np.sign(lng), mcy * np.sign(lat_c)


def _baidu_native_url(bx, by, bz):
    s = random.randint(0, 3)
    return (f"https://maponline{s}.bdimg.com/starpic/?qt=satepc&u=x={bx};y={by};z={bz};v=009;"
            f"type=sate&fm=46&udt={_BAIDU_UDT}&app=webearth2&from=webearth")


def _fetch_baidu_native(bx, by, bz):
    """單張百度原生衛星 tile（BD09MC 索引、north-up 256px），含磁碟快取。"""
    cache = TILE_CACHE_DIR / "baidu_native" / str(bz) / f"{bx}_{by}.jpg"
    if cache.exists():
        try:
            return Image.open(cache).convert("RGB")
        except Exception:
            cache.unlink(missing_ok=True)
    if bx < 0 or by < 0:
        return None
    headers = {"User-Agent": random.choice(USER_AGENTS), "Referer": "https://map.baidu.com/",
               "Accept": "image/*,*/*;q=0.8"}
    for attempt in range(MAX_RETRY):
        try:
            with urlopen(Request(_baidu_native_url(bx, by, bz), headers=headers), timeout=TIMEOUT) as r:
                data = r.read()
            img = Image.open(BytesIO(data)).convert("RGB")
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(data)
            except Exception:
                pass
            return img
        except HTTPError as e:
            if e.code in (429, 403):
                time.sleep(RATE_LIMIT_BACKOFF * (attempt + 1) + random.uniform(0, 8))
            elif attempt < MAX_RETRY - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                return None
        except Exception:
            if attempt < MAX_RETRY - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                return None
    return None


def get_baidu_3857_tile(z, x, y):
    """標準 EPSG:3857 slippy tile (z,x,y) ← 反投影自百度 BD09MC 影像（含座標偏移校正）。
    回傳對齊的 256px PIL；覆蓋範圍全無百度圖磚時回 None。"""
    n = 2 ** int(z)
    total = n * 256.0
    js, is_ = np.meshgrid(np.arange(256), np.arange(256))
    gpx = x * 256 + js + 0.5
    gpy = y * 256 + is_ + 0.5
    lng = gpx / total * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(math.pi * (1.0 - 2.0 * gpy / total))))
    glng, glat = _wgs84_to_gcj02(lng, lat)
    blng, blat = _gcj02_to_bd09(glng, glat)
    mcx, mcy = _bd09_to_bd09mc(blng, blat)
    bz = min(BAIDU_MAX_Z, int(z))
    res = 2.0 ** (18 - bz)
    bgx = mcx / res
    bgy = mcy / res
    if not (np.all(np.isfinite(bgx)) and np.all(np.isfinite(bgy))):
        return None
    btx_min, btx_max = int(np.floor(bgx.min() / 256)), int(np.floor(bgx.max() / 256))
    bty_min, bty_max = int(np.floor(bgy.min() / 256)), int(np.floor(bgy.max() / 256))
    W, H = btx_max - btx_min + 1, bty_max - bty_min + 1
    if W <= 0 or H <= 0 or W * H > 64:
        return None
    mosaic = np.zeros((H * 256, W * 256, 3), np.uint8)
    got = False
    for btx in range(btx_min, btx_max + 1):
        for bty in range(bty_min, bty_max + 1):
            t = _fetch_baidu_native(btx, bty, bz)
            if t is None:
                continue
            got = True
            ta = np.asarray(t)
            if ta.shape[:2] != (256, 256):
                ta = np.asarray(t.resize((256, 256), Image.LANCZOS))
            rr = (bty_max - bty) * 256
            cc = (btx - btx_min) * 256
            mosaic[rr:rr + 256, cc:cc + 256] = ta
    if not got:
        return None
    col = np.clip((bgx - btx_min * 256).astype(np.int64), 0, W * 256 - 1)
    row = np.clip(((bty_max + 1) * 256 - bgy).astype(np.int64), 0, H * 256 - 1)
    out = mosaic[row, col]
    return Image.fromarray(out)


# ═══════════════════════════════════════════════════════════════════════════
# 騰訊衛星（GCJ02 + 標準 WebMercator + TMS y 翻轉）→ 標準 EPSG:3857 tile 反投影代理
# 騰訊比百度單純：座標系是 GCJ02（只一層中國偏移、無 BD09），投影就是標準球墨卡托，僅 tile y 軸
# 由下往上（TMS）。故反投影＝逐像素 WGS84→GCJ02→（標準 slippy 全域像素）→ 取騰訊 tile(TMS 翻轉 y)。
# 端點：https://p{0-3}.map.gtimg.com/sateTiles/{z}/{x>>4}/{ty>>4}/{x}_{ty}.jpg，ty=2^z-1-y_slippy。
# ═══════════════════════════════════════════════════════════════════════════
TENCENT_MAX_Z = int(os.getenv("GMAPS_DEMO_TENCENT_MAXZ", "18"))


def _tencent_native_url(tx, ty, z):
    s = random.randint(0, 3)
    return f"https://p{s}.map.gtimg.com/sateTiles/{z}/{tx >> 4}/{ty >> 4}/{tx}_{ty}.jpg"


def _fetch_tencent_native(tx, ty, z):
    """單張騰訊原生衛星 tile（tx＝標準 slippy x、ty＝TMS 翻轉後 y、north-up 256px），含磁碟快取。"""
    cache = TILE_CACHE_DIR / "tencent_native" / str(z) / f"{tx}_{ty}.jpg"
    if cache.exists():
        try:
            return Image.open(cache).convert("RGB")
        except Exception:
            cache.unlink(missing_ok=True)
    if tx < 0 or ty < 0:
        return None
    headers = {"User-Agent": random.choice(USER_AGENTS), "Referer": "https://map.qq.com/",
               "Accept": "image/*,*/*;q=0.8"}
    for attempt in range(MAX_RETRY):
        try:
            with urlopen(Request(_tencent_native_url(tx, ty, z), headers=headers), timeout=TIMEOUT) as r:
                data = r.read()
            img = Image.open(BytesIO(data)).convert("RGB")
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(data)
            except Exception:
                pass
            return img
        except HTTPError as e:
            if e.code in (429, 403):
                time.sleep(RATE_LIMIT_BACKOFF * (attempt + 1) + random.uniform(0, 8))
            elif attempt < MAX_RETRY - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                return None
        except Exception:
            if attempt < MAX_RETRY - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                return None
    return None


def get_tencent_3857_tile(z, x, y):
    """標準 EPSG:3857 slippy tile (z,x,y) ← 反投影自騰訊 GCJ02 影像（含座標偏移校正）。
    回傳對齊的 256px PIL；覆蓋範圍全無騰訊圖磚時回 None。"""
    zz = min(TENCENT_MAX_Z, int(z))
    if zz != int(z):
        # 目標 zoom 超過騰訊上限：改在 zz 取樣（解析度稍降，位置仍正確）
        pass
    n = 2 ** int(z)
    total = n * 256.0
    js, is_ = np.meshgrid(np.arange(256), np.arange(256))
    gpx = x * 256 + js + 0.5
    gpy = y * 256 + is_ + 0.5
    lng = gpx / total * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(math.pi * (1.0 - 2.0 * gpy / total))))
    glng, glat = _wgs84_to_gcj02(lng, lat)                    # WGS84 → GCJ02
    nz = 2 ** zz
    totalz = nz * 256.0
    tgx = (glng + 180.0) / 360.0 * totalz                     # GCJ02 → 標準 slippy 全域像素（zz）
    tgy = (1.0 - np.arcsinh(np.tan(np.radians(glat))) / math.pi) / 2.0 * totalz
    if not (np.all(np.isfinite(tgx)) and np.all(np.isfinite(tgy))):
        return None
    stx_min, stx_max = int(np.floor(tgx.min() / 256)), int(np.floor(tgx.max() / 256))
    sty_min, sty_max = int(np.floor(tgy.min() / 256)), int(np.floor(tgy.max() / 256))
    W, H = stx_max - stx_min + 1, sty_max - sty_min + 1
    if W <= 0 or H <= 0 or W * H > 64:
        return None
    mosaic = np.zeros((H * 256, W * 256, 3), np.uint8)
    got = False
    for stx in range(stx_min, stx_max + 1):
        for sty in range(sty_min, sty_max + 1):
            ty_tc = nz - 1 - sty                              # TMS y 翻轉
            t = _fetch_tencent_native(stx, ty_tc, zz)
            if t is None:
                continue
            got = True
            ta = np.asarray(t)
            if ta.shape[:2] != (256, 256):
                ta = np.asarray(t.resize((256, 256), Image.LANCZOS))
            mosaic[(sty - sty_min) * 256:(sty - sty_min) * 256 + 256,
                   (stx - stx_min) * 256:(stx - stx_min) * 256 + 256] = ta
    if not got:
        return None
    col = np.clip((tgx - stx_min * 256).astype(np.int64), 0, W * 256 - 1)
    row = np.clip((tgy - sty_min * 256).astype(np.int64), 0, H * 256 - 1)
    out = mosaic[row, col]
    return Image.fromarray(out)
