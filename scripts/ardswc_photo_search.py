# -*- coding: utf-8 -*-
"""
水保署 ARDSWC 歷史災害影像庫的真實查詢——依「事件分類＋西元年份＋中心經緯度」找出鄰近的
歷史照片，取代原本只能連到官方搜尋首頁（無法帶查詢條件）的做法。

背景：本站首頁文件記載資料是靠「自適應遞迴網格掃描 GetEventPositionList API（突破單次查詢
500 筆上限）」蒐集而來——這支模組沿用同一個精神（分批查詢、逐頁累加、直到取完為止），直接
呼叫水保署自己的公開資料 API，而非去操作官方網站那個帶 CSRF token 的搜尋表單（見 app.py
對該表單的既有記錄：純 URL 查詢字串對它無效）。

**API 來源與行為說明（2026-09-05 直接探索確認，非憑空假設）**：
  https://photo.ardswc.gov.tw/api/v1/rest/dataset/metadata/<photo_type>
欄位命名（PhotoType/DisasterYear/Lat/Lng/County/Town/Vill/FileName 等）與 PhotoType
數字碼（0/6/8/10）與本站自己 `events_trimmed.json` 的 `photo_type` 完全一致，`FileName`
本身就是可直接開啟的照片網址（`https://photo.ardswc.gov.tw/api/Media/<EventID>`，已實測
本站自己資料集裡的 `id`／`EventID` 可直接對應到真實照片）。

實測確認兩個真正有效的查詢參數（其餘常見猜測如 skip/offset/pageIndex/lat/lng/radius 皆
無效、會被忽略回傳預設第一頁）：
  - `page=N`：真分頁，每頁固定上限 1000 筆（`?page=2` 回傳 No 1001 起，非重複第一頁）。
  - `year=YYYY`：**部分有效但不精確**——實測 `year=1999` 回傳整頁 1000 筆全部精確為 1999
    年（且 `page=2` 仍全部 1999，代表該年份真實筆數 >1000，需要分頁才拿得完）；但
    `year=2017` 回傳的 595 筆卻混雜 2004–2020 多個年份，並非嚴格篩選。**因此年份參數只能
    當「伺服器端粗篩、減少要抓的資料量」的最佳化，不能信任它真的只回傳該年——呼叫端一律
    要自己再依 `DisasterYear` 精確比對一次，這正是本站一貫「機器只建議、自己驗證」的
    治理原則在這裡的具體實踐。**

分頁邏輯（`_fetch_year_pages`）：對 `type_<photo_type>?year=<year>` 逐頁（`page=1,2,3...`）
抓取，直到某一頁回傳筆數 <1000（代表該伺服器端粗篩結果已經取完）或達到安全上限
`_MAX_PAGES`（防止對方回傳異常大量資料時無限迴圈）；每頁抓到的原始資料一律先做本地精確
年份比對，只有『DisasterYear 剛好等於要找的年份』才留下——不精確比對到的資料直接丟棄，不
當作候選。這樣才是名符其實的「用分頁掃描突破單頁上限」，而不是只抓第一頁就交差。

治理：這裡查到的照片是水保署官方資料，不是本站自建的判定——本模組只做「依座標距離+年份
精確比對」，不做任何自動化的災害成因判斷；找不到結果時如實回報「無符合結果」，不擴大條件
靜默湊數。
"""
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

_API_BASE = "https://photo.ardswc.gov.tw/api/v1/rest/dataset/metadata"
_MEDIA_BASE = "https://photo.ardswc.gov.tw/api/Media"
_CACHE_TTL_S = 24 * 3600  # 官方資料集非即時性資料，24 小時內重複查詢直接用快取，不重打 API
_TIMEOUT_S = 20
_PAGE_SIZE = 1000     # 實測該 API 單頁上限
_MAX_PAGES = 15        # 安全上限（=最多 15,000 筆／年／分類），避免異常大量資料時無限分頁

PHOTO_TYPE_LABELS = {"0": "災害事件", "6": "重要地景", "8": "媒體報導", "10": "出版品照片"}


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _http_get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as r:
        return json.loads(r.read().decode("utf-8"))


def _fetch_year_pages(photo_type, year, cache_dir):
    """分頁掃描某分類＋年份粗篩後的所有頁面，本地精確比對年份後才保留——見模組開頭說明。
    磁碟快取以 (photo_type, year) 為鍵，24 小時內重複查詢同一年份直接用快取，不重打 API。
    fail-soft：抓取中途失敗時，用目前已經抓到的頁面繼續（總比整個查詢失敗好），但會在
    回傳的 meta 標記 partial，讓呼叫端誠實告知使用者這次結果可能不完整。
    """
    cache_dir = Path(cache_dir)
    cache_fp = cache_dir / f"type_{photo_type}_year_{year}.json"
    if cache_fp.exists() and (time.time() - cache_fp.stat().st_mtime) < _CACHE_TTL_S:
        payload = json.loads(cache_fp.read_text(encoding="utf-8"))
        return payload["records"], payload["meta"]

    exact_records = []
    pages_fetched = 0
    partial = False
    fetch_error = None
    for page in range(1, _MAX_PAGES + 1):
        url = f"{_API_BASE}/{photo_type}?" + urllib.parse.urlencode({"year": year, "page": page})
        try:
            page_data = _http_get_json(url)
        except Exception as e:  # noqa: BLE001
            partial = True
            fetch_error = f"{type(e).__name__}: {e}"
            break
        pages_fetched += 1
        exact_records.extend(r for r in page_data if r.get("DisasterYear") == year)
        if len(page_data) < _PAGE_SIZE:
            break  # 該頁未滿，代表伺服器端粗篩結果已經取完
    else:
        partial = True  # 撞到 _MAX_PAGES 安全上限、可能還有更多頁沒抓

    meta = {"pages_fetched": pages_fetched, "partial": partial, "fetch_error": fetch_error,
            "fetched_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}

    if exact_records or not fetch_error:
        # 只要至少成功抓到一頁（即使中途出錯），就把目前結果快取下來——好過完全沒有快取；
        # partial 旗標會一路帶到最終回應，使用者看得到「這次結果可能不完整」的誠實揭露。
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_fp.write_text(json.dumps({"records": exact_records, "meta": meta}, ensure_ascii=False),
                                 encoding="utf-8")
        except OSError:
            pass
    elif cache_fp.exists():
        payload = json.loads(cache_fp.read_text(encoding="utf-8"))
        return payload["records"], payload["meta"]
    elif fetch_error:
        raise RuntimeError(f"無法取得水保署官方資料且無快取可用：{fetch_error}")

    return exact_records, meta


def search(lat, lon, year, photo_type="0", radius_km=10.0, limit=30, cache_dir="data/ardswc_meta_cache"):
    """依中心座標＋精確年份＋分類，從官方資料集找出鄰近候選照片，依距離排序。

    找不到結果就是找不到——不會為了湊出結果而放寬年份或半徑。
    回傳 (matches, meta)：meta 含是否為部分結果（partial）、實際掃描頁數等，供呼叫端誠實揭露。
    """
    year = int(year)
    exact_records, fetch_meta = _fetch_year_pages(photo_type, year, cache_dir)

    out = []
    for r in exact_records:
        rlat, rlon = r.get("Lat"), r.get("Lng")
        if rlat is None or rlon is None:
            continue
        dist = _haversine_km(lat, lon, rlat, rlon)
        if dist > radius_km:
            continue
        out.append({
            "event_id": r.get("EventID"), "county": r.get("County"), "town": r.get("Town"),
            "vill": r.get("Vill"), "disaster_name": r.get("DisasterName"), "photo_date": r.get("PhotoDate"),
            "description": r.get("Description"), "source": r.get("Source"),
            "media_url": r.get("FileName") or (f"{_MEDIA_BASE}/{r.get('EventID')}" if r.get("EventID") else None),
            "distance_km": round(dist, 2),
        })
    out.sort(key=lambda x: x["distance_km"])
    meta = dict(fetch_meta)
    meta["total_exact_year_records"] = len(exact_records)
    meta["photo_type_label"] = PHOTO_TYPE_LABELS.get(str(photo_type), str(photo_type))
    return out[:limit], meta
