"""E3「近期衛星變化」改版：Sentinel-2 L2A 反射率 NDRE 時間序列（取代 8-bit 真彩色兩期規則）。
依據：小試驗 s2_rededge_pilot.py（凱米颱風 15 對案例／對照：持續下降規則抓到 6/15、對照誤報 0/15；坡面中位數約只看得到占有效坡面兩成以上的崩塌；
NDVI、NDRE、MTCI 抓到同一批，選 NDRE 為唯一主指標——B5、B8A 都是 20 m 原生解析度，不需重採樣 10 m 的 B4）。
參考：水保署技術電子報第 173 期〈利用 Sentinel-2 紅邊指標追蹤 0403 地震後坡地變化〉（SCL 去雲、20 m、固定樣區取有效像元中位數、缺值不補零）。
步驟：
  fetch ：Planetary Computer sentinel-2-l2a（免費），POC 範圍，2023-11-01～2026-09-30、圖幅雲量 <70%；每景讀 B05、B8A、SCL（20 m），
          直接算每個坡面（v2 有效坡面）的有效像元比例與 NDRE 中位數，只存坡面統計（poc/s2ts/items/<日期>_<圖幅>.npz），不存影像。
  build ：同日多圖幅取有效比例高者；有效 ≥50% 的觀測才用。
          基準期 2023-11-01～2024-06-30（≥3 筆）；之後任一日 t 起，前 6 筆有效觀測至少 3 筆低於「基準中位數 − 0.08」，且 t 之後全部觀測的中位數也低於 → 持續下降，記第一個 t。
          另記最近 6 個月（資料最後日往前 183 天）的中位數相對基準的差。輸出 poc/s2ts/su_ndre.json。
門檻 0.08 是小試驗看過資料後訂的，未獨立驗證；只用來「請人看」。
用法：python scripts/s2_ndre_series.py fetch|build|all
"""
import json
import sys
import time
import urllib.request
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
POC = REPO / "data" / "biggis_interp" / "poc"
OUT = POC / "s2ts"
ITEMS = OUT / "items"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
UA = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}
WIN = ("2023-11-01", "2026-09-30")
BASE = ("2023-11-01", "2024-06-30")
SCL_OK = (4, 5, 6, 7)
THR = 0.08
MIN_VALID = 0.5


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jget(u, data=None):
    return json.loads(urllib.request.urlopen(urllib.request.Request(u, data=data, headers=UA), timeout=120).read())


def su_key():
    """POC 坡面 su_id → su_labels_eff.npz 的標籤值（注意：su_table 的 i 是另一套連續索引，不能混用）。"""
    M = {u["su_id"]: u["key"] for u in json.load(open(POC / "su_meta_eff.json", encoding="utf-8"))["units"]}
    return {t["su_id"]: M[t["su_id"]] for t in json.load(open(POC / "su_table_eff.json", encoding="utf-8"))}


def labels():
    L = np.load(POC / "su_labels_eff.npz")
    lab = L["key20"]
    x0, y0 = float(L["x0"]), float(L["y0"])
    keys = sorted(su_key().values())
    # 只保留 POC 單元，裁到其外包框
    m = np.isin(lab, keys)
    rr, cc = np.nonzero(m)
    r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
    sub = np.where(m[r0:r1, c0:c1], lab[r0:r1, c0:c1], -1)
    return sub, x0 + c0 * 20.0, y0 - r0 * 20.0, int(lab.max()) + 1


def search():
    from pyproj import Transformer
    sub, x0, y0, _ = labels()
    H, W = sub.shape
    t = Transformer.from_crs(3826, 4326, always_xy=True)
    lon, lat = t.transform([x0, x0 + W * 20, x0, x0 + W * 20], [y0, y0, y0 - H * 20, y0 - H * 20])
    body = {"collections": ["sentinel-2-l2a"], "bbox": [min(lon), min(lat), max(lon), max(lat)], "datetime": f"{WIN[0]}T00:00:00Z/{WIN[1]}T23:59:59Z",
            "query": {"eo:cloud_cover": {"lt": 70}}, "limit": 1000}
    j = jget(STAC + "/search", json.dumps(body).encode())
    feats = j["features"]
    while True:
        nx = [l_ for l_ in j.get("links", []) if l_.get("rel") == "next"]
        if not nx:
            break
        n_ = nx[0]
        j = jget(n_["href"], json.dumps(n_.get("body", body)).encode()) if n_.get("method", "GET") == "POST" else jget(n_["href"])
        feats += j["features"]
    return feats


def grouped_median(vals, labs, n):
    o = np.lexsort((vals, labs))
    v, l_ = vals[o], labs[o]
    cnt = np.bincount(l_, minlength=n)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    med = np.full(n, np.nan, np.float32)
    has = cnt > 0
    med[has] = v[start[has] + (cnt[has] - 1) // 2]
    return med


def fetch():
    import cv2
    import rasterio
    from pyproj import Transformer
    ITEMS.mkdir(parents=True, exist_ok=True)
    sub, x0, y0, n = labels()
    H, W = sub.shape
    m = sub >= 0
    tot = np.bincount(sub[m], minlength=n)
    X, Y = np.meshgrid(x0 + (np.arange(W) + 0.5) * 20, y0 - (np.arange(H) + 0.5) * 20)
    feats = search()
    log("景數", len(feats), "網格", sub.shape)
    tok = {"t": jget(SAS)["token"], "at": time.time()}
    xy = {}

    def token():
        if time.time() - tok["at"] > 1800:
            tok.update(t=jget(SAS)["token"], at=time.time())
        return tok["t"]

    def one(f):
        p = f["properties"]
        d, tile = p["datetime"][:10], p.get("s2:mgrs_tile")
        dst = ITEMS / f"{d}_{tile}.npz"
        if dst.exists():
            return "skip"
        epsg = int(str(p.get("proj:code") or p.get("proj:epsg")).replace("EPSG:", ""))
        if epsg not in xy:
            ux, uy = Transformer.from_crs(3826, epsg, always_xy=True).transform(X.ravel(), Y.ravel())
            xy[epsg] = (np.asarray(ux, np.float64).reshape(H, W), np.asarray(uy, np.float64).reshape(H, W))
        ux, uy = xy[epsg]
        arr = {}
        for b in ("SCL", "B05", "B8A"):
            with rasterio.open(f["assets"][b]["href"] + "?" + token()) as src:
                col, row = (~src.transform) * (ux, uy)
                a0, a1 = max(int(np.floor(np.nanmin(col))) - 2, 0), min(int(np.ceil(np.nanmax(col))) + 3, src.width)
                b0, b1 = max(int(np.floor(np.nanmin(row))) - 2, 0), min(int(np.ceil(np.nanmax(row))) + 3, src.height)
                if a1 <= a0 or b1 <= b0:
                    np.savez_compressed(dst, valid=np.zeros(n, np.float16), ndre=np.full(n, np.nan, np.float16))
                    return "nocover"
                a = src.read(1, window=rasterio.windows.Window(a0, b0, a1 - a0, b1 - b0)).astype(np.float32)
            arr[b] = cv2.remap(a, (col - a0 - 0.5).astype(np.float32), (row - b0 - 0.5).astype(np.float32),
                               cv2.INTER_NEAREST if b == "SCL" else cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        off = 1000.0 if str(p.get("s2:processing_baseline", "0")) >= "04.00" else 0.0
        b5, b8 = (arr["B05"] - off) / 1e4, (arr["B8A"] - off) / 1e4
        ok = m & np.isin(arr["SCL"], SCL_OK) & (arr["B8A"] > 0) & (arr["B05"] > 0)
        valid = np.bincount(sub[ok], minlength=n) / np.maximum(tot, 1)
        ndre = (b8 - b5) / np.maximum(b8 + b5, 1e-6)
        med = grouped_median(ndre[ok].astype(np.float32), sub[ok], n)
        np.savez_compressed(dst, valid=valid.astype(np.float16), ndre=med.astype(np.float16))
        return f"ok valid>=0.5: {int((valid >= MIN_VALID).sum())}"

    def safe(f):
        err = ""
        for _ in range(3):
            try:
                return one(f)
            except Exception as e:  # noqa: BLE001
                err = repr(e)[:150]
                time.sleep(8)
        return "fail " + err

    done = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        for f, r in zip(feats, ex.map(safe, feats)):
            done += 1
            if done % 10 == 0 or r.startswith("fail"):
                log(done, "/", len(feats), f["properties"]["datetime"][:10], f["properties"].get("s2:mgrs_tile"), r)


def build():
    _, _, _, n = labels()
    T = json.load(open(POC / "su_table_eff.json", encoding="utf-8"))
    by_date = {}
    for f in sorted(ITEMS.glob("*.npz")):
        d = f.stem.split("_")[0]
        z = np.load(f)
        v, nd = z["valid"].astype(np.float32), z["ndre"].astype(np.float32)
        if d in by_date:     # 同日多圖幅：每個坡面取有效比例高者
            pv, pn = by_date[d]
            take = v > pv
            by_date[d] = (np.where(take, v, pv), np.where(take, nd, pn))
        else:
            by_date[d] = (v, nd)
    dates = sorted(by_date)
    V = np.stack([by_date[d][0] for d in dates])
    N = np.stack([by_date[d][1] for d in dates])
    N[V < MIN_VALID] = np.nan
    last = dates[-1]
    import datetime as _dt
    recent0 = (_dt.date.fromisoformat(last) - _dt.timedelta(days=183)).isoformat()
    months = sorted({d[:7] for d in dates})
    out = {}
    stats = defaultdict(int)
    K = su_key()
    for t in T:
        i = K[t["su_id"]]
        s = N[:, i]
        okd = [(d, float(x)) for d, x in zip(dates, s) if np.isfinite(x)]
        base = [x for d, x in okd if BASE[0] <= d <= BASE[1]]
        after = [(d, x) for d, x in okd if d > BASE[1]]
        rec = {"n_obs": len(okd), "n_base": len(base), "n_after": len(after)}
        if len(base) >= 3 and len(after) >= 6:
            b = float(np.median(base))
            rec["base"] = round(b, 3)
            drop = None
            for k in range(len(after)):
                nxt = [x for _, x in after[k:k + 6]]
                rest = [x for _, x in after[k:]]
                if len(nxt) >= 3 and sum(x < b - THR for x in nxt) >= 3 and np.median(rest) < b - THR:
                    drop = {"date": after[k][0], "delta": round(float(np.median(rest) - b), 3)}
                    break
            rec["drop"] = drop
            rec_recent = [x for d, x in okd if d >= recent0]
            rec["recent_delta"] = round(float(np.median(rec_recent) - b), 3) if len(rec_recent) >= 3 else None
            stats["drop" if drop else "no_drop"] += 1
        else:
            rec["drop"] = None
            stats["insufficient"] += 1
        mm = defaultdict(list)
        for d, x in okd:
            mm[d[:7]].append(x)
        rec["monthly"] = [round(float(np.median(mm[mo])), 3) if mo in mm else None for mo in months]
        out[t["su_id"]] = rec
    res = {"index": "NDRE (B8A,B5) L2A, SCL 4-7, valid>=0.5, SU effective-slope median", "window": [dates[0], last], "base": BASE, "thr": THR,
           "recent_from": recent0, "n_dates": len(dates), "months": months, "stats": dict(stats), "su": out}
    (OUT / "su_ndre.json").write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    log("日期", len(dates), dates[0], "→", last, dict(stats))


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("fetch", "all"):
        fetch()
    if step in ("build", "all"):
        build()
