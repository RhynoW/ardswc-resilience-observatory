"""小試驗：Sentinel-2 L2A 紅邊指標時間序列能否在坡面單元上分出「凱米颱風（2024-07-23）新增崩塌」與相似對照坡面。
參考：水保署技術電子報第 173 期（2026-10-15，鄭偉成）〈利用 Sentinel-2 紅邊指標追蹤 0403 地震後坡地變化〉——
  NDRE（B8A、B5）、MTCI（B4、B5、B6）；SCL 去雲與雲影；統一 20 m；固定樣區取有效像元中位數；缺值不補零；鄰近對照區。
本試驗把「固定樣區」換成 v2 坡面單元的 ≥15° 有效坡面，對照區換成坡向、坡度、高程相近的鄰近「無崩塌、無事件、無照片」坡面。
步驟：
  select ：選案例（凱米事件判釋多邊形在有效坡面內 ≥1 ha 的單元）與配對對照；限同一個 Sentinel-2 圖幅內的一個方框，以控制下載量。
  fetch  ：Planetary Computer sentinel-2-l2a（免費），2023-11-01～2025-03-31、圖幅雲量 <70%；讀方框視窗的 B04、B05、B06、B8A、SCL（20 m）。
  series ：每景每單元：SCL 有效（4 植生、5 裸地、6 水、7 未分類以外皆剔除：雲、雲影、陰影、雪、飽和、缺值）的像元比例；有效 ≥50% 才取中位數
           NDVI＝(B8A−B4)/(B8A+B4)、NDRE＝(B8A−B5)/(B8A+B5)、MTCI＝(B6−B5)/(B5−B4)（|B5−B4| 太小時不算）。
  eval   ：事件前（2023-11-01～2024-07-20）與事件後（2024-07-28～2025-03-31）中位數差 Δ；案例 vs 對照的 Δ 分布、配對差、AUC。
用法：python scripts/s2_rededge_pilot.py select|fetch|series|eval|all
輸出：data/biggis_interp/poc/rededge/（select.json、scenes/*.npz、series.json、eval.json、fig_*.png）
限制：單一事件、單一方框、樣本小；NDVI／NDRE／MTCI 共用波段，不是獨立證據；事件判釋多邊形本身是衛星判釋。
"""
import json
import sys
import time
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
POC = REPO / "data" / "biggis_interp" / "poc"
OUT = POC / "rededge"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
UA = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}
EVENT = "20240723_凱米颱風"
PRE = ("2023-11-01", "2024-07-20")
POST = ("2024-07-28", "2025-03-31")
BANDS = ("B04", "B05", "B06", "B8A", "SCL")
SCL_OK = (4, 5, 6, 7)
BOX_KM = 14.0
N_CASES = 15


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jget(u, data=None):
    return json.loads(urllib.request.urlopen(urllib.request.Request(u, data=data, headers=UA), timeout=120).read())


def grid20():
    L = np.load(POC / "su_labels_eff.npz")
    from rasterio.transform import from_origin
    return L["key20"], from_origin(float(L["x0"]), float(L["y0"]), 20.0, 20.0)


# ── select ─────────────────────────────────────────────────────────────
def select():
    import geopandas as gpd
    import rasterio
    from rasterio import features
    from landslide_su_events import load_events
    OUT.mkdir(parents=True, exist_ok=True)
    lab, tf = grid20()
    H, W = lab.shape
    n = int(lab.max()) + 1
    with rasterio.open(POC / "su_work" / "dem.tif") as r:
        dem = r.read(1).astype(np.float32)
    dem = np.where(dem > -9990, dem, np.nan)
    gy, gx = np.gradient(dem, 20.0)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    # 坡向（0＝北，順時針）；以單位向量平均
    asp = np.arctan2(-gx, gy)      # 下坡方向：列向南為正 → 北為 -gy
    m = lab >= 0
    cnt = np.bincount(lab[m], minlength=n)
    sx = np.bincount(lab[m], weights=np.sin(asp[m]), minlength=n)
    sy = np.bincount(lab[m], weights=np.cos(asp[m]), minlength=n)
    aspect = (np.degrees(np.arctan2(sx, sy)) + 360) % 360
    sl = np.bincount(lab[m], weights=np.nan_to_num(slope[m]), minlength=n) / np.maximum(cnt, 1)
    el = np.bincount(lab[m], weights=np.nan_to_num(dem[m]), minlength=n) / np.maximum(cnt, 1)
    rr, cc = np.nonzero(m)
    cx = np.bincount(lab[m], weights=tf.c + (cc + 0.5) * 20, minlength=n) / np.maximum(cnt, 1)
    cy = np.bincount(lab[m], weights=tf.f - (rr + 0.5) * 20, minlength=n) / np.maximum(cnt, 1)
    evs = load_events()
    in_any = np.zeros(n)
    case_ha = np.zeros(n)
    for name, (dt, g) in evs.items():
        em = features.rasterize(((geom, 1) for geom in g.geometry), out_shape=(H, W), transform=tf, fill=0, dtype="uint8").astype(bool)
        c = np.bincount(lab[m & em], minlength=n) * 0.04
        in_any += c
        if name == EVENT:
            case_ha = c
    meta = json.load(open(POC / "su_meta_eff.json", encoding="utf-8"))
    sid = {u["key"]: u["su_id"] for u in meta["units"]}
    T = {t["su_id"]: t for t in json.load(open(POC / "su_table_eff.json", encoding="utf-8"))}
    ev = {e["su_id"]: e for e in json.load(open(POC / "su_evidence.json", encoding="utf-8"))}
    poc_keys = [k for k in range(n) if cnt[k] > 0 and sid.get(k) in T]

    def has_photo(s):
        e = ev.get(s) or {}
        return bool((e.get("field_photos") or {}).get("classes"))

    cases = [k for k in poc_keys if case_ha[k] >= 1.0 and cnt[k] * 0.04 >= 3]
    ctrl_pool = [k for k in poc_keys if in_any[k] == 0 and T[sid[k]]["category"] == "無崩塌" and not has_photo(sid[k]) and cnt[k] * 0.04 >= 3]
    log("候選案例", len(cases), "對照池", len(ctrl_pool))
    # 方框：案例最密集的 BOX_KM 方框（以案例中心為方框中心，取涵蓋最多案例者）
    best = None
    for k in cases:
        inb = [j for j in cases if abs(cx[j] - cx[k]) <= BOX_KM * 500 and abs(cy[j] - cy[k]) <= BOX_KM * 500]
        if best is None or len(inb) > len(best[1]):
            best = (k, inb)
    k0, inb = best
    box = (cx[k0] - BOX_KM * 500, cy[k0] - BOX_KM * 500, cx[k0] + BOX_KM * 500, cy[k0] + BOX_KM * 500)
    inb = sorted(inb, key=lambda j: -case_ha[j])
    pool = [j for j in ctrl_pool if box[0] <= cx[j] <= box[2] and box[1] <= cy[j] <= box[3]]
    used, pairs = set(), []
    for k in inb:
        def adiff(a, b):
            d = abs(a - b) % 360
            return min(d, 360 - d)
        cand = [j for j in pool if j not in used and adiff(aspect[j], aspect[k]) <= 45 and abs(sl[j] - sl[k]) <= 5 and abs(el[j] - el[k]) <= 300]
        if not cand:
            continue
        j = min(cand, key=lambda j: np.hypot(cx[j] - cx[k], cy[j] - cy[k]))
        if np.hypot(cx[j] - cx[k], cy[j] - cy[k]) > 5000:
            continue
        used.add(j)
        pairs.append({"case": sid[k], "case_key": int(k), "ctrl": sid[j], "ctrl_key": int(j), "case_event_ha": round(float(case_ha[k]), 2),
                      "case_eff_ha": round(float(cnt[k] * 0.04), 1), "ctrl_eff_ha": round(float(cnt[j] * 0.04), 1),
                      "aspect": [round(float(aspect[k])), round(float(aspect[j]))], "slope": [round(float(sl[k]), 1), round(float(sl[j]), 1)],
                      "elev": [round(float(el[k])), round(float(el[j]))], "dist_m": round(float(np.hypot(cx[j] - cx[k], cy[j] - cy[k])))})
        if len(pairs) >= N_CASES:
            break
    out = {"event": EVENT, "box_tm2": [round(v, 1) for v in box], "n_candidates_in_box": len(inb), "pairs": pairs,
           "rule": "案例：凱米事件判釋多邊形在有效坡面內 ≥1 ha、有效坡面 ≥3 ha；對照：任何事件 0 ha、四年圖層類別無崩塌、無災害／媒體照片、有效坡面 ≥3 ha，"
                   "坡向差 ≤45°、坡度差 ≤5°、平均高程差 ≤300 m、距離 ≤5 km，取最近者；同一對照不重複使用。"}
    (OUT / "select.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    log("方框", out["box_tm2"], "方框內案例", len(inb), "配對", len(pairs))
    for p in pairs:
        log(p)


# ── fetch ──────────────────────────────────────────────────────────────
def box_ll(box):
    from pyproj import Transformer
    t = Transformer.from_crs(3826, 4326, always_xy=True)
    xs, ys = [box[0], box[2], box[0], box[2]], [box[1], box[1], box[3], box[3]]
    lon, lat = t.transform(xs, ys)
    return min(lon), min(lat), max(lon), max(lat)


def fetch():
    import cv2
    import rasterio
    from pyproj import Transformer
    sel = json.load(open(OUT / "select.json", encoding="utf-8"))
    box = sel["box_tm2"]
    lab, tf = grid20()
    c0 = int((box[0] - tf.c) // 20)
    r0 = int((tf.f - box[3]) // 20)
    c1 = int((box[2] - tf.c) // 20) + 1
    r1 = int((tf.f - box[1]) // 20) + 1
    H, W = r1 - r0, c1 - c0
    xs = tf.c + (np.arange(c0, c1) + 0.5) * 20
    ys = tf.f - (np.arange(r0, r1) + 0.5) * 20
    X, Y = np.meshgrid(xs, ys)
    lon0, lat0, lon1, lat1 = box_ll(box)
    body = json.dumps({"collections": ["sentinel-2-l2a"], "bbox": [lon0, lat0, lon1, lat1], "datetime": f"{PRE[0]}T00:00:00Z/{POST[1]}T23:59:59Z",
                       "query": {"eo:cloud_cover": {"lt": 70}}, "limit": 500}).encode()
    items = jget(STAC + "/search", body)["features"]
    # 只用涵蓋整個方框的圖幅：以 MGRS 圖幅分組，取景數最多的一個
    from collections import Counter
    tiles = Counter(f["properties"].get("s2:mgrs_tile") for f in items)
    log("景數", len(items), "圖幅", dict(tiles))
    sc = OUT / "scenes"
    sc.mkdir(parents=True, exist_ok=True)
    tok = {"t": jget(SAS)["token"], "at": time.time()}
    done = 0

    def token():
        if time.time() - tok["at"] > 1800:
            tok.update(t=jget(SAS)["token"], at=time.time())
        return tok["t"]

    cache_xy = {}

    def one(f):
        p = f["properties"]
        d = p["datetime"][:10]
        tile = p.get("s2:mgrs_tile")
        dst = sc / f"{d}_{tile}.npz"
        if dst.exists():
            return "skip"
        epsg = int(str(p.get("proj:code") or p.get("proj:epsg")).replace("EPSG:", ""))
        if epsg not in cache_xy:
            ux, uy = Transformer.from_crs(3826, epsg, always_xy=True).transform(X.ravel(), Y.ravel())
            cache_xy[epsg] = (np.asarray(ux).reshape(H, W), np.asarray(uy).reshape(H, W))
        ux, uy = cache_xy[epsg]
        arrs = {}
        for b in BANDS:
            href = f["assets"][b]["href"] + "?" + token()
            with rasterio.open(href) as src:
                col, row = (~src.transform) * (ux, uy)
                a0, a1 = max(int(np.floor(col.min())) - 2, 0), min(int(np.ceil(col.max())) + 3, src.width)
                b0, b1 = max(int(np.floor(row.min())) - 2, 0), min(int(np.ceil(row.max())) + 3, src.height)
                if a1 <= a0 or b1 <= b0:
                    return "nocover"
                arr = src.read(1, window=rasterio.windows.Window(a0, b0, a1 - a0, b1 - b0)).astype(np.float32)
            interp = cv2.INTER_NEAREST if b == "SCL" else cv2.INTER_AREA if b == "B04" else cv2.INTER_LINEAR
            if b == "B04":   # 10 m → 20 m：先 2×2 平均再取樣
                arr = cv2.resize(arr, (arr.shape[1] // 2, arr.shape[0] // 2), interpolation=cv2.INTER_AREA)
                col, row = col / 2, row / 2
                a0, b0 = a0 / 2, b0 / 2
                interp = cv2.INTER_LINEAR
            out = cv2.remap(arr, (col - a0 - 0.5).astype(np.float32), (row - b0 - 0.5).astype(np.float32), interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            arrs[b] = out.astype(np.uint16)
        np.savez_compressed(dst, **arrs, offset=float(p.get("s2:processing_baseline", "0") >= "04.00"))
        return "ok"

    def safe(f):
        for k in range(3):
            try:
                return one(f)
            except Exception as e:  # noqa: BLE001
                err = repr(e)[:120]
                time.sleep(5)
        return "fail " + err

    with ThreadPoolExecutor(max_workers=6) as ex:
        for f, r in zip(items, ex.map(safe, items)):
            done += 1
            if done % 20 == 0 or not r.startswith(("ok", "skip")):
                log(done, "/", len(items), f["properties"]["datetime"][:10], f["properties"].get("s2:mgrs_tile"), r)
    json.dump({"box_rc": [r0, c0, r1, c1], "n_items": len(items)}, open(OUT / "fetch_meta.json", "w"))


# ── series ─────────────────────────────────────────────────────────────
def series():
    sel = json.load(open(OUT / "select.json", encoding="utf-8"))
    fm = json.load(open(OUT / "fetch_meta.json"))
    r0, c0, r1, c1 = fm["box_rc"]
    lab, _ = grid20()
    sub = lab[r0:r1, c0:c1]
    keys = [p["case_key"] for p in sel["pairs"]] + [p["ctrl_key"] for p in sel["pairs"]]
    masks = {k: sub == k for k in keys}
    rows = []
    for f in sorted((OUT / "scenes").glob("*.npz")):
        d, tile = f.stem.split("_")
        z = np.load(f)
        scl = z["SCL"]
        off = 1000.0 if float(z["offset"]) else 0.0      # processing baseline ≥04.00：DN 含 +1000 偏移
        refl = {b: (z[b].astype(np.float32) - off) / 10000.0 for b in ("B04", "B05", "B06", "B8A")}
        ok = np.isin(scl, SCL_OK) & (z["B8A"] > 0)
        for k in keys:
            mk = masks[k]
            npx = int(mk.sum())
            if npx == 0:
                continue
            v = ok & mk
            frac = float(v.sum() / npx)
            rec = {"date": d, "tile": tile, "key": k, "valid_frac": round(frac, 3)}
            if frac >= 0.5:
                b4, b5, b6, b8 = (refl[b][v] for b in ("B04", "B05", "B06", "B8A"))
                ndvi = (b8 - b4) / np.maximum(b8 + b4, 1e-6)
                ndre = (b8 - b5) / np.maximum(b8 + b5, 1e-6)
                den = b5 - b4
                good = np.abs(den) > 0.005
                mtci = (b6[good] - b5[good]) / den[good]
                rec.update(ndvi=round(float(np.median(ndvi)), 4), ndre=round(float(np.median(ndre)), 4),
                           mtci=round(float(np.median(mtci)), 3) if good.mean() > 0.5 else None)
            rows.append(rec)
    (OUT / "series.json").write_text(json.dumps(rows), encoding="utf-8")
    ok = [r for r in rows if "ndvi" in r]
    log("單元×景", len(rows), "有效（≥50%）", len(ok), "日期數", len({r["date"] for r in ok}))


# ── eval ───────────────────────────────────────────────────────────────
def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    return round(float(gt), 3)


def evaluate():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sel = json.load(open(OUT / "select.json", encoding="utf-8"))
    rows = json.load(open(OUT / "series.json", encoding="utf-8"))
    by = {}
    for r in rows:
        if "ndvi" in r:
            by.setdefault(r["key"], []).append(r)
    res = {"event": EVENT, "pre": PRE, "post": POST, "pairs": [], "indices": {}}
    for idx in ("ndvi", "ndre", "mtci"):
        dc, dk, paired = [], [], []
        for p in sel["pairs"]:
            vals = {}
            for role in ("case", "ctrl"):
                rs = [r for r in by.get(p[f"{role}_key"], []) if r.get(idx) is not None]
                pre = [r[idx] for r in rs if PRE[0] <= r["date"] <= PRE[1]]
                post = [r[idx] for r in rs if POST[0] <= r["date"] <= POST[1]]
                vals[role] = (np.median(pre) if pre else None, np.median(post) if post else None, len(pre), len(post))
            if None in (vals["case"][0], vals["case"][1], vals["ctrl"][0], vals["ctrl"][1]):
                continue
            d_case = vals["case"][1] - vals["case"][0]
            d_ctrl = vals["ctrl"][1] - vals["ctrl"][0]
            dc.append(d_case)
            dk.append(d_ctrl)
            paired.append(d_case - d_ctrl)
            if idx == "ndvi":
                res["pairs"].append({"case": p["case"], "ctrl": p["ctrl"], "case_event_ha": p["case_event_ha"], "case_eff_ha": p["case_eff_ha"],
                                     "n_obs_case_pre_post": vals["case"][2:], "n_obs_ctrl_pre_post": vals["ctrl"][2:]})
        dc, dk, paired = np.array(dc), np.array(dk), np.array(paired)
        res["indices"][idx] = {"n_pairs": int(len(paired)), "delta_case_median": round(float(np.median(dc)), 4) if len(dc) else None,
                               "delta_ctrl_median": round(float(np.median(dk)), 4) if len(dk) else None,
                               "paired_diff_median": round(float(np.median(paired)), 4) if len(paired) else None,
                               "pairs_case_drop_more": int((paired < 0).sum()), "auc_drop": auc(-dc, -dk),
                               "per_pair_case_minus_ctrl": [round(float(x), 4) for x in paired]}
    # 案例的事件面積占有效坡面比例 vs NDVI 配對差（面積小的單元是否被稀釋）
    (OUT / "eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res["indices"], ensure_ascii=False, indent=1))
    # 圖：每個指標，案例與對照的時間序列（各對配對差），事件日垂直線
    fig, axs = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    import datetime as _dt
    for ax, idx in zip(axs, ("ndvi", "ndre", "mtci")):
        for p in sel["pairs"]:
            for role, col in (("case", "tab:red"), ("ctrl", "tab:blue")):
                rs = sorted([r for r in by.get(p[f"{role}_key"], []) if r.get(idx) is not None], key=lambda r: r["date"])
                ax.plot([_dt.date.fromisoformat(r["date"]) for r in rs], [r[idx] for r in rs], "-", color=col, alpha=0.25, lw=0.8)
        ax.axvline(_dt.date(2024, 7, 23), color="k", ls="--", lw=1)
        ax.set_ylabel(idx.upper())
        if idx == "mtci":
            ax.set_ylim(-1, 8)
    axs[0].set_title("red = case SUs (Gaemi 2024-07-23 event polygons >=1 ha), blue = matched controls; SCL-masked medians, valid>=50%")
    plt.tight_layout()
    plt.savefig(OUT / "fig_series.png", dpi=80)


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("select", select), ("fetch", fetch), ("series", series), ("eval", evaluate)):
        if step in (name, "all"):
            fn()
