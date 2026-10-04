"""POC：以 BigGIS「UAV空拍正射影像圖磚」（政府資料開放授權第 1 版）盤點每個 Slope Unit 的 UAV 歷史涵蓋，
並為有 ≥2 個不同年份的 SU 產出前／（中）／後期對照圖（只畫 SU 外框，盲測編號 U01…）。

前置：landslide_su_events.py（su_table.json、su_lab10.npy）。
階段：
  inventory  下載（快取）UAV 清單 → 每個 SU 以「質心落在影像範圍框內」找涵蓋，輸出 uav_su_inventory.json 與 uav_su_summary.json。
             範圍框只是外接矩形，不保證影像填滿；實際有無影像要在 panels 階段抓圖磚確認。
  panels     對「≥2 個相隔 ≥365 天的拍攝日」的 SU 依圖層類別分層抽樣，抓 z18 圖磚拼接裁切，輸出 uav_validation/U01.png…、key.json、skipped.json。
圖磚網址：<AccessURL>{z}/{y}/{x}.jpg（BigGIS 清單 API https://data.geodac.tw/geoinfo_api/api/ardswc/tiles/ortho）。
限制：UAV 只在已知災區或監控區飛行，涵蓋不是隨機，樣本偏向災區；同日多筆取質心最靠近影像中心者；
      拍攝季節、光線與解析度因任務而異；前後期不是剛好對到年度圖層日期，類別以「對位到的年度」配對（相隔 ≤365 天），對不到則只記錄 2021→2024 類別。
用法：python scripts/su_uav_inventory.py inventory | panels [--n 30]
"""
import json
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from pyproj import Transformer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from landslide_incremental import RES, grid  # noqa: E402
from su_wayback_sample import LAYER_DATE, MAX_GAP_D, MIN_SEP_M, outline_px, pair_category  # noqa: E402

POC = REPO / "data" / "biggis_interp" / "poc"
OUT = POC / "uav_validation"
LIST_URL = "https://data.geodac.tw/geoinfo_api/api/ardswc/tiles/ortho"
LIST_CACHE = POC / "uav_ortho_list.json"
Z = 18
HALF_M = 380.0
SEED = 20261004
QUOTA = {"擴大": 7, "首次出現": 4, "縮減": 5, "持續裸露": 5, "消失": 3, "短暫出現": 2, "無崩塌": 4}   # 以 SU 全期類別（su_table category）分層
K = ("WestBoundLongitude", "EastBoundLongitude", "SouthBoundLatitude", "NorthBoundLatitude")
UA = {"User-Agent": "Mozilla/5.0"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_list(refresh=False):
    if LIST_CACHE.exists() and not refresh:
        rows = json.load(open(LIST_CACHE, encoding="utf-8"))
    else:
        req = urllib.request.Request(LIST_URL, headers=UA)
        rows = json.loads(urllib.request.urlopen(req, timeout=120).read().decode("utf-8"))
        LIST_CACHE.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    good, bad = [], []
    for r in rows:
        dt = r.get("Date")
        if not dt:
            import re
            m = re.search(r"/(\d{8})_", r.get("AccessURL") or "")
            dt = f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}" if m else None
        try:
            date.fromisoformat(dt)
        except (TypeError, ValueError):
            bad.append(r.get("Id")); continue
        if dt > date.today().isoformat() or any(r.get(k) is None for k in K):
            bad.append(r.get("Id")); continue          # 日期在未來（登錄錯誤）或缺範圍
        good.append(dict(r, Date=dt))
    return good, bad


def d(s):
    return date.fromisoformat(s)


def cover(rows, lon, lat):
    hit = [r for r in rows if r[K[0]] <= lon <= r[K[1]] and r[K[2]] <= lat <= r[K[3]]]
    by = {}
    for r in hit:   # 同日多筆：取質心最靠近影像中心者
        c = np.hypot(lon - r["CenterLongitude"], lat - r["CenterLatitude"])
        if r["Date"] not in by or c < by[r["Date"]][0]:
            by[r["Date"]] = (c, r)
    return [by[k][1] for k in sorted(by)]


def align_year(dt):
    y, g = min(((y, abs((d(dt) - LAYER_DATE[y]).days)) for y in LAYER_DATE), key=lambda t: t[1])
    return (y, g) if g <= MAX_GAP_D else (None, g)


def pick_pair(cov):
    """前＝最早、後＝最晚（相隔 ≥365 天）；中＝其間最接近中點且距兩端 ≥90 天者。"""
    if len(cov) < 2 or (d(cov[-1]["Date"]) - d(cov[0]["Date"])).days < 365:
        return None
    a, b = cov[0], cov[-1]
    mid_t = (d(a["Date"]).toordinal() + d(b["Date"]).toordinal()) / 2
    mids = [r for r in cov if (d(r["Date"]) - d(a["Date"])).days >= 90 and (d(b["Date"]) - d(r["Date"])).days >= 90]
    m = min(mids, key=lambda r: abs(d(r["Date"]).toordinal() - mid_t)) if mids else None
    return a, m, b


def pick_pair_aligned(cov):
    """優先在「可對位到年度圖層（≤365 天）」的拍攝日中取最早／最晚（對位到不同年度）；沒有才退回任意最早／最晚。
    回傳 (pair, aligned)。"""
    al = [r for r in cov if align_year(r["Date"])[0] is not None]
    if len(al) >= 2:
        a, b = al[0], al[-1]
        if align_year(a["Date"])[0] != align_year(b["Date"])[0] and (d(b["Date"]) - d(a["Date"])).days >= 365:
            mid_t = (d(a["Date"]).toordinal() + d(b["Date"]).toordinal()) / 2
            mids = [r for r in al if align_year(r["Date"])[0] not in (align_year(a["Date"])[0], align_year(b["Date"])[0])]
            m = min(mids, key=lambda r: abs(d(r["Date"]).toordinal() - mid_t)) if mids else None
            return (a, m, b), True
    pk = pick_pair(cov)
    return (pk, False) if pk else (None, False)


def inventory():
    rows, bad = load_list()
    T = json.load(open(POC / "su_table.json", encoding="utf-8"))
    to_ll = Transformer.from_crs(3826, 4326, always_xy=True)
    inv, nyears = [], {}
    for t in T:
        lon, lat = to_ll.transform(t["cx"], t["cy"])
        cov = cover(rows, lon, lat)
        yrs = sorted({r["Date"][:4] for r in cov})
        pr = pick_pair(cov)
        inv.append({"i": t["i"], "su_id": t["su_id"], "category": t["category"], "area_ha": t["area_ha"], "lat": round(lat, 6), "lon": round(lon, 6),
                    "dates": [r["Date"] for r in cov], "years": yrs, "has_pair": pr is not None})
        nyears[t["su_id"]] = len(yrs)
    n = len(inv)
    cats = sorted({x["category"] for x in inv})
    summ = {"uav_records_used": len(rows), "uav_records_dropped": len(bad), "su_total": n,
            "su_with_any_uav": sum(1 for x in inv if x["dates"]),
            "su_with_2plus_years": sum(1 for x in inv if len(x["years"]) >= 2),
            "su_with_pair_ge365d": sum(1 for x in inv if x["has_pair"]),
            "by_category": {c: {"su": sum(1 for x in inv if x["category"] == c), "any_uav": sum(1 for x in inv if x["category"] == c and x["dates"]),
                                "pair": sum(1 for x in inv if x["category"] == c and x["has_pair"])} for c in cats},
            "note": "涵蓋以 SU 質心落在 UAV 範圍框內判定；範圍框不保證影像填滿"}
    (POC / "uav_su_inventory.json").write_text(json.dumps(inv, ensure_ascii=False), encoding="utf-8")
    (POC / "uav_su_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))


# ── 圖磚拼接
def fetch_tile(url):
    for _ in range(3):
        try:
            b = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read()
            im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            return im
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except Exception:  # noqa: BLE001
            time.sleep(1)
    return None


def merc(lon, lat):
    import math
    x = (lon + 180) / 360
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2
    return x, y


def mosaic(base, lat, lon, half_m, z):
    """回傳 (BGR 影像, (minx,miny,maxx,maxy) EPSG:3857 邊界, 有效像素比例)。"""
    import math
    n = 2 ** z
    mpp = 40075016.686 / (256 * n)
    hw = half_m / math.cos(math.radians(lat))               # 3857 公尺（Mercator 放大）
    cx, cy = merc(lon, lat)
    px, py = cx * n * 256, cy * n * 256
    r = hw / mpp
    x0, x1, y0, y1 = int(px - r), int(px + r), int(py - r), int(py + r)
    tiles = [(tx, ty) for ty in range(y0 // 256, y1 // 256 + 1) for tx in range(x0 // 256, x1 // 256 + 1)]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda t: fetch_tile(f"{base}{z}/{t[1]}/{t[0]}.jpg"), tiles))
    W, H = (x1 // 256 + 1 - x0 // 256) * 256, (y1 // 256 + 1 - y0 // 256) * 256
    canvas = np.zeros((H, W, 3), np.uint8)
    valid = np.zeros((H, W), bool)
    for (tx, ty), im in zip(tiles, res):
        if im is None:
            continue
        oy, ox = (ty - y0 // 256) * 256, (tx - x0 // 256) * 256
        canvas[oy:oy + 256, ox:ox + 256] = im
        valid[oy:oy + 256, ox:ox + 256] = (im.sum(axis=2) > 30) & (im.min(axis=2) < 250)
    ox0, oy0 = x0 - (x0 // 256) * 256, y0 - (y0 // 256) * 256
    crop = canvas[oy0:oy0 + (y1 - y0), ox0:ox0 + (x1 - x0)]
    vcrop = valid[oy0:oy0 + (y1 - y0), ox0:ox0 + (x1 - x0)]
    # EPSG:3857 邊界
    ox_m = (px - r) * mpp - 20037508.343
    oy_m = 20037508.343 - (py + r) * mpp
    bounds = (ox_m, oy_m, ox_m + 2 * r * mpp, oy_m + 2 * r * mpp)
    return crop, bounds, float(vcrop.mean())


def panels(n_total):
    inv = json.load(open(POC / "uav_su_inventory.json", encoding="utf-8"))
    rows, _ = load_list()
    T = {t["i"]: t for t in json.load(open(POC / "su_table.json", encoding="utf-8"))}
    lab10 = np.load(POC / "su_lab10.npy")
    tf, _ = grid()
    to_ll = Transformer.from_crs(3826, 4326, always_xy=True)
    to_m = Transformer.from_crs(3826, 3857, always_xy=True)
    rng = random.Random(SEED)
    def okpool(x):
        t = T[x["i"]]
        if not x["has_pair"]:
            return False
        return x["category"] != "無崩塌" or (t["slope_deg"] > 25 and t["area_ha"] >= 5.0)
    pool = [x for x in inv if okpool(x)]
    rng.shuffle(pool)
    # 可對位的優先（圖層類別才有意義）
    pre = {x["su_id"]: pick_pair_aligned(cover(rows, x["lon"], x["lat"]))[1] for x in pool}
    pool.sort(key=lambda x: not pre[x["su_id"]])
    log("有前後期配對的候選 SU", len(pool), "其中可對位年度圖層", sum(pre.values()))
    OUT.mkdir(parents=True, exist_ok=True)
    chosen, skipped = [], []
    got = {k: 0 for k in QUOTA}
    for x in pool:
        if sum(got.values()) >= n_total:
            break
        t = T[x["i"]]
        if any(np.hypot(t["cx"] - c["t"]["cx"], t["cy"] - c["t"]["cy"]) < MIN_SEP_M for c in chosen):
            continue
        cat = x["category"]
        if cat not in QUOTA or got[cat] >= QUOTA[cat]:
            continue
        cov = cover(rows, x["lon"], x["lat"])
        pk, aligned = pick_pair_aligned(cov)
        if pk is None:
            continue
        ya, _ = align_year(pk[0]["Date"])
        yb, _ = align_year(pk[2]["Date"])
        lcat = pair_category(t, ya, yb) if aligned else None
        stratum = "對位年度" if aligned else "未對位（拍攝日不在圖層日期±365天內）"
        imgs, bad = [], None
        for r in pk:
            if r is None:
                imgs.append(None); continue
            try:
                im, bounds, vf = mosaic(r["AccessURL"], x["lat"], x["lon"], HALF_M, Z)
            except Exception as e:  # noqa: BLE001
                bad = f"抓圖失敗 {repr(e)[:80]}"; break
            if vf < 0.6:
                bad = f"{r['Date']} 圖磚有效像素僅 {vf:.0%}（範圍框內未涵蓋）"; break
            imgs.append((im, bounds, r, vf))
        if bad:
            skipped.append({"su_id": t["su_id"], "reason": bad}); log("略過", t["su_id"], bad); continue
        chosen.append({"t": t, "x": x, "cat": cat, "lcat": lcat, "aligned": aligned, "stratum": stratum, "pair_years": [ya, yb] if aligned else None, "imgs": imgs})
        got[cat] += 1
        log(f"{cat} {got[cat]}/{QUOTA[cat]} {t['su_id']} {pk[0]['Date']}/{pk[1]['Date'] if pk[1] else '-'}/{pk[2]['Date']}")
    log("湊額：", got)
    order = list(range(len(chosen)))
    rng.shuffle(order)
    key = {}
    PW = 640
    for nn, j in enumerate(order, 1):
        c = chosen[j]
        t = c["t"]
        uid = f"U{nn:02d}"
        pans = []
        for tag, im in zip(("1", "2", "3"), [im for im in c["imgs"] if im is not None]):
            img, bounds, r, vf = im
            h_, w_ = img.shape[:2]
            vis = img.copy()
            for poly in outline_px(lab10, t["i"], tf, bounds, w_, h_, to_m):
                cv2.polylines(vis, [poly.reshape(-1, 1, 2)], True, (255, 255, 255), 2, cv2.LINE_AA)
            vis = cv2.resize(vis, (PW, int(h_ * PW / w_)), interpolation=cv2.INTER_AREA)
            bar = np.full((34, PW, 3), 30, np.uint8)
            cv2.putText(bar, f"{uid} #{len(pans) + 1} {r['Date']} UAV", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            pans.append(np.vstack([bar, vis]))
        hh = max(p_.shape[0] for p_ in pans)
        pans = [np.vstack([p_, np.zeros((hh - p_.shape[0], p_.shape[1], 3), np.uint8)]) if p_.shape[0] < hh else p_ for p_ in pans]
        cv2.imwrite(str(OUT / f"{uid}.png"), np.hstack(pans))
        key[uid] = {k: v for k, v in t.items()}
        key[uid].update({"method": "UAV", "layer_category": c["cat"], "pair_layer_category": c["lcat"], "aligned": c["aligned"], "stratum": c["stratum"], "pair_years": c["pair_years"], "lat": c["x"]["lat"], "lon": c["x"]["lon"],
                         "frames": [None if im is None else {"date": im[2]["Date"], "title": im[2]["Title"], "project": im[2].get("ProjectName"), "id": im[2]["Id"], "valid_frac": round(im[3], 2)} for im in c["imgs"]]})
    (OUT / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "skipped.json").write_text(json.dumps(skipped, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"完成：{len(chosen)} 個樣本、略過 {len(skipped)} 個 → {OUT}")


if __name__ == "__main__":
    a = sys.argv[1:] or ["inventory"]
    if a[0] == "inventory":
        inventory()
    elif a[0] == "panels":
        panels(int(a[a.index("--n") + 1]) if "--n" in a else 30)
    else:
        sys.exit(__doc__)
