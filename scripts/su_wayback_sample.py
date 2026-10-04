"""POC：抽 Slope Unit 代表樣本，抓 Esri Wayback 前／（中）／後期影像，做成「盲測」判讀圖。

前置：landslide_su_events.py（su_table.json、su_lab10.npy）。
Wayback 在此區影像稀疏（常只有 2017–2019 與 2022–2023，少數有 2025），所以不硬湊固定年份，而是：
  1. 每個候選 SU 查 Wayback 各拍攝日；每個影像若與某期年度圖層（影像日期中位數 2021-07-01／2022-06-23／2023-07-12／2025-03-01）
     相差 ≤ 365 天才算「可對位」，並記下對位的圖層年度。
  2. 前期＝最早的可對位影像、後期＝最晚的可對位影像（兩者對位到不同年度且相隔 ≥ 365 天）；中期取其間最接近中點者（有則附）。
  3. 以該「對位年度配對」的圖層面積重新分類（裸露 ≥0.5 ha 才算有；變化 >20%；與全期分類同一套門檻），
     依類別湊額：擴大 8、首次出現 5、縮減 5、持續裸露 5、消失 3、無崩塌對照（坡度>25°、面積≥5 ha）4，共 30。
抽樣候選依固定亂數種子打亂、樣本間距 ≥ 1.2 km。
Wayback 階段有時間上限（WB_MAX_MIN 分鐘）；湊不滿的類別改用 **Sentinel-2 備援**（Copernicus，10 m，真彩）：
  每個候選在四期圖層日期各 ±75 天內找雲量低的影像（場景雲量 ≤30%，再檢查裁切後的局部雲與空白），四期都找到才採用，
  以全期（2021→2024）的圖層類別分類（同一套門檻）。備援樣本在 key.json 標 method="S2"，Wayback 樣本標 "Wayback"。
判讀圖只畫 SU 外框，不畫崩塌圖層、也不標類別；圖名是隨機編號 V01…，對照表存 key.json（含對位年度與圖層類別），
先寫判讀（verdicts.json）再對照，避免看到答案再判讀。
輸出：data/biggis_interp/poc/su_validation/{V01.png…, key.json, skipped.json}
"""
import json
import random
import sys
import time
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from pyproj import Transformer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "webapp" / "change_detect_viewer"))
sys.path.insert(0, str(REPO / "scripts"))
import wayback_assist as WB  # noqa: E402
import sentinel_assist as S2  # noqa: E402
from landslide_incremental import RES, grid  # noqa: E402

POC = REPO / "data" / "biggis_interp" / "poc"
OUT = POC / "su_validation"
Z = 18
HALF_M = 380.0
SEED = 20261004
MIN_SEP_M = 1200.0
LAYER_DATE = {2021: date(2021, 7, 1), 2022: date(2022, 6, 23), 2023: date(2023, 7, 12), 2024: date(2025, 3, 1)}
MAX_GAP_D = 365
MINH, THR = 0.5, 0.20
WB_MAX_MIN = 20          # Wayback 階段時間上限（分鐘）
S2_WINDOW_D = 75
S2_MAX_CLOUD = 30.0
S2_LOCAL_CLOUD = 0.08
S2_HALF_M = 450.0
S2_MAX_CAND = 400        # 備援階段最多嘗試的候選數（額度與時間保護）
QUOTA = {"擴大": 8, "首次出現": 5, "縮減": 5, "持續裸露": 5, "消失": 3, "無崩塌": 4}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def d(ymd):
    return date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8]))


def align(frames):
    """每個影像對位到最近的圖層年度（≤365 天）。"""
    out = []
    for f in frames:
        y, g = min(((y, abs((d(f["date"]) - LAYER_DATE[y]).days)) for y in LAYER_DATE), key=lambda t: t[1])
        if g <= MAX_GAP_D:
            out.append(dict(f, layer_year=y, gap_days=g))
    return out


def pick(frames):
    al = align(frames)
    if len(al) < 2:
        return None, "可對位影像不足（需 ≥2 個與年度圖層相差 ≤365 天的拍攝日）"
    fa, fb = al[0], al[-1]
    if fa["layer_year"] == fb["layer_year"] or (d(fb["date"]) - d(fa["date"])).days < 365:
        return None, "前後期影像對位到同一年度或相隔不足一年"
    mids = [f for f in al if d(fa["date"]) < d(f["date"]) < d(fb["date"]) and f["layer_year"] not in (fa["layer_year"], fb["layer_year"])]
    mid_t = d(fa["date"]).toordinal() / 2 + d(fb["date"]).toordinal() / 2
    fm = min(mids, key=lambda f: abs(d(f["date"]).toordinal() - mid_t)) if mids else None
    return (fa, fm, fb), None


def pair_category(t, ya, yb):
    a, b = t["bare_ha"][str(ya)], t["bare_ha"][str(yb)]
    if a < MINH and b < MINH:
        return "無崩塌" if max(t["bare_ha"].values()) < MINH else None   # 其他年份有、這一對沒有：不抽
    if a < MINH:
        return "首次出現" if b >= 1.0 else None
    if b < MINH:
        return "消失" if a >= 1.0 else None
    r = (b - a) / a
    if r > THR:
        return "擴大" if b - a >= 1.0 else None
    if r < -THR:
        return "縮減" if a - b >= 1.0 else None
    return "持續裸露" if min(a, b) >= 2.0 else None


def outline_px(lab10, i, tf, bounds, wpx, hpx, tr):
    m = (lab10 == i).astype(np.uint8)
    rr, cc = np.nonzero(m)
    r0, c0 = max(rr.min() - 2, 0), max(cc.min() - 2, 0)
    sub = np.pad(m[r0:rr.max() + 3, c0:cc.max() + 3], 1)
    cs, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    minx, miny, maxx, maxy = bounds
    out = []
    for c in cs:
        pts = c[:, 0, :].astype(float) - 1
        X = tf.c + (pts[:, 0] + c0 + 0.5) * RES
        Y = tf.f - (pts[:, 1] + r0 + 0.5) * RES
        mx, my = tr.transform(X, Y)
        out.append(np.stack([(np.asarray(mx) - minx) / (maxx - minx) * wpx, (maxy - np.asarray(my)) / (maxy - miny) * hpx], 1).astype(np.int32))
    return out


def main():
    T = json.load(open(POC / "su_table.json", encoding="utf-8"))
    lab10 = np.load(POC / "su_lab10.npy")
    tf, _ = grid()
    to_ll = Transformer.from_crs(3826, 4326, always_xy=True)
    to_m = Transformer.from_crs(3826, 3857, always_xy=True)
    rng = random.Random(SEED)
    pool = [t for t in T if max(t["bare_ha"].values()) >= 1.0 or (t["slope_deg"] > 25 and t["area_ha"] >= 5.0 and max(t["bare_ha"].values()) < MINH)]
    rng.shuffle(pool)
    log("候選池", len(pool))
    OUT.mkdir(parents=True, exist_ok=True)
    chosen, skipped, got = [], [], {k: 0 for k in QUOTA}
    t0 = time.time()
    for t in pool:
        if all(got[k] >= QUOTA[k] for k in QUOTA):
            break
        if time.time() - t0 > WB_MAX_MIN * 60:
            log(f"Wayback 階段達 {WB_MAX_MIN} 分鐘上限，轉 Sentinel-2 備援")
            break
        if any(np.hypot(t["cx"] - c["cx"], t["cy"] - c["cy"]) < MIN_SEP_M for c in chosen):
            continue
        # 先用年度圖層估計「這個 SU 可能屬於哪類」，省掉不需要類別的網路查詢
        if all(got[k] >= QUOTA[k] for k in QUOTA if any(pair_category(t, ya, yb) == k for ya in LAYER_DATE for yb in LAYER_DATE if ya < yb)):
            continue
        lon, lat = to_ll.transform(t["cx"], t["cy"])
        try:
            frames = WB.list_frames(lat, lon, Z)
            pk, why = pick(frames)
            if why:
                skipped.append({"su_id": t["su_id"], "reason": why, "frames": [f["date"] for f in frames]}); log("略過", t["su_id"], why); continue
            ya, yb = pk[0]["layer_year"], pk[2]["layer_year"]
            cat = pair_category(t, ya, yb)
            if cat is None or got[cat] >= QUOTA[cat]:
                skipped.append({"su_id": t["su_id"], "reason": f"對位年度 {ya}→{yb} 的類別={cat}，額滿或不符"}); continue
            imgs = []
            for f in pk:
                imgs.append(None if f is None else (*WB.fetch_image(lat, lon, HALF_M, f["release"], Z), f))
        except Exception as e:  # noqa: BLE001
            skipped.append({"su_id": t["su_id"], "reason": repr(e)[:150]}); log("略過", t["su_id"], repr(e)[:100]); continue
        chosen.append(dict(t, method="Wayback", pair_years=[ya, yb], pair_category=cat, lat=lat, lon=lon, imgs=imgs,
                           tags=[x for x, f in zip(("前", "中", "後"), pk) if f]))
        got[cat] += 1
        log(f"{cat} {got[cat]}/{QUOTA[cat]} {t['su_id']} {ya}→{yb} 影像 {pk[0]['date']}/{pk[1]['date'] if pk[1] else '-'}/{pk[2]['date']}")
    log("Wayback 階段湊額：", got)
    # ── Sentinel-2 備援
    if not all(got[k] >= QUOTA[k] for k in QUOTA):
        tried = 0
        used = {c["su_id"] for c in chosen}
        for t in pool:
            if all(got[k] >= QUOTA[k] for k in QUOTA) or tried >= S2_MAX_CAND:
                break
            cat = pair_category(t, 2021, 2024)
            if t["su_id"] in used or cat is None or got[cat] >= QUOTA[cat]:
                continue
            if any(np.hypot(t["cx"] - c["cx"], t["cy"] - c["cy"]) < MIN_SEP_M for c in chosen):
                continue
            tried += 1
            lon, lat = to_ll.transform(t["cx"], t["cy"])
            try:
                picks = []
                for y, ld in LAYER_DATE.items():
                    sc = S2.search_scenes(lat, lon, date.fromordinal(ld.toordinal() - S2_WINDOW_D), date.fromordinal(ld.toordinal() + S2_WINDOW_D), S2_MAX_CLOUD, 600.0)
                    if not sc:
                        raise RuntimeError(f"{y} 年圖層日期 ±{S2_WINDOW_D} 天內沒有雲量≤{S2_MAX_CLOUD:.0f}% 的 S2 場景")
                    picks.append((y, sorted(sc, key=lambda x: (x["cloud"], abs(d(x["date"]).toordinal() - ld.toordinal())))[:3]))
                imgs = []
                for y, cands in picks:
                    ok = None
                    for sc in cands:
                        img, bounds = S2.fetch_image(lat, lon, S2_HALF_M, sc["date"], "TRUE_COLOR")
                        if S2.cloud_fraction(img) <= S2_LOCAL_CLOUD and S2.blank_fraction(img) < 0.02:
                            ok = (img, bounds, {"date": sc["date"], "res_m": 10, "layer_year": y, "scene_cloud": sc["cloud"], "source": "Sentinel-2 L2A"})
                            break
                    if ok is None:
                        raise RuntimeError(f"{y} 年：候選影像局部有雲或空白")
                    imgs.append(ok)
            except Exception as e:  # noqa: BLE001
                skipped.append({"su_id": t["su_id"], "reason": "S2 備援：" + repr(e)[:140]}); log("S2 略過", t["su_id"], repr(e)[:90]); continue
            chosen.append(dict(t, method="S2", pair_years=[2021, 2024], pair_category=cat, lat=lat, lon=lon, imgs=imgs, tags=["2021", "2022", "2023", "2025"]))
            got[cat] += 1
            used.add(t["su_id"])
            log(f"S2 {cat} {got[cat]}/{QUOTA[cat]} {t['su_id']} 影像 " + "/".join(i[2]["date"] for i in imgs))
        log("S2 備援嘗試", tried, "個候選")
    log("最終湊額：", got)
    order = list(range(len(chosen)))
    rng.shuffle(order)
    key = {}
    for n, j in enumerate(order, 1):
        t = chosen[j]
        vid = f"V{n:02d}"
        panels = []
        ims = [im for im in t["imgs"] if im is not None]
        PW = 640 if t["method"] == "Wayback" else 480
        for tag, im in zip(t["tags"], ims):
            img, bounds, f = im
            h_, w_ = img.shape[:2]
            if w_ < PW:                                        # S2：10 m 影像先放大再畫外框（外框以放大後像素座標計）
                img = cv2.resize(img, (PW, int(h_ * PW / w_)), interpolation=cv2.INTER_CUBIC)
                h_, w_ = img.shape[:2]
            vis = img.copy()
            for poly in outline_px(lab10, t["i"], tf, bounds, w_, h_, to_m):
                cv2.polylines(vis, [poly.reshape(-1, 1, 2)], True, (255, 255, 255), 2, cv2.LINE_AA)
            vis = cv2.resize(vis, (PW, int(h_ * PW / w_)), interpolation=cv2.INTER_AREA)
            bar = np.full((34, PW, 3), 30, np.uint8)
            cv2.putText(bar, f"{vid} {tag} {f['date']} {f.get('res_m') or '?'}m", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(np.vstack([bar, vis]))
        hh = max(p_.shape[0] for p_ in panels)
        panels = [np.vstack([p_, np.zeros((hh - p_.shape[0], p_.shape[1], 3), np.uint8)]) if p_.shape[0] < hh else p_ for p_ in panels]
        cv2.imwrite(str(OUT / f"{vid}.png"), np.hstack(panels))
        key[vid] = {k: v for k, v in t.items() if k != "imgs"}
        key[vid]["frames"] = [im[2] if im else None for im in t["imgs"]]
        key[vid].pop("tags", None)
    (OUT / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "skipped.json").write_text(json.dumps(skipped, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"完成：{len(chosen)} 個樣本、略過 {len(skipped)} 個 → {OUT}")


if __name__ == "__main__":
    main()
