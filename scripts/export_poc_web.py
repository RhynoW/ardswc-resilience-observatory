"""把五個 POC 區域的成果匯出成網頁互動展示用的靜態資料（webapp/change_detect_viewer/static/poc/）。
輸出：index.json（區域清單與摘要指標）、<id>.json（每區：範圍、事件窗口、指標、圖層 GeoJSON、標記點）、img/（花蓮目錄外大塊複核圖）。
圖層（WGS84、已簡化；單檔 <3 MB 以符合 HF 限制）：
  events      事件窗口內的事件型目錄多邊形（衛星判釋，非現地真值）
  s1_k2/3/4   Sentinel-1 變化偵測（降軌 VH；K 為穩健門檻倍數，事先固定）的 ≥0.3 ha 偵測塊
  s2_state    高雄 Sentinel-2（2025-03-25）基準偵測；s2_new 花蓮 Sentinel-2（事前 06-15→事後 10-11）新增
  official113 高雄官方 113 年度全臺崩塌地圖層（≥0.5 ha）
標記點：高雄的盲測與目視複核點（V、U、R）、花蓮目錄外大塊（O）。
所有數字取自既有評估輸出（eval*.json），不重新計算結論。
用法：python scripts/export_poc_web.py
"""
import json
import shutil
import sys
import warnings
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy import ndimage as ndi
from shapely.geometry import Polygon, box, mapping

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import area_run as AR  # noqa: E402
import area_run_s1 as ARS  # noqa: E402
import s2_area as A  # noqa: E402
import s1_area as S1  # noqa: E402
import reverse_detect as R  # noqa: E402
from landslide_incremental import grid as kgrid  # noqa: E402

WEB = REPO / "webapp" / "change_detect_viewer" / "static" / "poc"
AREAS = REPO / "data" / "biggis_interp" / "areas"
POC = REPO / "data" / "biggis_interp" / "poc"
to_ll = Transformer.from_crs(3826, 4326, always_xy=True)


def jload(p, default=None):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def vectorize(m, tf, min_cells, eps_px=1.2):
    """二值遮罩 → GeoJSON features（外環，簡化；面積 ha 與格數過濾）。"""
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    feats = []
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cs:
        if len(c) < 3:
            continue
        x, y = c[0][0]
        if areas[lab[y, x]] < min_cells:
            continue
        ap = cv2.approxPolyDP(c, eps_px, True)[:, 0, :].astype(np.float64)
        if len(ap) < 3:
            continue
        X, Y = tf.c + (ap[:, 0] + 0.5) * 10.0, tf.f - (ap[:, 1] + 0.5) * 10.0
        lon, lat = to_ll.transform(X, Y)
        ring = [[round(float(a), 5), round(float(b), 5)] for a, b in zip(lon, lat)]
        ring.append(ring[0])
        feats.append({"type": "Feature", "properties": {"ha": round(float(areas[lab[y, x]] * 0.01), 1)}, "geometry": {"type": "Polygon", "coordinates": [ring]}})
    return {"type": "FeatureCollection", "features": feats}


def events_geojson(bbox, lo, hi, evt_range):
    pj = jload(REPO / "data" / "biggis_interp" / "polygons.json")
    b = box(*bbox)
    by = {}
    for p in pj["polys"]:
        e = pj["events"][p[0]]
        if not (lo < e["date"] <= hi):
            continue
        xy = p[2]
        poly = Polygon(list(zip(xy[0::2], xy[1::2]))).buffer(0)
        if poly.is_empty or not poly.intersects(b):
            continue
        by.setdefault((e["date"], e["event"].split("_", 1)[-1]), []).append(poly)
    feats, summ = [], []
    for (d, n), polys in sorted(by.items()):
        a = float(gpd.GeoSeries(polys, crs=4326).to_crs(3826).area.sum() / 1e4)
        summ.append({"date": d, "event": n, "n": len(polys), "area_ha": round(a, 1)})
        for poly in polys:
            g = poly.simplify(0.00004, preserve_topology=True)
            if g.is_empty:
                continue
            feats.append({"type": "Feature", "properties": {"event": n, "date": d}, "geometry": json.loads(json.dumps(mapping(g), default=list))})
    return {"type": "FeatureCollection", "features": feats}, summ


def round_coords(o):
    if isinstance(o, float):
        return round(o, 5)
    if isinstance(o, (list, tuple)):
        return [round_coords(i) for i in o]
    if isinstance(o, dict):
        return {k: round_coords(v) for k, v in o.items()}
    return o


def s1_layers(tag):
    """回傳 {layer: geojson}, 指標（降軌 VH 的 K=2/3/4）。"""
    steep = A.slope(*A.grid()[:1], *A.grid()[1]) >= R.SLOPE_MIN
    tf, (H, W) = A.grid()
    key = "desc105"
    S1.ORBITS = {key: None}
    pre, dpre = S1.stack(key, "pre", "vh")
    post, dpost = S1.stack(key, "post", "vh")
    with np.errstate(all="ignore"):
        d = 10 * np.log10(np.maximum(post, 1e-6)) - 10 * np.log10(np.maximum(pre, 1e-6))
    region = np.isfinite(d) & steep & (pre > 1e-4) & (post > 1e-4)
    med = float(np.median(d[region]))
    mad = float(np.median(np.abs(d[region] - med))) * 1.4826
    out = {}
    for K in (2.0, 3.0, 4.0):
        new = R.post((d < med - K * mad) & region) & region
        new = ndi.median_filter(new.astype(np.uint8), size=3).astype(bool) & region
        out[f"s1_k{int(K)}"] = vectorize(new, tf, 30)
    return out, dpre, dpost, max(dpre), min(dpost)


def build_area(tag, name, label, bbox, evt_range, s1win, desc, event_note):
    out = {"id": tag, "name": name, "label": label, "bbox": bbox, "center": [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2], "desc": desc, "event_note": event_note, "layers": {}, "markers": [], "metrics": {}}
    if tag == "H":
        A.NAME = "hualien_barrier_lake"
        S1.PRE_WIN, S1.POST_WIN = ("2025-05-15", "2025-07-05"), ("2025-10-01", "2025-10-16")
    else:
        AR.setup(tag)
        S1.PRE_WIN, S1.POST_WIN = ARS.WIN[tag]
    layers, dpre, dpost, lo, hi = s1_layers(tag)
    out["layers"].update(layers)
    out["s1_dates"] = {"pre": dpre, "post": dpost}
    ev, summ = events_geojson(bbox, lo, hi, evt_range)
    out["layers"]["events"] = ev
    out["events_in_window"] = summ
    s1e = jload(AREAS / A.NAME / "s1" / "eval.json", {})
    run = (s1e.get("runs") or {}).get("desc105_vh", {})
    out["metrics"]["s1"] = {"orbit": "降軌 105、VH", "pre_dates": run.get("pre_dates"), "post_dates": run.get("post_dates"), "by_K": run.get("by_K"), "event_area_ha_in_region": run.get("event_area_ha_in_region")}
    s2e = jload(AREAS / A.NAME / "s2" / "eval.json", {})
    pairs = s2e.get("pairs") or {}
    rat = [v["ratio"] for v in pairs.values() if v.get("ratio") is not None]
    out["metrics"]["s2"] = {"n_pairs": len(pairs), "ratio_min": min(rat) if rat else None, "ratio_max": max(rat) if rat else None,
                            "new_ha_min": min((v["new_ha"] for v in pairs.values()), default=None), "new_ha_max": max((v["new_ha"] for v in pairs.values()), default=None),
                            "event_area_ha": next(iter(pairs.values()), {}).get("event_area_ha_in_region")}
    large = jload(AREAS / "s1_large_eval.json", {}).get("areas", {}).get(tag, {}).get("desc105")
    out["metrics"]["s1_large"] = large
    return out


def main():
    WEB.mkdir(parents=True, exist_ok=True)
    (WEB / "img").mkdir(exist_ok=True)
    index = []
    cfg = [
        ("H", "hualien", "② 花蓮（馬太鞍溪堰塞湖）", (121.1456, 23.5494, 121.4456, 23.8494), ("2025-05-01", "2026-06-30"), None,
         "2025-07 薇帕颱風降雨使上游崩塌擴大，形成堰塞湖；2025-09-23 溢流。事件窗口：2025-07-04～2025-10-02。", "事件目錄含 2025-07-06 丹娜絲薇帕（平均 11.8 ha／塊，含 688 ha 超大崩塌）與 2025-07-28 豪雨。"),
        ("A", "taroko", "③ A 太魯閣（2024-04-03 花蓮地震）", AR.CFG["A"]["bbox"], AR.CFG["A"]["evt_range"], None,
         "地震觸發的崩塌，大理岩／片岩峽谷。", "事件目錄 2,646 個多邊形、2,215 ha（平均 0.84 ha／塊）。"),
        ("B", "central", "④ B 中部山區（2023-08-03 卡努）", AR.CFG["B"]["bbox"], AR.CFG["B"]["evt_range"], None,
         "颱風降雨觸發，中央山脈板岩／變質岩。", "事件目錄 1,202 個多邊形、312 ha（平均 0.26 ha／塊）。"),
        ("C", "chiayi", "⑤ C 嘉義阿里山（2024-07-23 凱米）", AR.CFG["C"]["bbox"], AR.CFG["C"]["evt_range"], None,
         "颱風降雨觸發，西部麓山帶泥岩區（與高雄同類型）。", "事件目錄 484 個多邊形、291 ha（平均 0.60 ha／塊）。"),
    ]
    for tag, name, label, bbox, evr, _, desc, note in cfg:
        d = build_area(tag, name, label, bbox, evr, None, desc, note)
        if tag == "H":
            keyj = jload(AREAS / "hualien_barrier_lake" / "s1_review" / "key.json", {})
            for oid, k in keyj.items():
                src = AREAS / "hualien_barrier_lake" / "s1_review" / f"{oid}.jpg"
                if src.exists():
                    shutil.copy(src, WEB / "img" / f"{oid}.jpg")
                d["markers"].append({"id": oid, "kind": "O", "lat": k["lat"], "lon": k["lon"], "title": f"{oid}：Sentinel-1 目錄外大塊（{k['ha']} ha）", "text": f"平均坡度 {k['mean_slope_deg']}°；VH 平均下降 {k['mean_dB_diff']} dB。AI 初判在 Sentinel-2 上看不到明顯崩塌，尚待人工判讀。", "img": f"/static/poc/img/{oid}.jpg"})
            tf, shp = A.grid()
            for pair, fn in (("20250615→20251011", "new_20250615_20251011.npy"),):
                p = AREAS / "hualien_barrier_lake" / "s2" / fn
                if p.exists():
                    d["layers"]["s2_new"] = vectorize(np.load(p), tf, 30)
                    d["s2_new_pair"] = pair
        (WEB / f"{name}.json").write_text(json.dumps(round_coords(d), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(name, round((WEB / f"{name}.json").stat().st_size / 1e6, 2), "MB", {k: len(v["features"]) for k, v in d["layers"].items()}, flush=True)
        index.append({"id": tag, "name": name, "label": label, "bbox": bbox, "center": d["center"], "file": f"/static/poc/{name}.json"})
    # 高雄
    kh = {"id": "K", "name": "kaohsiung", "label": "① 高雄（桃源・那瑪夏・六龜・甲仙）", "bbox": [120.55, 22.95, 120.95, 23.35], "center": [23.15, 120.75],
          "desc": "本專案第一個 POC（約 1,800 km²）：官方年度圖層、事件目錄、Slope Unit、Wayback／UAV／Sentinel-2 的盲測與反向偵測。", "event_note": "事件目錄含 2021–2025 多場颱風豪雨；Sentinel-2 變化評估窗口 2024-04-04～2025-03-25。", "layers": {}, "markers": [], "metrics": {}}
    tf, (H, W) = kgrid()
    Z = np.load(POC / "incremental_masks.npz")
    kh["layers"]["official113"] = vectorize(Z["2024"], type("T", (), {"c": tf.c, "f": tf.f})(), 50)
    p = POC / "s2" / "det_20250325_A_fixed.npy"
    if p.exists():
        kh["layers"]["s2_state"] = vectorize(np.load(p), type("T", (), {"c": tf.c, "f": tf.f})(), 30)
    ev, summ = events_geojson(tuple(kh["bbox"]), "2024-04-04", "2025-03-25", None)
    kh["layers"]["events"] = ev
    kh["events_in_window"] = summ
    # 標記點
    hv = jload(POC / "su_validation" / "verdicts_human.json", {})
    av = jload(POC / "su_validation" / "verdicts.json", {})
    for vid, k in (jload(POC / "su_validation" / "key.json", {})).items():
        kh["markers"].append({"id": vid, "kind": "V", "lat": k["lat"], "lon": k["lon"], "title": f"{vid}：坡面盲測（{k['method']}）",
                              "text": f"官方圖層類別：{k['pair_category']}；AI 判讀：{av.get(vid, {}).get('verdict', '—')}；人工判讀：{hv.get(vid, {}).get('verdict', '—')}（信心 {hv.get(vid, {}).get('conf', '—')}）"})
    uh = jload(POC / "uav_validation" / "verdicts_human.json", {})
    ua = jload(POC / "uav_validation" / "verdicts.json", {})
    for uid, k in (jload(POC / "uav_validation" / "key.json", {})).items():
        kh["markers"].append({"id": uid, "kind": "U", "lat": k["lat"], "lon": k["lon"], "title": f"{uid}：UAV 對照盲測", "text": f"官方圖層類別：{k.get('layer_category')}；AI：{ua.get(uid, {}).get('verdict', '—')}；人工：{uh.get(uid, {}).get('verdict', '—')}"})
    rv = jload(POC / "reverse" / "review" / "verdicts.json", {})
    for rid, k in (jload(POC / "reverse" / "review" / "key.json", {})).items():
        lon, lat = to_ll.transform(k["cx"], k["cy"])
        kh["markers"].append({"id": rid, "kind": "R", "lat": round(lat, 5), "lon": round(lon, 5), "title": f"{rid}：偵測有、官方無（{k['area_ha']} ha）", "text": f"判讀（AI＋使用者人工更正）：{rv.get(rid, {}).get('class', '—')}"})
    kh["metrics"] = {
        "state_wayback": jload(POC / "reverse" / "eval.json", {}).get("A_fixed"),
        "state_s2": {k: v for k, v in (jload(POC / "s2" / "eval.json", {}).get("A_fixed", {}) or {}).items() if k in ("state_late_vs_2024layer", "change")},
        "buffer": jload(POC / "reverse" / "eval_buffer.json"),
        "su_sensitivity": jload(POC / "su_sens" / "sensitivity.json"),
        "human_blind": (jload(POC / "su_validation" / "score_verdicts_human.json", {}) or {}).get("overall"),
        "ai_blind": (jload(POC / "su_validation" / "score.json", {}) or {}).get("overall"),
        "change_wayback": jload(POC / "reverse" / "eval_change.json"),
    }
    (WEB / "kaohsiung.json").write_text(json.dumps(round_coords(kh), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("kaohsiung", round((WEB / "kaohsiung.json").stat().st_size / 1e6, 2), "MB", {k: len(v["features"]) for k, v in kh["layers"].items()}, len(kh["markers"]), "markers")
    index.insert(0, {"id": "K", "name": "kaohsiung", "label": kh["label"], "bbox": kh["bbox"], "center": kh["center"], "file": "/static/poc/kaohsiung.json"})
    (WEB / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
