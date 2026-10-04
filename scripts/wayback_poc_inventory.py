"""反向流程第 1 步：Esri Wayback 全 POC 影像盤點（只查 metadata，不抓影像）。

每個 Wayback 版本（2014 起）有一個 metadata 圖層服務，內含該版本影像的拍攝範圍多邊形（SRC_DATE2＝拍攝日、SRC_RES＝解析度、NICE_NAME＝供應來源）。
以 POC 範圍框做 envelope 查詢（每次約 1 秒），各版本合併去重（拍攝日＋範圍簽名），再：
  1. 每個拍攝日：涵蓋 POC 的面積與比例（0.001° ≈ 100 m 格網點陣化）。
  2. 每個 Slope Unit（質心）：有幾個不同拍攝日、涵蓋的年份、最早／最晚拍攝日、在研究期（2017–2026）內相隔 ≥365 天的前後期是否可成對。
輸出：data/biggis_interp/poc/wayback_footprints.json（簡化多邊形）、wayback_poc_coverage.json（統計）、wayback_su_depth.json（每個 SU）。
限制：metadata 的拍攝日是「該筆影像來源日期」，同一範圍可能是多景拼接（日期為其中一景）；範圍多邊形簡化 ≈50 m；
      有 metadata 涵蓋不等於該版本圖磚真的有影像，抽驗請另用 fetch_image。2017 以前（Wayback 2014–2016 版本）只作參考，研究期為 2017–2026。
用法：python scripts/wayback_poc_inventory.py [--refresh]
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from pyproj import Transformer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "webapp" / "change_detect_viewer"))
import wayback_assist as W  # noqa: E402

POC = REPO / "data" / "biggis_interp" / "poc"
BBOX = (120.55, 22.95, 120.95, 23.35)
STEP = 0.001
LAYERS = range(4, 10)           # metadata 圖層依解析度編號；z17–z19 附近的圖層（見 wayback_assist.acquisition_info）
STUDY = (date(2017, 1, 1), date(2026, 12, 31))
OFFSET = 0.0005                 # 多邊形簡化容差（度，≈50 m）


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def query(base, layer):
    env = ",".join(str(v) for v in BBOX)
    out, off = [], 0
    while True:
        q = (f"{base}/{layer}/query?f=json&geometry={env}&geometryType=esriGeometryEnvelope&inSR=4326&outSR=4326"
             f"&spatialRel=esriSpatialRelIntersects&outFields=SRC_DATE2,SRC_RES,NICE_NAME&returnGeometry=true"
             f"&maxAllowableOffset={OFFSET}&resultOffset={off}&resultRecordCount=1000")
        j = W._get_json(q, 90, retries=2)
        f = j.get("features") or []
        out += f
        if not j.get("exceededTransferLimit") or not f:
            return out
        off += len(f)


def one_release(r):
    feats = []
    for layer in LAYERS:
        try:
            feats += query(r["meta_url"], layer)
        except Exception as e:  # noqa: BLE001
            log("失敗", r["date"], layer, repr(e)[:70])
    return r, feats


def main():
    refresh = "--refresh" in sys.argv
    raw = POC / "wayback_footprints.json"
    if raw.exists() and not refresh:
        fps = json.load(open(raw, encoding="utf-8"))
        log("使用快取", len(fps), "個範圍")
    else:
        rels = [r for r in W.releases() if r["meta_url"]]
        log("Wayback 版本", len(rels), "個（含 2014 起）")
        seen, fps = set(), []
        with ThreadPoolExecutor(8) as ex:
            for n, (r, feats) in enumerate(ex.map(one_release, rels), 1):
                add = 0
                for f in feats:
                    a = f["attributes"]
                    ms = a.get("SRC_DATE2")
                    rings = (f.get("geometry") or {}).get("rings")
                    if not ms or not rings:
                        continue
                    d = datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                    xs = [p[0] for ring in rings for p in ring]
                    ys = [p[1] for ring in rings for p in ring]
                    sig = (d, round(min(xs), 3), round(min(ys), 3), round(max(xs), 3), round(max(ys), 3), len(rings))
                    if sig in seen:
                        continue
                    seen.add(sig)
                    fps.append({"date": d, "res_m": a.get("SRC_RES"), "name": a.get("NICE_NAME"), "release": r["num"], "release_date": r["date"],
                                "rings": [[[round(p[0], 5), round(p[1], 5)] for p in ring] for ring in rings]})
                    add += 1
                if n % 20 == 0 or n == len(rels):
                    log(f"{n}/{len(rels)} 版本，累計 {len(fps)} 個不同範圍（本版新增 {add}）")
        raw.write_text(json.dumps(fps, ensure_ascii=False), encoding="utf-8")
    # 點陣化
    nx, ny = int(round((BBOX[2] - BBOX[0]) / STEP)), int(round((BBOX[3] - BBOX[1]) / STEP))

    def rasterize(f):
        m = np.zeros((ny, nx), np.uint8)
        for ring in f["rings"]:
            pts = np.array([[(p[0] - BBOX[0]) / STEP, (BBOX[3] - p[1]) / STEP] for p in ring])
            cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
        return m.astype(bool)
    masks = {}
    for f in fps:
        k = f["date"]
        m = rasterize(f)
        masks[k] = masks[k] | m if k in masks else m
    cell_km2 = (STEP * 111.0) * (STEP * 102.2)
    days = sorted(masks)
    per_date = [{"date": k, "km2": round(float(masks[k].sum() * cell_km2), 1), "frac": round(float(masks[k].mean()), 3),
                 "res_m": sorted({f["res_m"] for f in fps if f["date"] == k and f["res_m"]}), "sources": sorted({f["name"] for f in fps if f["date"] == k and f["name"]})} for k in days]
    study = [k for k in days if STUDY[0] <= date.fromisoformat(k) <= STUDY[1]]
    yr = {}
    for k in study:
        yr.setdefault(k[:4], np.zeros((ny, nx), bool))
        yr[k[:4]] |= masks[k]
    # SU 深度
    T = json.load(open(POC / "su_table.json", encoding="utf-8"))
    to_ll = Transformer.from_crs(3826, 4326, always_xy=True)
    su = []
    for t in T:
        lon, lat = to_ll.transform(t["cx"], t["cy"])
        ix, iy = int((lon - BBOX[0]) / STEP), int((BBOX[3] - lat) / STEP)
        ds = [k for k in study if 0 <= ix < nx and 0 <= iy < ny and masks[k][iy, ix]]
        pair = bool(ds) and (date.fromisoformat(ds[-1]) - date.fromisoformat(ds[0])).days >= 365
        su.append({"i": t["i"], "su_id": t["su_id"], "category": t["category"], "n_dates": len(ds), "years": sorted({k[:4] for k in ds}), "first": ds[0] if ds else None, "last": ds[-1] if ds else None, "pair_ge365d": pair})
    n = len(su)
    summ = {"footprints": len(fps), "distinct_dates_all": len(days), "distinct_dates_2017_2026": len(study),
            "poc_cells": int(nx * ny), "poc_area_km2": round(nx * ny * cell_km2, 0),
            "year_union_coverage_frac_2017_2026": {y: round(float(m.mean()), 3) for y, m in sorted(yr.items())},
            "dates_per_year_2017_2026": {y: sum(1 for k in study if k[:4] == y) for y in sorted(yr)},
            "su_total": n, "su_ge1_date": sum(1 for s in su if s["n_dates"] >= 1), "su_ge2_dates": sum(1 for s in su if s["n_dates"] >= 2),
            "su_ge4_dates": sum(1 for s in su if s["n_dates"] >= 4), "su_pair_ge365d": sum(1 for s in su if s["pair_ge365d"]),
            "res_m_by_date_median": float(np.median([f["res_m"] for f in fps if f["res_m"]])) if any(f["res_m"] for f in fps) else None,
            "by_source": {s: sum(1 for f in fps if f["name"] == s) for s in sorted({f["name"] for f in fps if f["name"]})},
            "per_date": per_date,
            "note": "metadata 範圍涵蓋不等於圖磚有影像；日期為影像來源日期（可能多景拼接）；多邊形簡化約 50 m"}
    (POC / "wayback_poc_coverage.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    (POC / "wayback_su_depth.json").write_text(json.dumps(su, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if k not in ("per_date",)}, ensure_ascii=False, indent=1))
    log("2017–2026 各拍攝日涵蓋（≥2% POC）：")
    for p in per_date:
        if STUDY[0] <= date.fromisoformat(p["date"]) <= STUDY[1] and p["frac"] >= 0.02:
            log(" ", p["date"], f"{p['frac']:.0%}", p["res_m"], ",".join(p["sources"]))


if __name__ == "__main__":
    main()
