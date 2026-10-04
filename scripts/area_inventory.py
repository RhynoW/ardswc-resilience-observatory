"""新 POC 區域的資料盤點（只查 metadata，不抓影像）：Wayback 拍攝範圍、BigGIS UAV 正射圖磚、事件型目錄、Sentinel-2 場景。
用法：python scripts/area_inventory.py hualien_barrier_lake 121.2956 23.6994 0.15
輸出：data/biggis_interp/areas/<name>/inventory.json（已被 .gitignore 的 data/biggis_interp/ 涵蓋？請確認）與終端摘要。
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "webapp" / "change_detect_viewer"))
sys.path.insert(0, str(REPO / "scripts"))
import wayback_assist as W  # noqa: E402
import wayback_poc_inventory as WI  # noqa: E402
import sentinel_assist as S2  # noqa: E402
import su_uav_inventory as UI  # noqa: E402

STEP = 0.001


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    name, lon, lat, half = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    bbox = (round(lon - half, 4), round(lat - half, 4), round(lon + half, 4), round(lat + half, 4))
    out_dir = REPO / "data" / "biggis_interp" / "areas" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    WI.BBOX = bbox
    log("範圍", bbox)
    # 1. Wayback
    rels = [r for r in W.releases() if r["meta_url"]]
    seen, fps = set(), []
    with ThreadPoolExecutor(8) as ex:
        for n, (r, feats) in enumerate(ex.map(WI.one_release, rels), 1):
            for f in feats:
                a = f["attributes"]
                ms, rings = a.get("SRC_DATE2"), (f.get("geometry") or {}).get("rings")
                if not ms or not rings:
                    continue
                d = datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                xs, ys = [p[0] for ring in rings for p in ring], [p[1] for ring in rings for p in ring]
                sig = (d, round(min(xs), 3), round(min(ys), 3), round(max(xs), 3), round(max(ys), 3), len(rings))
                if sig in seen:
                    continue
                seen.add(sig)
                fps.append({"date": d, "res_m": a.get("SRC_RES"), "name": a.get("NICE_NAME"), "release": r["num"], "rings": [[[round(p[0], 5), round(p[1], 5)] for p in ring] for ring in rings]})
    nx, ny = int(round((bbox[2] - bbox[0]) / STEP)), int(round((bbox[3] - bbox[1]) / STEP))
    masks = {}
    for f in fps:
        m = np.zeros((ny, nx), np.uint8)
        for ring in f["rings"]:
            pts = np.array([[(p[0] - bbox[0]) / STEP, (bbox[3] - p[1]) / STEP] for p in ring])
            cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
        masks[f["date"]] = masks.get(f["date"], np.zeros((ny, nx), bool)) | m.astype(bool)
    per_date = [{"date": k, "frac": round(float(m.mean()), 3), "res_m": sorted({f["res_m"] for f in fps if f["date"] == k and f["res_m"]})} for k, m in sorted(masks.items())]
    study = [p for p in per_date if p["date"] >= "2017-01-01"]
    yr = {}
    for p in study:
        yr.setdefault(p["date"][:4], np.zeros((ny, nx), bool))
        yr[p["date"][:4]] |= masks[p["date"]]
    wb = {"footprints": len(fps), "dates_2017plus_ge10pct": [p for p in study if p["frac"] >= 0.10], "year_union_coverage": {y: round(float(m.mean()), 3) for y, m in sorted(yr.items())}}
    # 2. UAV
    rows, _ = UI.load_list()
    K = UI.K
    hit = [r for r in rows if r[K[0]] <= bbox[2] and r[K[1]] >= bbox[0] and r[K[2]] <= bbox[3] and r[K[3]] >= bbox[1]]
    uav = [{"date": r["Date"], "title": r.get("Title"), "id": r.get("Id"), "bbox": [r[k] for k in K]} for r in sorted(hit, key=lambda r: r["Date"])]
    # 3. 事件型目錄
    pj = json.load(open(REPO / "data" / "biggis_interp" / "polygons.json", encoding="utf-8"))
    ev_hit = {}
    for p in pj["polys"]:
        xy = p[2]
        xs, ys = xy[0::2], xy[1::2]
        if min(xs) <= bbox[2] and max(xs) >= bbox[0] and min(ys) <= bbox[3] and max(ys) >= bbox[1]:
            e = pj["events"][p[0]]
            k = e["event"]
            ev_hit.setdefault(k, {"date": e["date"], "n_polys": 0})
            ev_hit[k]["n_polys"] += 1
    # 4. Sentinel-2
    s2 = {}
    for tag, a, b in (("2024", date(2024, 1, 1), date(2024, 12, 31)), ("2025-01_08", date(2025, 1, 1), date(2025, 8, 31)), ("2025-09_2026", date(2025, 9, 1), date(2026, 6, 30))):
        try:
            s2[tag] = S2.search_scenes(lat, lon, a, b, max_cloud=30, half_m=half * 111000)
        except Exception as e:  # noqa: BLE001
            s2[tag] = repr(e)[:120]
    inv = {"name": name, "center": [lon, lat], "bbox": bbox, "wayback": wb, "uav": uav, "events": ev_hit, "sentinel2": s2}
    (out_dir / "inventory.json").write_text(json.dumps(inv, ensure_ascii=False, indent=1), encoding="utf-8")
    log("Wayback", wb["footprints"], "個範圍；2017+ 年度聯集", wb["year_union_coverage"])
    log("Wayback ≥10% 的拍攝日", [(p["date"], p["frac"], p["res_m"]) for p in wb["dates_2017plus_ge10pct"]])
    log("UAV 影像", len(uav), [(u["date"], u["title"]) for u in uav][:30])
    log("事件", len(ev_hit), sorted((v["date"], k.split("_", 1)[-1], v["n_polys"]) for k, v in ev_hit.items())[-15:])
    log("S2", {k: (len(v) if isinstance(v, list) else v) for k, v in s2.items()})


if __name__ == "__main__":
    main()
