"""第三個 POC 候選區掃描：事件目錄面積＋事件前後 Sentinel-2 實測雲量（z11 縮圖，與 s2_area_select.py 同一算法）。
對每個候選框列出：事件多邊形數與面積（框內）、事件前 75 天／後 75 天的 S2 日期中實測亮雲比例最低者。
用法：python scripts/poc_candidate_scan.py
輸出：data/biggis_interp/areas/candidates_scan.json
"""
import json
import math
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, box

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "webapp" / "change_detect_viewer"))
import sentinel_assist as S  # noqa: E402

MATRIX = 11
WORLD = 2 * math.pi * S.R_MERC
P = WORLD / (S.TILE_PX * 2 ** (MATRIX - 1))
CANDS = {
    "A_taroko_quake": {"bbox": (121.45, 24.05, 121.75, 24.35), "event": "2024-04-03"},
    "B_central_khanun": {"bbox": (120.98, 23.85, 121.28, 24.15), "event": "2023-08-03"},
    "C_chiayi_gaemi": {"bbox": (120.55, 23.35, 120.85, 23.65), "event": "2024-07-23"},
    "D_pingtung_krathon": {"bbox": (120.73, 22.55, 121.03, 22.85), "event": "2024-09-30"},
}


def merc(lon, lat):
    return S.R_MERC * math.radians(lon), S.R_MERC * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def cloud_scan(bbox, dates, scenes):
    x0, y0 = merc(bbox[0], bbox[3])
    x1, y1 = merc(bbox[2], bbox[1])
    px0, px1 = (x0 + WORLD / 2) / P, (x1 + WORLD / 2) / P
    py0, py1 = (WORLD / 2 - y0) / P, (WORLD / 2 - y1) / P
    c0, c1, r0, r1 = int(px0) // 512, int(px1) // 512, int(py0) // 512, int(py1) // 512
    tiles = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    iid = S.instance_id()

    def one(ymd):
        try:
            mos = np.zeros(((r1 - r0 + 1) * 512, (c1 - c0 + 1) * 512, 3), np.uint8)
            for r, c in tiles:
                mos[(r - r0) * 512:(r - r0 + 1) * 512, (c - c0) * 512:(c - c0 + 1) * 512] = S._fetch_tile(iid, ymd, "TRUE_COLOR", r, c, matrix=MATRIX)
            crop = mos[int(py0) - r0 * 512:int(py1) - r0 * 512 + 1, int(px0) - c0 * 512:int(px1) - c0 * 512 + 1]
            f = crop.astype(np.float32)
            v = f.max(axis=2)
            sat = (v - f.min(axis=2)) / (v + 1e-6)
            m1 = cv2.blur(v, (5, 5))
            sd = np.sqrt(np.maximum(cv2.blur(v * v, (5, 5)) - m1 * m1, 0))
            return {"date": ymd, "scene_cloud": scenes[ymd], "nodata": round(float((v < 8).mean()), 3), "cloud": round(float(((v >= 150) & (sat < 0.25) & (sd < 12) & (v >= 8)).mean()), 3)}
        except Exception as e:  # noqa: BLE001
            return {"date": ymd, "error": repr(e)[:60]}
    with ThreadPoolExecutor(4) as ex:
        return list(ex.map(one, dates))


def main():
    pj = json.load(open(REPO / "data" / "biggis_interp" / "polygons.json", encoding="utf-8"))
    out = {}
    for name, c in CANDS.items():
        bb = c["bbox"]
        ev = date.fromisoformat(c["event"])
        b = box(*bb)
        by = {}
        for p in pj["polys"]:
            e = pj["events"][p[0]]
            if not ("2023-06-01" <= e["date"] <= "2026-12-31"):
                continue
            xy = p[2]
            poly = Polygon(list(zip(xy[0::2], xy[1::2]))).buffer(0)
            if poly.is_empty or not poly.intersects(b):
                continue
            by.setdefault((e["date"], e["event"].split("_", 1)[-1][:16]), []).append(poly)
        evs = []
        for (d, n), polys in by.items():
            a = float(gpd.GeoSeries(polys, crs=4326).to_crs(3826).area.sum() / 1e4)
            evs.append({"date": d, "event": n, "n": len(polys), "area_ha": round(a, 1)})
        evs.sort(key=lambda r: -r["area_ha"])
        lon, lat = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
        sc = {}
        for s in S.search_scenes(lat, lon, ev - timedelta(days=75), ev + timedelta(days=75), max_cloud=80.0, half_m=15000):
            sc[s["date"]] = s["cloud"]
        evd = ev.strftime("%Y%m%d")
        pre = sorted(d for d in sc if d < evd)
        post = sorted(d for d in sc if d > evd)
        t0 = time.time()
        rp, rq = cloud_scan(bb, pre, sc), cloud_scan(bb, post, sc)
        ok = lambda rs: sorted([r for r in rs if "cloud" in r], key=lambda r: r["cloud"])[:3]
        out[name] = {"bbox": bb, "event_date": c["event"], "events_in_bbox": evs[:6], "n_pre_dates": len(pre), "n_post_dates": len(post), "best_pre": ok(rp), "best_post": ok(rq)}
        print(name, json.dumps({k: v for k, v in out[name].items() if k != "bbox"}, ensure_ascii=False), round(time.time() - t0), "s", flush=True)
    (REPO / "data" / "biggis_interp" / "areas" / "candidates_scan.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
