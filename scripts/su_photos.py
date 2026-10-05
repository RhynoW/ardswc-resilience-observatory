"""把水保署歷史災害照片（photo.ardswc.gov.tw 平台紀錄）掛入 Slope Unit（依陳振宇委員 2026-10-04 建議：照片是「補充證據」，與衛星事件目錄分兩類，不相加）。
資料：data/ardswc_hotspots/events_trimmed.json（id、lat、lon、photo_type、year；PhotoType 0＝災害事件、8＝媒體報導、6＝重要地景、10＝出版品）。
掛接：照片座標 → 20 m SU 標籤（原始幾何，su_labels.npz）取得所屬 SU；同時標記是否落在有效坡面（≥15°，su_labels_eff.npz）與到最近有效坡面的距離（m）。
      河谷道路旁的照片常落在 <15° 的緩坡，故以「所屬 SU＋是否有效坡面＋距離」同時保留，不丟棄。
輸出（poc/）：su_photos.json：逐照片（id、type、year、su_id、in_eff、dist_eff_m）；su_evidence.json：逐 SU 的兩類證據——
  satellite：事件目錄／年度圖層類別、四年裸露面積、新增裸露、主因事件（來自 su_table_eff.json）
  field_photos：現地照片筆數（災害事件、媒體報導分開；重要地景另列、不算災害）、年份清單（不與衛星事件次數相加）
限制：趨勢檔只有座標、類型、年份——沒有描述、拍攝日與尺度，無法判斷是不是「小型崩塌／落石」；照片是拍了多少張，不是災害次數；
      道路可達處拍得多（可達性偏誤）；POC 範圍外不處理。
用法：python scripts/su_photos.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pyproj import Transformer
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parent.parent
POC = REPO / "data" / "biggis_interp" / "poc"
BBOX = (120.55, 22.95, 120.95, 23.35)
TYPES = {"0": "災害事件", "8": "媒體報導", "6": "重要地景", "10": "出版品照片"}


def main():
    ph = json.load(open(REPO / "data" / "ardswc_hotspots" / "events_trimmed.json", encoding="utf-8"))
    ph = [p for p in ph if BBOX[0] <= p["lon"] <= BBOX[2] and BBOX[1] <= p["lat"] <= BBOX[3]]
    print("POC 範圍內照片", len(ph), dict(Counter(TYPES[p["photo_type"]] for p in ph)))
    L = np.load(POC / "su_labels.npz")
    Le = np.load(POC / "su_labels_eff.npz")
    key20, ke, x0, y0 = L["key20"], Le["key20"], float(L["x0"]), float(L["y0"])
    meta = json.load(open(POC / "su_meta_eff.json", encoding="utf-8"))
    uid = {u["key"]: u for u in meta["units"]}
    dist = ndi.distance_transform_edt(ke < 0) * 20.0          # 到最近有效坡面格的距離（m）
    to = Transformer.from_crs(4326, 3826, always_xy=True)
    X, Y = to.transform([p["lon"] for p in ph], [p["lat"] for p in ph])
    rows = []
    for p, x, y in zip(ph, X, Y):
        c, r = int((x - x0) // 20), int((y0 - y) // 20)
        if not (0 <= r < key20.shape[0] and 0 <= c < key20.shape[1]) or key20[r, c] < 0:
            rows.append({"id": p["id"], "type": TYPES[p["photo_type"]], "year": p["year"], "su_id": None, "in_eff": False, "dist_eff_m": None})
            continue
        k = int(key20[r, c])
        rows.append({"id": p["id"], "type": TYPES[p["photo_type"]], "year": p["year"], "su_id": uid[k]["su_id"], "in_eff": bool(ke[r, c] >= 0), "dist_eff_m": round(float(dist[r, c]), 0)})
    (POC / "su_photos.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    tab = {t["su_id"]: t for t in json.load(open(POC / "su_table_eff.json", encoding="utf-8"))}
    by = defaultdict(lambda: {"災害事件": [], "媒體報導": [], "重要地景": [], "出版品照片": []})
    for r in rows:
        if r["su_id"]:
            by[r["su_id"]][r["type"]].append(r["year"])
    ev = []
    for sid, t in tab.items():
        ph_ = by.get(sid, {})
        ev.append({"su_id": sid, "su_uid": f"{sid}@{meta['params']['su_version']}", "satellite": {"category": t["category"], "bare_ha": t["bare_ha"], "new_bare_ha_total": t["new_bare_ha_total"], "dominant_event": t["dominant_event"], "explained_by_events": t["explained_by_events"]},
                   "field_photos": {k: {"n": len(v), "years": sorted(set(v))} for k, v in ph_.items() if v}})
    (POC / "su_evidence.json").write_text(json.dumps(ev, ensure_ascii=False), encoding="utf-8")
    # 摘要
    inside = [r for r in rows if r["su_id"]]
    eff = [r for r in inside if r["in_eff"]]
    near = [r for r in inside if (not r["in_eff"]) and r["dist_eff_m"] is not None and r["dist_eff_m"] <= 60]
    print(f"掛到 SU {len(inside)}/{len(rows)}；落在有效坡面 {len(eff)}；有效坡面外但 ≤60 m {len(near)}；其餘（緩坡／谷底）{len(inside) - len(eff) - len(near)}")
    dis = lambda e: e["field_photos"].get("災害事件", {}).get("n", 0) + e["field_photos"].get("媒體報導", {}).get("n", 0)
    n_dis = sum(1 for e in ev if dis(e) > 0)
    n_sat = sum(1 for e in ev if e["satellite"]["category"] != "無崩塌")
    both = sum(1 for e in ev if dis(e) > 0 and e["satellite"]["category"] != "無崩塌")
    photo_only = sum(1 for e in ev if dis(e) > 0 and e["satellite"]["category"] == "無崩塌")
    sat_only = n_sat - both
    print(f"SU {len(ev)}：有災害／媒體照片 {n_dis}；衛星有裸露（≥0.5 ha 任一年）{n_sat}；兩者皆有 {both}；只有照片（衛星無）{photo_only}；只有衛星（無照片）{sat_only}")
    cat = Counter(e["satellite"]["category"] for e in ev if dis(e) > 0)
    print("有照片的 SU 之衛星類別：", dict(cat))


if __name__ == "__main__":
    main()
