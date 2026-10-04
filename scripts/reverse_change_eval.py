"""反向流程「變化」評估：Wayback 2022-08-26 → 2023-11-02 的影像新增裸露 vs 窗口內事件型崩塌目錄。
新增裸露＝後期偵測有、前期偵測（膨脹 1 格）無，移除 <0.1 ha；偵測用基準 A（T=0.08）與 A＋雲塊移除（AC，見 reverse_cloud_rule.py）。
評估範圍＝兩期影像皆有效、坡度 ≥15°（region.npy，409 km² 有效／372 km²）。
指標同 s2_detect.py：事件內／外新增機率與倍數、新增面積落在事件內的比例、事件面積被偵測為新增的比例、
與官方圖層新增（2023 有、2022 無）的重疊。另以「事件日期錯置對照」估計倍數有多少來自空間相關：把事件多邊形平移 3 km 重算倍數。
限制：兩期季節不同（夏 vs 秋）；影像期與圖層期不同；單一影像對；事件判釋與年度圖層同為水保署衛星判釋。
輸出：poc/reverse/eval_change.json。用法：python scripts/reverse_change_eval.py
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import reverse_detect as R  # noqa: E402
from landslide_incremental import grid, mask  # noqa: E402
from landslide_su_events import load_events  # noqa: E402

WIN = (pd.Timestamp("2022-08-26"), pd.Timestamp("2023-11-02"))


def main():
    tf, shape = grid()
    region = np.load(R.OUT / "region.npy")
    Z = np.load(R.POC / "incremental_masks.npz")
    ev = {n: v for n, v in load_events().items() if WIN[0] < v[0] <= WIN[1]}
    names = [f"{n}（{v[0].date()}）" for n, v in ev.items()]

    def rast(shift_m=0.0):
        U = np.zeros(shape, bool)
        for n, (dt, g) in ev.items():
            gg = g.translate(xoff=shift_m) if shift_m else g
            U |= ndi.binary_dilation(mask(gpd.GeoDataFrame(geometry=gg.values, crs=3826), tf, shape), iterations=3)
        return U
    U0 = rast()
    shifts = {s: rast(s) for s in (3000.0, -3000.0, 6000.0)}
    off_new = Z["2023"] & ~Z["2022"] & region
    out = {"window": [str(WIN[0].date()), str(WIN[1].date())], "events_in_window": names, "events_area_ha_in_region": round(float((U0 & region).sum() * 0.01), 1),
           "official_new_ha": round(float(off_new.sum() * 0.01), 1)}
    for tag, pre in (("A", "bare_A_"), ("AC", "bare_AC_")):
        d22 = np.load(R.OUT / f"{pre}2022-08-26.npy")
        d23 = np.load(R.OUT / f"{pre}2023-11-02.npy")
        new = R.post(d23 & ~ndi.binary_dilation(d22, iterations=1)) & region
        avail = region & ~d22

        def ratio(U):
            pin = float(new[U & avail].sum() / max((U & avail).sum(), 1))
            pout = float(new[~U & avail].sum() / max((~U & avail).sum(), 1))
            return round(pin, 4), round(pout, 4), round(pin / pout, 2) if pout else None
        pin, pout, rt = ratio(U0)
        out[tag] = {"new_ha": round(float(new.sum() * 0.01), 1), "new_ha_in_events": round(float((new & U0).sum() * 0.01), 1),
                    "share_new_in_events": round(float((new & U0).sum() / max(new.sum(), 1)), 3), "P(new|in)": pin, "P(new|out)": pout, "ratio": rt,
                    "ratio_when_events_shifted": {str(int(s)) + "m": ratio(U)[2] for s, U in shifts.items()},
                    "events_detected_as_new_frac": round(float((new & U0).sum() / max((U0 & region).sum(), 1)), 3),
                    "official_new_hit_by_new_frac": round(float((off_new & ndi.binary_dilation(new, iterations=2)).sum() / max(off_new.sum(), 1)), 3),
                    "new_overlapping_official_new_frac": round(float((new & ndi.binary_dilation(off_new, iterations=2)).sum() / max(new.sum(), 1)), 3)}
        print(tag, out[tag], flush=True)
    print(out["events_in_window"], out["events_area_ha_in_region"], out["official_new_ha"])
    (R.OUT / "eval_change.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
