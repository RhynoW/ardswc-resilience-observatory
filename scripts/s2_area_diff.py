"""花蓮 Sentinel-2：以「ExG 差異」取代「兩期各自門檻」的新增裸露定義，並做輻射一致性處理（不用事件標籤調參）。
問題：固定 ExG 門檻在不同日期的大氣／光照下不穩，造成「新增」面積在 1,900–18,000 ha 之間隨事後日期劇烈變動（見 s2_area.py 第三輪）。
做法：dExG = ExG_前 − ExG_後；T_d = 兩期評估範圍內 dExG 的中位數 + K×MAD（穩健，抵消兩期整體輻射偏移）；
      新增裸露 = (dExG > T_d) 且 後期 ExG<0.08 且 V≥70，3×3 開運算、移除 <0.1 ha；範圍＝兩期有效且坡度 ≥15°，並排除前期已裸露（膨脹 1 格）。
K 取 4（固定，不調參）。其餘評估同 s2_area.py：事件內／外新增機率與倍數、事件平移對照、事件被偵測比例。
輸出：areas/hualien_barrier_lake/s2/eval_diff.json。用法：python scripts/s2_area_diff.py
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
import s2_area as A  # noqa: E402
import reverse_detect as R  # noqa: E402
from landslide_incremental import mask  # noqa: E402

K = 4.0


def main():
    tf, (H, W) = A.grid()
    shape = (H, W)
    F = {d: A.feats(d) for d in A.DATES}
    steep = A.slope(tf, H, W) >= R.SLOPE_MIN
    ev = A.events()
    bb = gpd.GeoSeries([A.box(*A.BBOX)], crs=4326).to_crs(3826).iloc[0]
    ev = {k: g for k, g in ev.items() if g.intersects(bb).any()}
    sel = [g for k, g in ev.items() if "2025-05-14" < k[1] <= "2025-10-16"]
    pol = gpd.GeoSeries(pd.concat(sel).values, crs=3826)
    U0 = ndi.binary_dilation(mask(gpd.GeoDataFrame(geometry=pol.values, crs=3826), tf, shape), iterations=3)
    shifted = {s: ndi.binary_dilation(mask(gpd.GeoDataFrame(geometry=pol.translate(xoff=s).values, crs=3826), tf, shape), iterations=3) for s in (3000.0, -3000.0, 6000.0)}
    out = {"K": K, "pairs": {}}
    for a in A.PRE:
        for b in A.POST:
            ea, eb = F[a][0], F[b][0]
            va = F[a][2] & F[b][2]
            region = va & steep
            d = ea - eb
            med = float(np.median(d[region]))
            mad = float(np.median(np.abs(d[region] - med))) * 1.4826
            td = max(med + K * mad, 0.05)
            bare_a = R.post((ea < 0.08) & (F[a][1] >= 70) & F[a][2])
            new = R.post((d > td) & (eb < 0.08) & (F[b][1] >= 70) & va) & region & ~ndi.binary_dilation(bare_a, iterations=1)
            avail = region & ~ndi.binary_dilation(bare_a, iterations=1)

            def ratio(U):
                pin = float(new[U & avail].sum() / max((U & avail).sum(), 1))
                pout = float(new[~U & avail].sum() / max((~U & avail).sum(), 1))
                return round(pin, 4), round(pout, 4), round(pin / pout, 2) if pout else None
            pin, pout, rt = ratio(U0)
            out["pairs"][f"{a}->{b}"] = {"dExG_median": round(med, 3), "dExG_mad": round(mad, 3), "T_d": round(td, 3), "new_ha": round(float(new.sum() * 0.01), 1),
                                         "new_ha_in_event": round(float((new & U0).sum() * 0.01), 1), "share_new_in_event": round(float((new & U0).sum() / max(new.sum(), 1)), 3),
                                         "P(new|in)": pin, "P(new|out)": pout, "ratio": rt, "ratio_shifted": {str(int(s)) + "m": ratio(U)[2] for s, U in shifted.items()},
                                         "event_detected_as_new_frac": round(float((new & U0).sum() / max((U0 & region).sum(), 1)), 3)}
            np.save(A.OUT / f"newdiff_{a}_{b}.npy", new)
            r = out["pairs"][f"{a}->{b}"]
            print(a, b, "Td", r["T_d"], "new", r["new_ha"], "in", r["new_ha_in_event"], "share", r["share_new_in_event"], "in/out", r["P(new|in)"], r["P(new|out)"], "ratio", r["ratio"], "shift", r["ratio_shifted"], "ev_det", r["event_detected_as_new_frac"], flush=True)
    (A.OUT / "eval_diff.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
