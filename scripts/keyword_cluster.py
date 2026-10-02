# -*- coding: utf-8 -*-
"""
與時間相關的關鍵字集群分析 → data/keyword_cluster.json（並存一份到 webapp/.../static/ 供 spatial3d.html 讀取）

資料：build_hotspots.py 抓下的原始快取（--raw-dir，type_0.json 災害事件、type_8.json 媒體報導，欄位為平台 API 原樣）。
  關鍵字以「描述＋備註＋災害名稱」子字串比對（與 build_analytics.py 相同做法）；年份用 DisasterYear；空間用 data/analytics.json 的 5 km 網格順序。
分析
  1. 逐年占比：關鍵字在當年所有災害紀錄中的比例（扣掉「某年拍特別多」的總量效應）。
  2. 關鍵字相似度：時間相似＝逐年占比的 Pearson 相關；空間相似＝各格 log1p(筆數) 的 Pearson 相關；
     合併相似度＝兩者平均 → 距離 1−相似度 → 平均連結階層分群（切成 6 群），輸出葉序供熱圖排序。
  3. 趨勢：逐年占比的 Kendall τ（Mann-Kendall）與 p 值；尖峰年份；上升／下降／無明顯趨勢（p<0.05）。
  4. 空間重心軌跡：每個關鍵字每年（≥10 筆）的筆數加權重心，看熱區是否隨時間移動。
  5. 稀疏的「關鍵字×格×年」計數，網頁端據此即時算任一（關鍵字、年度）組合的 Getis-Ord Gi*（附鄰格表）。
限制：子字串比對會有誤判（例如「橋」也出現在地名）；一筆紀錄可含多個關鍵字；占比的年份只含每年 ≥300 筆的年份；
  拍攝張數受可達性與調查人力影響，相似度＝「一起出現／同地出現」，不是因果。
用法：python scripts/keyword_cluster.py --raw-dir <快取目錄>
"""
import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "keyword_cluster.json"
STATIC = REPO / "webapp" / "change_detect_viewer" / "static" / "keyword_cluster.json"
KEYWORDS = ["崩塌", "坍方", "邊坡", "落石", "土石流", "潛勢溪流", "堰塞湖", "淹水", "沖毀", "道路中斷", "護岸", "橋", "擋土牆", "地震", "颱風", "豪雨"]
BAND_KM = 8.0
NCLUST = 6


def km_matrix(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    a = np.sin((la[:, None] - la[None, :]) / 2) ** 2 + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin((lo[:, None] - lo[None, :]) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--min-centroid", type=int, default=10)
    a = ap.parse_args()
    raw = Path(a.raw_dir)
    A = json.loads((REPO / "data" / "analytics.json").read_text(encoding="utf-8"))
    S = json.loads((REPO / "data" / "spatial_analysis.json").read_text(encoding="utf-8"))
    years = S["temporal"]["years"]; T = len(years); yi = {y: k for k, y in enumerate(years)}
    g = np.array(A["grid"], float); lat, lon = g[:, 0], g[:, 1]; n = len(g)
    ckey = {(round(float(x), 2), round(float(y), 2)): i for i, (x, y) in enumerate(zip(lat, lon))}
    K = len(KEYWORDS)
    cnt = np.zeros((K, n, T + 1))          # 最後一欄＝年份不在 years（筆數太少的年份）內
    ytot = np.zeros(T)
    rows = []
    for t in ("0", "8"):
        for r in json.loads((raw / f"type_{t}.json").read_text(encoding="utf-8")):
            if not (r.get("Lat") and r.get("Lng")):
                continue
            i = ckey.get((round(round(float(r["Lat"]) / 0.05) * 0.05, 2), round(round(float(r["Lng"]) / 0.05) * 0.05, 2)))
            if i is None:
                continue
            rows.append((i, int(r["DisasterYear"]) if r.get("DisasterYear") else 0, float(r["Lat"]), float(r["Lng"]),
                         (r.get("Description") or "") + " " + (r.get("Note") or "") + " " + (r.get("DisasterName") or "")))
    cent = [[[0.0, 0.0, 0] for _ in range(T)] for _ in range(K)]
    for i, y, la_, lo_, txt in rows:
        k_t = yi.get(y, T)
        if k_t < T:
            ytot[k_t] += 1
        for k, kw in enumerate(KEYWORDS):
            if kw in txt:
                cnt[k, i, k_t] += 1
                if k_t < T:
                    c = cent[k][k_t]; c[0] += la_; c[1] += lo_; c[2] += 1
    tot = cnt.sum(2)                                            # K×n 全期間（含不在 years 內的年份）
    yearly = cnt[:, :, :T].sum(1)                               # K×T
    share = yearly / np.maximum(ytot, 1)[None, :]

    # 相似度與階層分群
    ct = np.corrcoef(share); cs = np.corrcoef(np.log1p(tot))
    sim = 0.5 * (ct + cs)
    from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
    from scipy.spatial.distance import squareform
    dist = np.clip(1 - sim, 0, 2); np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), "average")
    order = [int(v) for v in leaves_list(Z)]
    cl = fcluster(Z, NCLUST, "maxclust")
    # 趨勢
    from scipy.stats import kendalltau
    trend = []
    for k in range(K):
        tau, p = kendalltau(years, share[k])
        pk = int(np.argmax(share[k]))
        d = "up" if (p < 0.05 and tau > 0) else "down" if (p < 0.05 and tau < 0) else "flat"
        trend.append({"tau": round(float(tau), 2), "p": round(float(p), 3), "dir": d, "peak_year": years[pk], "peak_share": round(float(share[k, pk]), 3)})
    # 稀疏計數
    cy = []
    for k in range(K):
        nz = np.argwhere(cnt[k] > 0)
        cy.append([int(v) for (i, t) in nz for v in (i, t, int(cnt[k, i, t]))])
    # 鄰格（供網頁即時算 Gi*）
    D = km_matrix(lat, lon)
    nbr = [[int(j) for j in np.where((D[i] <= BAND_KM) & (D[i] > 0))[0]] for i in range(n)]
    cents = [[[round(c[0] / c[2], 3), round(c[1] / c[2], 3), int(c[2])] if c[2] >= a.min_centroid else None for c in row] for row in cent]

    out = {"generated": A.get("generated"), "keywords": KEYWORDS, "n_records": len(rows), "years": years, "year_total": [int(v) for v in ytot],
           "total": [int(tot[k].sum()) for k in range(K)], "count": [[int(v) for v in yearly[k]] for k in range(K)],
           "share": [[round(float(v), 4) for v in share[k]] for k in range(K)],
           "corr_time": np.round(ct, 2).tolist(), "corr_space": np.round(cs, 2).tolist(), "sim": np.round(sim, 2).tolist(),
           "order": order, "cluster": [int(v) for v in cl], "trend": trend, "centroids": cents, "cy": cy, "nbr": nbr,
           "cy_note": "cy[k]＝扁平陣列 [格索引, 年份索引, 筆數, …]；年份索引 = len(years) 代表不在 years 內的年份",
           "note": "關鍵字為子字串比對，一筆紀錄可含多個關鍵字；占比＝關鍵字筆數／當年災害紀錄總數；相似度＝時間與空間相關的平均，不是因果。"}
    txt = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
    OUT.write_text(txt, encoding="utf-8"); STATIC.write_text(txt, encoding="utf-8")
    print(f"{len(rows)} 筆；{K} 個關鍵字；{T} 個年份 → {len(txt) // 1024} KB")
    for c in range(1, NCLUST + 1):
        print(f"  集群 {c}：", "、".join(KEYWORDS[k] for k in order if cl[k] == c))
    for k in range(K):
        t = trend[k]
        print(f"  {KEYWORDS[k]:<5} n={out['total'][k]:>5}  尖峰 {t['peak_year']}（占比 {t['peak_share']:.0%}）  趨勢 {t['dir']}（τ={t['tau']}, p={t['p']}）")


if __name__ == "__main__":
    main()
