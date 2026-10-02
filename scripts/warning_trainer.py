# -*- coding: utf-8 -*-
"""
預警「輔助訓練」工具原型：把 BigGIS 競賽作品的「疊圖→目視判『高度吻合』」改成可量化、可交叉驗證、可換圖層的流程。

動機（見 ARCHITECTURE §13.2）：18 件得獎作品裡與災害相關的 7 件，共同做法是把災點疊在防災圖層上
（地文脆弱度、山崩地滑地質敏感區、土石流潛勢溪流、不安定土砂、雨量、地震），再用「高度吻合／不吻合」下結論；
露營區那件還手訂 6 因子加權（30/20/15/15/10/10）。缺的是：命中率、對照組、權重由誰決定、是否在沒看過的地區仍成立。
本工具補這四件事：

  1. 正例＝本系統的複發災害熱點；負例＝山區內與任何熱點距離 ≥ --min-dist-km 的隨機點（--neg-ratio 倍）。
  2. 特徵表（每個圖層一欄）。目前只放 DTM 衍生（高程、平均坡度、500 m 起伏、粗糙度，離線可重現）；
     BigGIS／官方圖層（地文脆弱度、土石流潛勢溪流…）下載後以 --extra-csv 併入，欄位名即圖層名，不改程式。
  3. 逐圖層單因子：AUC、命中率（正例落在「高風險」前 20% 的比例）與 lift——取代「高度吻合」四個字。
  4. 多因子透明模型（標準化 logistic regression）：係數即權重，可與專家手訂權重並列比較；
     以「10 km 空間區塊」做交叉驗證（GroupKFold），避免相鄰點洩漏導致 AUC 虛高。

限制（務必一併報告）：
  * 正例是「有通報的災害」，帶有曝險偏差（有人、有路的地方才有通報）；負例是隨機山區點，可能含未通報的真災害。
    所以 AUC 衡量的是「圖層能否區分通報熱點與一般山區」，不是真實災害機率。
  * 只用 DTM 特徵時模型很可能主要在學「坡度／靠近道路的山麓」；真正有判別力的圖層要靠 --extra-csv 加進來再比。
  * 不含時間維度（雨量觸發）——預警需要「靜態潛勢 × 動態雨量」，這裡只做靜態潛勢那一半。

用法：python scripts/warning_trainer.py [--hotspots data/ardswc_hotspots/top100_consolidated.json]
      [--extra-csv layers.csv  # 欄位：id,<layer1>,<layer2>…；id = 熱點 rank（正例）或 neg_<n>（負例，見輸出的 samples.csv）]
輸出：data/warning_trainer/samples.csv、result.json
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import dtm20 as DTM  # noqa: E402

OUT = REPO / "data" / "warning_trainer"
TW_BOX = (120.0, 21.9, 122.05, 25.35)      # lon_min, lat_min, lon_max, lat_max（含離島以外的本島）


def km(a, b, c, d):
    p = math.pi / 180
    x = math.sin((c - a) * p / 2) ** 2 + math.cos(a * p) * math.cos(c * p) * math.sin((d - b) * p / 2) ** 2
    return 12742 * math.asin(math.sqrt(x))


def terrain_features(dtm, lon, lat, half_m=500, step_m=25):
    """以點為中心 ±half_m 取 DTM 網格 → 高程、平均坡度、起伏、粗糙度。"""
    n = int(2 * half_m / step_m) + 1
    dlat = half_m / 111320.0
    dlon = half_m / (111320.0 * math.cos(math.radians(lat)))
    LO, LA = np.meshgrid(np.linspace(lon - dlon, lon + dlon, n), np.linspace(lat + dlat, lat - dlat, n))
    z = dtm.sample_points(LO, LA).astype(np.float64)
    if (z <= 0).mean() > 0.2:                       # 海或無資料太多
        return None
    gy, gx = np.gradient(z, step_m)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    c = n // 2
    return {"elev_m": float(z[c, c]), "slope_mean_deg": float(slope.mean()), "slope_p90_deg": float(np.percentile(slope, 90)),
            "relief_m": float(z.max() - z.min()), "roughness_m": float(np.std(z - cv2_blur(z)))}


def cv2_blur(z):
    import cv2
    return cv2.GaussianBlur(z.astype(np.float32), (0, 0), 3).astype(np.float64)


def sample_negatives(dtm, hs, n, min_dist_km, seed, elev_min=100.0):
    rng = np.random.default_rng(seed)
    out = []
    tries = 0
    while len(out) < n and tries < n * 200:
        tries += 1
        lon, lat = rng.uniform(TW_BOX[0], TW_BOX[2]), rng.uniform(TW_BOX[1], TW_BOX[3])
        z = float(dtm.sample_points(np.array([lon]), np.array([lat]))[0])
        if z < elev_min:
            continue
        if min(km(lat, lon, h["lat"], h["lon"]) for h in hs) < min_dist_km:
            continue
        f = terrain_features(dtm, lon, lat)
        if f:
            out.append((lon, lat, f))
    return out


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def main():
    ap = argparse.ArgumentParser(description="預警輔助訓練工具原型：圖層—災害一致性量化＋透明加權模型")
    ap.add_argument("--hotspots", default=str(REPO / "data/ardswc_hotspots/top100_consolidated.json"))
    ap.add_argument("--neg-ratio", type=int, default=5)
    ap.add_argument("--min-dist-km", type=float, default=3.0)
    ap.add_argument("--block-km", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--extra-csv", default=None)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    hs = json.loads(Path(args.hotspots).read_text(encoding="utf-8"))
    dtm = DTM.Dtm20Source()
    rows = []
    for h in hs:
        f = terrain_features(dtm, h["lon"], h["lat"])
        if f:
            rows.append({"id": str(h["rank"]), "y": 1, "lon": h["lon"], "lat": h["lat"], **f})
    npos = len(rows)
    for i, (lon, lat, f) in enumerate(sample_negatives(dtm, hs, npos * args.neg_ratio, args.min_dist_km, args.seed)):
        rows.append({"id": f"neg_{i}", "y": 0, "lon": lon, "lat": lat, **f})
    feats = ["elev_m", "slope_mean_deg", "slope_p90_deg", "relief_m", "roughness_m"]
    if args.extra_csv:
        ex = {r["id"]: r for r in csv.DictReader(open(args.extra_csv, encoding="utf-8"))}
        extra = [c for c in next(iter(ex.values())) if c != "id"]
        rows = [r for r in rows if r["id"] in ex]
        for r in rows:
            r.update({c: float(ex[r["id"]][c]) for c in extra})
        feats += extra
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "samples.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "y", "lon", "lat"] + feats)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})

    y = np.array([r["y"] for r in rows])
    X = np.array([[r[f] for f in feats] for r in rows])
    print(f"正例 {int(y.sum())}、負例 {int((1 - y).sum())}")

    # 1. 單因子（取方向使 AUC ≥ 0.5；方向另列）
    single = {}
    for j, f in enumerate(feats):
        a = auc(y, X[:, j])
        sgn = 1 if a >= 0.5 else -1
        s = sgn * X[:, j]
        thr = np.percentile(s, 80)
        sel = s >= thr                                                 # 二元欄位有並列，選到的比例可能遠大於 20%
        hit = float((sel & (y == 1)).sum() / y.sum())                  # 正例被選中的比例（命中率）
        frac = float(sel.mean())
        lift = hit / frac                                              # = 選中者中的正例濃度 / 基準
        single[f] = {"auc": round(max(a, 1 - a), 3), "direction": "高→風險高" if sgn == 1 else "低→風險高",
                     "selected_frac": round(frac, 3), "hit_rate": round(hit, 3), "lift": round(lift, 2)}
        print(f"  {f:18s} AUC {single[f]['auc']:.3f} {single[f]['direction']}  選中 {frac:.0%}  命中 {hit:.0%}  lift {lift:.2f}")

    # 2. 透明多因子模型 + 空間區塊交叉驗證
    gx = np.floor(np.array([r["lon"] for r in rows]) * 111.32 * math.cos(math.radians(23.5)) / args.block_km).astype(int)
    gy = np.floor(np.array([r["lat"] for r in rows]) * 110.57 / args.block_km).astype(int)
    groups = gx * 10000 + gy
    k = min(5, len(set(groups)))
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(class_weight="balanced", max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    cv_auc = auc(y, oof)
    order = np.argsort(-oof)
    topk = int(0.2 * len(y))
    prec20 = float(y[order[:topk]].mean())
    sc = StandardScaler().fit(X)
    m = LogisticRegression(class_weight="balanced", max_iter=2000).fit(sc.transform(X), y)
    coef = dict(zip(feats, (m.coef_[0]).round(3).tolist()))
    w = np.abs(m.coef_[0]); w = (w / w.sum() * 100).round(1).tolist()
    print(f"\n空間區塊 CV（{k} 折，{args.block_km:.0f} km 區塊）AUC {cv_auc:.3f}；前 20% 高風險中的正例比例 {prec20:.0%}（基準 {y.mean():.0%}）")
    print("標準化係數（正＝越大越偏熱點）:", coef)
    res = {"n_pos": int(y.sum()), "n_neg": int((1 - y).sum()), "features": feats, "single_factor": single,
           "multi": {"cv": f"GroupKFold k={k}, block {args.block_km} km", "cv_auc": round(cv_auc, 3),
                     "precision_top20pct": round(prec20, 3), "base_rate": round(float(y.mean()), 3),
                     "std_coef": coef, "relative_weight_pct": dict(zip(feats, w))},
           "caveat": "正例為有通報的災害（曝險偏差）；負例為隨機山區點；僅靜態 DTM 特徵；非真實災害機率。"}
    (OUT / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
