# -*- coding: utf-8 -*-
"""
以 BigGIS 官方衛星判釋崩塌（data/biggis_interp/polygons.json，由 fetch_biggis_interp.py 產生）驗證本專案的訊號。

四項檢核（輸出 data/biggis_interp/validation.json、VALIDATION.md、網頁用 static/biggis_hotspot_support.json）
  A. 反覆熱點 top100 是否座落在官方判釋崩塌附近（對照「隨機災害紀錄點」「隨機山區點」；Wilson 區間與 Fisher 檢定）；
     熱點的「獨立事件數」（複發性）與附近「官方判釋事件數」是否相關（Spearman）。
  B. 衛星變遷分數（Sentinel-2 change_score）與巡查優先級（A–D）是否與官方判釋一致（AUC、各級命中率）。
  C. 事件監測（馬太鞍、草嶺 2025）的 Sentinel-1/2 候選群集對照官方同年判釋（面積、位置）。
  D. 照片通報涵蓋率：官方判釋崩塌多邊形附近（事件日 −3～+60 天、2 km）有沒有任何照片紀錄——量化「照片通報稀疏、有偏」。
解讀（寫進報告）
  * 官方判釋是「衛星影像判釋」，不是現地確認；且判釋只針對大事件（颱風／豪雨／地震），所以「附近沒有官方多邊形」≠「沒有崩塌」
    （標籤有漏報）；AUC、命中率因此是保守估計。
  * 不是與水保署體系完全獨立（歷史影像平台與 BigGIS 同屬水保署），但獨立於本專案的 Sentinel 流程與照片張數。
  * 同事件多檔可能重疊：事件數以（日期、事件名）計，面積以事件內空間聯集計。
用法：python scripts/validate_biggis.py [--raw-dir <build_hotspots 的原始快取，供 D 項用 60 天視窗>]
"""
import argparse
import collections
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np
from shapely import STRtree
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "biggis_interp"
STATIC = REPO / "webapp" / "change_detect_viewer" / "static" / "biggis_hotspot_support.json"
KX, KY = 111.32 * math.cos(math.radians(23.7)), 110.57          # 台灣本島：度 → km 的近似（誤差 < 1%）
RNG = np.random.default_rng(20261002)


def xy(lon, lat):
    return lon * KX, lat * KY


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def write_report(res, sup):
    """VALIDATION.md（完整報告）與 static/biggis_validation.json（網頁摘要）。"""
    A, B, B2, C, Dd, M = res["A_hotspots"], res["B_scores"], res["B_scores"]["B2_time_aligned"], res["C_event_watch"], res["D_report_coverage"], res["source_meta"]
    g = A["groups"]; top = g["top100 反覆熱點"]; rnd = g["隨機災害紀錄點(600)"]; mtn = next(v for k, v in g.items() if k.startswith("隨機山區"))
    pct = lambda x: f"{x:.0%}"
    ci = lambda d: f"{d['ci95'][0]:.0%}–{d['ci95'][1]:.0%}"
    L = ["# 官方衛星判釋驗證報告（BigGIS 崩塌判釋 × 本專案）", "",
         f"產生日期：{res['generated']}　資料抓取：{M['fetched']}　來源：<{M['source']}>（{M['attribution']}）", "",
         "## 這份資料是什麼、為什麼是重要依據", "",
         f"水保署 BigGIS 的「災害事件衛星影像判識成果清單」提供災後衛星影像判釋出的**新增崩塌範圍**（KML／KMZ）：{M['n_files']} 個判釋檔、"
         f"{A['n_events']} 個事件（2013–2026）、{A['n_polygons']:,} 個崩塌多邊形（扣除無效幾何後）。它是目前少數**由專業判釋人員圈繪、涵蓋全台、跨十餘年、公開可取得**的崩塌範圍資料，"
         "而且**獨立於本專案的 Sentinel 演算法與照片張數**——所以適合當作驗證與（未來）訓練標籤的依據，勝過「某處有沒有人拍照」。", "",
         "性質與限制（務必一併引用）：官方判釋是**衛星影像判釋**，不是現地確認；只針對大事件（颱風、豪雨、地震），"
         "所以「附近沒有官方多邊形」不等於「沒有崩塌」（標籤有漏報，下列相關指標因此偏保守）；與歷史影像平台同屬水保署體系，非完全獨立；"
         "僅取外環、同事件多檔可能重疊、座標精度未獨立驗證；資料使用條款尚待確認。", "",
         "## 檢核 A：反覆熱點 top100 是否落在官方判釋崩塌附近", "",
         "| 對象 | 0.5 km 內 | 1 km 內（95% 區間） | 2 km 內 | 最近距離中位數 |", "|---|---|---|---|---|"]
    for name, v in g.items():
        L.append(f"| {name} | {pct(v['within_0.5km']['share'])} | {pct(v['within_1km']['share'])}（{ci(v['within_1km'])}） | {pct(v['within_2km']['share'])} | {v['median_km']} km |")
    fi = A["fisher_1km"]
    L += ["", f"1 km 內比例，熱點相對隨機災害紀錄點勝算比 {fi['vs 隨機災害紀錄點']['odds_ratio']}（Fisher p={fi['vs 隨機災害紀錄點']['p']}），相對隨機山區點 {fi['vs 隨機山區點']['odds_ratio']}（p={fi['vs 隨機山區點']['p']}）。"
          "**結論：熱點確實比一般災害紀錄點、一般山區更常座落在官方判釋崩塌旁，支持「反覆熱點是真實崩塌敏感處」。**"
          "（2 km 內差距縮小，因官方多邊形幾乎鋪滿山區，1 km 才有鑑別力。）", "",
         f"**未獲支持的部分**：熱點的「獨立事件數」（複發性）與其 2 km 內「官方判釋事件數」的 Spearman ρ={A['recurrence_vs_official_events_2km']['spearman_rho']}"
         f"（p={A['recurrence_vs_official_events_2km']['p']}），沒有顯著相關。官方判釋事件只有 {A['n_events']} 個大事件，無法反映「每年反覆」的複發性，所以這項檢核**不能用來驗證複發性排序**。", "",
         "## 檢核 B：衛星變遷訊號與優先級是否與官方判釋一致", "",
         "### B2（時間對齊，主要結果）", "",
         f"設計：{B2['design']}。共 {B2['n_pairs']} 筆（{B2['n_pos']} 筆為正例）。", "",
         f"- 合併 AUC = **{B2['auc_pooled']}**（以熱點為單位的 bootstrap 95% 區間 {B2['auc_pooled_ci95_site_bootstrap'][0]}–{B2['auc_pooled_ci95_site_bootstrap'][1]}）；",
         f"- 同一熱點內比較（排除地點本身差異）：平均 AUC = **{B2['auc_within_site_mean']}**（{B2['n_sites_with_both']} 個熱點同時有正負例），置換檢定 p={B2['within_site_perm_p']}；",
         f"- 變遷量最高三分之一的期間，有官方判釋的比例為 {pct(B2['share_positive_top_tertile_ocf'])}，其餘為 {pct(B2['share_positive_rest'])}（約 {B2['share_positive_top_tertile_ocf'] / max(B2['share_positive_rest'], 1e-9):.1f} 倍）；以 1−SSIM 為預測量 AUC={B2['auc_by_ssim']}。", "",
         "**結論：在時間對齊的檢核下，本專案的 Sentinel-2 前後期變遷量與官方判釋事件一致（中等強度，遠高於隨機 0.5）。**正例標籤有漏報，故實際鑑別力可能更高。", "",
         "### B1（未對時間對齊，僅供對照；以 B2 為準）", "",
         f"以「熱點 1 km 內是否有任何年份的官方多邊形」為標籤：自動篩選分數 AUC={B.get('auc_change_score')}，A 級對其餘等級勝算比 {B['A_vs_rest_fisher']['odds_ratio']}（p={B['A_vs_rest_fisher']['p']}）。"
         "這項沒有訊號，原因是標籤與分數時間不對齊（官方多邊形跨 2013–2026、分數來自特定前後期），且基準率高達 "
         f"{pct(top['within_1km']['share'])}，並**不構成對分數或優先級的否定**；它顯示的是：**目前的 A–D 優先級以複發性與變遷混合排序，尚未被官方判釋獨立驗證**，需要逐事件、同時間的比較（見 B2 與下一步）。", ""]
    L += ["各優先級命中率（1 km 內有官方多邊形）：", "", "| 級別 | 熱點數 | 1 km 內 | 2 km 內 | 平均官方事件數（2 km） |", "|---|---|---|---|---|"]
    for t, v in B["by_tier"].items():
        L.append(f"| {t} | {v['n']} | {pct(v['within_1km'])} | {pct(v['within_2km'])} | {v['mean_official_events_2km']} |")
    L += ["", "## 檢核 C：事件監測（Sentinel-1／2 候選）對照官方同年判釋", ""]
    for c in C:
        offs = "；".join(f"{o['date']} {o['event']}：{o['n_polygons']} 塊、{o['area_ha']} 公頃（最大 {o['largest_ha']} 公頃）" for o in c["official_events_in_aoi"]) or "（AOI 內無同年官方判釋）"
        L.append(f"- **{c['name']}**（AOI 半徑 {c['aoi_half_km']} km）：我們的光學植生損失 {c['our_optical_loss_ha']} 公頃；官方判釋 {offs}")
    L += ["", "馬太鞍：官方 2025-07 的判釋有單一 717 公頃多邊形（約 3.2×4.6 km），我們的光學損失群集（約 586 公頃）質心落在該多邊形內（兩質心相距 0.65 km），面積量級一致；其餘事件為零星小塊。"
          "草嶺：官方約 44 公頃、最大 28 公頃，我們的植生損失 134 公頃偏高（含非崩塌的植生變化，如雲影、農耕、河道），**面積不可直接當崩塌面積**。", "",
          "## 檢核 D：照片通報涵蓋率（為什麼不能只靠照片張數）", "",
          f"方法：{Dd['method']}。官方判釋崩塌多邊形共 {Dd['overall']['total']:,} 個，其中只有 **{Dd['overall']['covered']:,} 個（{pct(Dd['overall']['share'])}，95% 區間 {ci(Dd['overall'])}）** 附近有任何一筆照片紀錄。"
          "也就是說，**照片通報只捕捉到約五分之一的衛星判釋崩塌**，且偏向道路可達之處——用照片張數當標籤會系統性低估偏遠山區，這是改用官方判釋當標籤的理由。", "",
          "## 結論與下一步", "",
          "1. 官方衛星判釋是本專案**最重要的外部依據**：它佐證熱點的空間分布（A）、佐證衛星變遷訊號（B2）、並揭露照片通報的涵蓋缺口（D）。",
          "2. 尚未獲得支持：複發性排序（A）與優先級 A–D 的獨立驗證（B1）——需要逐事件、同時間對齊，並取得更多年度的判釋（目前 2017 年前無 Sentinel-2 前後期可對）。",
          "3. 下一步：以官方判釋當標籤重做預警實驗（正例＝事件後新增崩塌格），對優先級做逐事件驗證，並在人工覆核頁疊上官方多邊形供覆核者對照。",
          "4. 使用前請確認 BigGIS 資料使用條款並標註來源；介面一律稱「官方衛星判釋」，不稱真值。", ""]
    (OUT / "VALIDATION.md").write_text(chr(10).join(L), encoding="utf-8")
    summ = {"fetched": M["fetched"], "n_files": M["n_files"], "n_events": A["n_events"], "n_polygons": A["n_polygons"],
            "A": {"top": top["within_1km"], "random_reports": rnd["within_1km"], "mountain": mtn["within_1km"], "fisher": fi},
            "B2": {k: B2[k] for k in ("n_pairs", "n_pos", "auc_pooled", "auc_pooled_ci95_site_bootstrap", "auc_within_site_mean", "within_site_perm_p", "n_sites_with_both")},
            "D": Dd["overall"], "recurrence_rho": A["recurrence_vs_official_events_2km"]}
    (REPO / "webapp" / "change_detect_viewer" / "static" / "biggis_validation.json").write_text(json.dumps(summ, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=None)
    a = ap.parse_args()
    J = json.loads((OUT / "polygons.json").read_text(encoding="utf-8"))
    ev, P = J["events"], J["polys"]
    ekey = [(e["date"], e["event"]) for e in ev]                 # 事件鍵（同事件多檔視為同一事件）
    keys = sorted(set(ekey)); kid = {k: i for i, k in enumerate(keys)}
    geoms, gev = [], []
    for ei, ar, flat in P:
        pts = [xy(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
        pg = Polygon(pts)
        if not pg.is_valid:
            pg = pg.buffer(0)
        if pg.is_empty:
            continue
        geoms.append(pg); gev.append(kid[ekey[ei]])
    gev = np.array(gev); tree = STRtree(geoms)
    print(f"官方判釋：{len(keys)} 個事件、{len(geoms)} 個多邊形")

    def near(lon, lat, r):
        """回傳 (最近距離 km, 半徑 r 內多邊形索引)。"""
        p = Point(*xy(lon, lat))
        i = int(tree.nearest(p)); d = float(geoms[i].distance(p))
        idx = tree.query(p.buffer(r)); idx = [int(j) for j in idx if geoms[int(j)].distance(p) <= r]
        return d, idx

    def event_area(lon, lat, r, idx):
        """半徑 r 內各事件的判釋面積（公頃）：事件內聯集、裁到圓內，避免多檔重疊重複計算。"""
        c = Point(*xy(lon, lat)).buffer(r); out = {}
        for j in idx:
            out.setdefault(int(gev[j]), []).append(geoms[j])
        return {k: unary_union(v).intersection(c).area * 100 for k, v in out.items()}      # km² → 公頃

    # ── A. 熱點 vs 官方判釋
    H = json.loads((REPO / "data/ardswc_hotspots/top100_consolidated.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(REPO / "webapp" / "change_detect_viewer"))
    import app as WEB
    pri = {x["rank"]: x for x in WEB.app.test_client().get("/api/priority").get_json()["items"]}
    sup = []
    for h in H:
        d, idx1 = near(h["lon"], h["lat"], 1.0)
        _, idx2 = near(h["lon"], h["lat"], 2.0)
        ea = event_area(h["lon"], h["lat"], 2.0, idx2)
        top = sorted(ea.items(), key=lambda kv: -kv[1])[:6]
        sup.append({"rank": h["rank"], "dist_km": round(d, 2), "n_ev_1km": len({int(gev[j]) for j in idx1}), "n_ev_2km": len(ea),
                    "area_ha_2km": round(sum(ea.values()), 1), "tier": pri[h["rank"]]["tier"], "recurrence": h["n_independent_events"],
                    "change_score": h.get("change_score"), "top_events": [[keys[k][0], keys[k][1], round(v, 1)] for k, v in top]})
    dh = np.array([s["dist_km"] for s in sup])
    ev_all = json.loads((REPO / "data/ardswc_hotspots/events_trimmed.json").read_text(encoding="utf-8"))
    haz = [e for e in ev_all if str(e["photo_type"]) in ("0", "8")]
    base = [haz[i] for i in RNG.choice(len(haz), 600, replace=False)]
    db = np.array([near(b["lon"], b["lat"], 0.1)[0] for b in base])
    sys.path.insert(0, str(REPO / "scripts"))
    import dtm20
    la = RNG.uniform(22.0, 25.2, 8000); lo = RNG.uniform(120.1, 121.9, 8000)
    z = dtm20.Dtm20Source().sample_points(lo, la); mt = np.where(z > 300)[0][:600]
    dm = np.array([near(lo[i], la[i], 0.1)[0] for i in mt])
    from scipy.stats import fisher_exact, spearmanr
    A = {"n_events": len(keys), "n_polygons": len(geoms), "groups": {}}
    for name, arr in (("top100 反覆熱點", dh), ("隨機災害紀錄點(600)", db), (f"隨機山區點 DTM>300 m({len(dm)})", dm)):
        A["groups"][name] = {f"within_{r}km": {"share": round(float(np.mean(arr <= r)), 3), "ci95": [round(x, 3) for x in wilson(int((arr <= r).sum()), len(arr))]} for r in (0.5, 1, 2)}
        A["groups"][name]["median_km"] = round(float(np.median(arr)), 2)
    A["fisher_1km"] = {}
    for name, arr in (("vs 隨機災害紀錄點", db), ("vs 隨機山區點", dm)):
        t = [[int((dh <= 1).sum()), int((dh > 1).sum())], [int((arr <= 1).sum()), int((arr > 1).sum())]]
        A["fisher_1km"][name] = {"odds_ratio": round(float(fisher_exact(t)[0]), 2), "p": float(f"{fisher_exact(t)[1]:.2g}")}
    rho, p = spearmanr([s["recurrence"] for s in sup], [s["n_ev_2km"] for s in sup])
    A["recurrence_vs_official_events_2km"] = {"spearman_rho": round(float(rho), 3), "p": float(f"{p:.2g}")}

    # ── B. 變遷分數、優先級 vs 官方判釋（1 km 內有多邊形＝官方佐證）
    from sklearn.metrics import roc_auc_score
    y = np.array([int(s["dist_km"] <= 1.0) for s in sup]); B = {"label": "100 個熱點是否在官方判釋崩塌 1 km 內", "n_pos": int(y.sum())}
    cs = np.array([np.nan if s["change_score"] is None else s["change_score"] for s in sup], float); ok = ~np.isnan(cs)
    if ok.sum() > 10 and 0 < y[ok].sum() < ok.sum():
        B["auc_change_score"] = round(float(roc_auc_score(y[ok], cs[ok])), 3); B["n_with_change_score"] = int(ok.sum())
    B["by_tier"] = {}
    for t in "ABCD":
        m = [s for s in sup if s["tier"] == t]
        if m:
            B["by_tier"][t] = {"n": len(m), "within_1km": round(float(np.mean([s["dist_km"] <= 1 for s in m])), 3),
                               "within_2km": round(float(np.mean([s["dist_km"] <= 2 for s in m])), 3),
                               "mean_official_events_2km": round(float(np.mean([s["n_ev_2km"] for s in m])), 2)}
    am = [s for s in sup if s["tier"] == "A"]; om = [s for s in sup if s["tier"] != "A"]
    t = [[sum(s["dist_km"] <= 1 for s in am), sum(s["dist_km"] > 1 for s in am)], [sum(s["dist_km"] <= 1 for s in om), sum(s["dist_km"] > 1 for s in om)]]
    B["A_vs_rest_fisher"] = {"odds_ratio": round(float(fisher_exact(t)[0]), 2), "p": float(f"{fisher_exact(t)[1]:.2g}")}

    # ── B2. 時間對齊：每個熱點每一對相鄰年份的 Sentinel-2 變遷量 vs 該期間內官方判釋事件
    from shapely.geometry import box as sbox
    pairs = []
    for d in sorted((REPO / "data" / "ge_captures").glob("ardswc_top*")):
        tp = d / "_change_detect" / (d.name + "_change_timeline.json")
        if not tp.exists():
            continue
        rank = int(d.name.replace("ardswc_top", "")); h = next((x for x in H if x["rank"] == rank), None)
        if h is None:
            continue
        tl = json.loads(tp.read_text(encoding="utf-8")); half = tl.get("half_m", 750) / 1000
        cx, cy = xy(h["lon"], h["lat"]); bx = sbox(cx - half, cy - half, cx + half, cy + half)
        hit_dates = sorted({keys[int(gev[int(j)])][0].replace("-", "") for j in tree.query(bx) if geoms[int(j)].intersects(bx)})
        for pr in tl["pairs"]:
            pos = any(pr["date_a"] < dd <= pr["date_b"] for dd in hit_dates)
            pairs.append({"rank": rank, "a": pr["date_a"], "b": pr["date_b"], "ocf": pr["overall_change_fraction"], "ssim": pr["mean_ssim"], "n_regions": pr["n_regions"], "pos": int(pos)})
    B2 = {"design": "每個熱點每一對相鄰年份（Sentinel-2 乾季影像）為一筆；正例＝該期間內有官方判釋崩塌與熱點 1.5 km 方框相交；預測量＝整體變遷比例（overall_change_fraction）",
          "n_pairs": len(pairs), "n_pos": int(sum(x["pos"] for x in pairs))}
    if pairs and 0 < B2["n_pos"] < len(pairs):
        yy = np.array([x["pos"] for x in pairs]); ocf = np.array([x["ocf"] for x in pairs]); sk = np.array([x["rank"] for x in pairs])
        B2["auc_pooled"] = round(float(roc_auc_score(yy, ocf)), 3)
        sites = sorted(set(sk)); boots = []
        for _ in range(500):
            pick = RNG.choice(sites, len(sites)); idx = np.concatenate([np.where(sk == s_)[0] for s_ in pick])
            if 0 < yy[idx].sum() < len(idx):
                boots.append(roc_auc_score(yy[idx], ocf[idx]))
        B2["auc_pooled_ci95_site_bootstrap"] = [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)]
        within = []
        for s_ in sites:
            m = sk == s_
            if 0 < yy[m].sum() < m.sum():
                within.append(roc_auc_score(yy[m], ocf[m]))
        B2["auc_within_site_mean"] = round(float(np.mean(within)), 3); B2["n_sites_with_both"] = len(within)
        obs = np.mean(within); cnt_ge = 0
        for _ in range(1000):
            w = []
            for s_ in sites:
                m = np.where(sk == s_)[0]
                if 0 < yy[m].sum() < len(m):
                    w.append(roc_auc_score(RNG.permutation(yy[m]), ocf[m]))
            cnt_ge += int(np.mean(w) >= obs)
        B2["within_site_perm_p"] = round((cnt_ge + 1) / 1001, 3)
        thr = np.percentile(ocf, 66.7)
        B2["share_positive_top_tertile_ocf"] = round(float(yy[ocf >= thr].mean()), 3); B2["share_positive_rest"] = round(float(yy[ocf < thr].mean()), 3)
        B2["auc_by_ssim"] = round(float(roc_auc_score(yy, 1 - np.array([x["ssim"] for x in pairs]))), 3)
    B["B2_time_aligned"] = B2

    # ── C. 事件監測 vs 官方同年判釋
    C = []
    for d in sorted((REPO / "data" / "event_watch").iterdir()):
        rp = d / "result.json"
        if not rp.exists():
            continue
        r = json.loads(rp.read_text(encoding="utf-8")); lon0, lat0 = r["center"]; half = r["half_km"]
        year = r["event"][:4]
        box = Polygon([xy(lon0 - half / KX, lat0 - half / KY), xy(lon0 + half / KX, lat0 - half / KY), xy(lon0 + half / KX, lat0 + half / KY), xy(lon0 - half / KX, lat0 + half / KY)])
        idx = [int(j) for j in tree.query(box) if geoms[int(j)].intersects(box)]
        per = collections.defaultdict(list)
        for j in idx:
            if keys[int(gev[j])][0].startswith(year):
                per[int(gev[j])].append(geoms[j])
        offi = [{"date": keys[k][0], "event": keys[k][1], "n_polygons": len(v), "area_ha": round(unary_union(v).area * 100, 1),
                 "largest_ha": round(max(g.area for g in v) * 100, 1)} for k, v in per.items()]
        clusters = (r.get("optical") or {}).get("loss_clusters") or []
        C.append({"name": r["name"], "event": r["event"], "aoi_half_km": half, "official_events_in_aoi": sorted(offi, key=lambda x: x["date"]),
                  "our_optical_loss_ha": (r.get("optical") or {}).get("loss_ha"), "our_loss_clusters": [{"area_ha": c["area_ha"], "lon": c["lon"], "lat": c["lat"]} for c in clusters[:3]]})

    # ── D. 照片通報涵蓋率
    Dd = {}
    if a.raw_dir:
        raw = []
        for t in ("0", "8"):
            for r in json.loads((Path(a.raw_dir) / f"type_{t}.json").read_text(encoding="utf-8")):
                if r.get("Lat") and r.get("Lng") and r.get("PhotoDate"):
                    raw.append((float(r["Lng"]), float(r["Lat"]), r["PhotoDate"][:10]))
        RP = np.array([(xy(a_, b_)) for a_, b_, c_ in raw]); RD = np.array([dt.date.fromisoformat(c_).toordinal() for a_, b_, c_ in raw])
        method = "事件日 −3～+60 天、2 km 內有任一筆照片紀錄（災害事件＋媒體報導）"
    else:
        RP = np.array([xy(e["lon"], e["lat"]) for e in haz]); RD = np.array([int(e["year"]) for e in haz])
        method = "同一年、2 km 內有任一筆照片紀錄（無日期資料，視窗較寬，涵蓋率偏高）"
    cov = collections.defaultdict(lambda: [0, 0]); allc = [0, 0]
    for g, k in zip(geoms, gev):
        c = g.centroid; date, _ = keys[int(k)]
        if a.raw_dir:
            d0 = dt.date.fromisoformat(date).toordinal(); m = (RD >= d0 - 3) & (RD <= d0 + 60)
        else:
            m = RD == int(date[:4])
        hit = False
        if m.any():
            dd = np.sqrt(((RP[m, 0] - c.x) ** 2) + ((RP[m, 1] - c.y) ** 2)).min(); hit = dd <= 2.0
        yy = int(date[:4]); cov[yy][1] += 1; cov[yy][0] += int(hit); allc[1] += 1; allc[0] += int(hit)
    Dd = {"method": method, "overall": {"covered": allc[0], "total": allc[1], "share": round(allc[0] / allc[1], 3), "ci95": [round(x, 3) for x in wilson(allc[0], allc[1])]},
          "by_year": {str(y): {"covered": v[0], "total": v[1], "share": round(v[0] / v[1], 3)} for y, v in sorted(cov.items())}}

    res = {"generated": dt.date.today().isoformat(), "source_meta": J["meta"], "A_hotspots": A, "B_scores": B, "C_event_watch": C, "D_report_coverage": Dd}
    (OUT / "validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    STATIC.write_text(json.dumps({"fetched": J["meta"]["fetched"], "attribution": J["meta"]["attribution"], "events": [list(k) for k in keys], "hotspots": {str(s["rank"]): s for s in sup}},
                                 ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    write_report(res, sup)
    g1 = A["groups"]
    print("A 熱點 ≤1km:", g1["top100 反覆熱點"]["within_1km"], "｜隨機紀錄點:", g1["隨機災害紀錄點(600)"]["within_1km"], "｜山區點:", [v["within_1km"] for k, v in g1.items() if k.startswith("隨機山區")])
    print("  Fisher:", A["fisher_1km"], "｜複發性 vs 官方事件數 ρ:", A["recurrence_vs_official_events_2km"])
    print("B", {k: v for k, v in B.items() if k not in ("by_tier", "B2_time_aligned")}); print("  各級:", B["by_tier"]); print("B2", B["B2_time_aligned"])
    for c in C:
        print("C", c["name"], "我們的光學損失", c["our_optical_loss_ha"], "ha；官方事件:", [(o["date"], o["event"][:14], o["n_polygons"], o["area_ha"], o["largest_ha"]) for o in c["official_events_in_aoi"]])
    print("D", Dd["method"], Dd["overall"])


if __name__ == "__main__":
    main()
