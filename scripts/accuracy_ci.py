"""驗證數字的信賴區間與分層加權估計（方法參考 Olofsson et al. 2014、Stehman 2014 的分層抽樣估計；自寫，未使用 Lite Classify 程式碼）。
- 比例的 95% 信賴區間：Wilson 分數區間（小樣本、比例接近 0 或 1 時比常態近似穩健）。
- 分層抽樣的整體比例：p̂ = Σ W_h p̂_h，W_h = N_h / N（N_h＝該層母體數）；
  變異數 V = Σ W_h² (1 − n_h/N_h) p̃_h (1 − p̃_h) / (n_h − 1)，其中 p̃_h = (x_h + 1)/(n_h + 2)（避免 p̂_h=0 或 1 時變異數為 0 的低估）；
  95% CI = p̂ ± 1.96 √V，截在 [0, 1]。樣本很小時常態近似仍偏樂觀，只當作量級參考。
- 花蓮、太魯閣的偵測率與誤報率是全數比對（不是抽樣）；這裡的區間把坡面視為同類事件下的獨立試驗，表示「換一場類似事件時」的不確定性，不是抽樣誤差。
輸出：data/biggis_interp/poc/accuracy_ci.json，並印出表格。
用法：python scripts/accuracy_ci.py
"""
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
POC = REPO / "data" / "biggis_interp" / "poc"
AREAS = REPO / "data" / "biggis_interp" / "areas"
LAYER_END = "2025-03-25"


def jl(p):
    return json.load(open(p, encoding="utf-8"))


def wilson(x, n, z=1.96):
    if n == 0:
        return None, None
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)


def rec(x, n, note=""):
    lo, hi = wilson(x, n)
    return {"x": x, "n": n, "p": round(x / n, 3) if n else None, "ci95": [lo, hi], "note": note}


def stratified(strata):
    """strata: [{name, N, n, x}] → 加權比例與 95% CI。"""
    N = sum(s["N"] for s in strata)
    p = sum(s["N"] / N * s["x"] / s["n"] for s in strata)
    v = 0.0
    for s in strata:
        W, n = s["N"] / N, s["n"]
        pt = (s["x"] + 1) / (n + 2)
        fpc = 1 - n / s["N"] if s["N"] > n else 0.0
        v += W * W * fpc * pt * (1 - pt) / max(n - 1, 1)
    se = math.sqrt(v)
    return {"p": round(p, 3), "ci95": [round(max(0.0, p - 1.96 * se), 3), round(min(1.0, p + 1.96 * se), 3)], "N": N,
            "strata": [{**s, "W": round(s["N"] / N, 3), "p_h": round(s["x"] / s["n"], 3), "ci95_h": list(wilson(s["x"], s["n"]))} for s in strata]}


def ndre_pools():
    """與 ndre_validation.select() 相同的抽樣框（有效坡面 ≥3 ha、排除測試包用過的坡面）。"""
    nd = jl(POC / "s2ts" / "su_ndre.json")["su"]
    T = {t["su_id"]: t for t in jl(POC / "su_table_eff.json")}
    mu = {u["su_id"]: u for u in jl(POC / "su_meta_eff.json")["units"]}
    kp = POC / "su_usability" / "key_usability.json"
    used = {v["su_id"] for v in jl(kp).values()} if kp.exists() else set()
    ok = lambda s: s in T and s not in used and mu[s]["eff_area_ha"] >= 3.0
    drops = {s: v["drop"] for s, v in nd.items() if v.get("drop") and ok(s)}
    return {"N1": sum(1 for s, d in drops.items() if d["date"] <= LAYER_END and T[s]["category"] != "無崩塌"),
            "N2": sum(1 for s, d in drops.items() if d["date"] <= LAYER_END and T[s]["category"] == "無崩塌"),
            "N3": sum(1 for s, d in drops.items() if d["date"] > LAYER_END)}


def main():
    out = {}
    # 1. 影像目視 vs 官方年度圖層「有無裸露」一致（30 個坡面盲測，29 個可判讀）
    hb = jl(POC / "su_validation" / "score_verdicts_human.json")["overall"]
    n = hb["n"] - hb["excluded"]
    out["bare_presence_agreement"] = rec(round(hb["bare_presence_agreement"] * n), n, "人工盲測，單一判讀者；分層配額抽樣，未加權")
    # 2. 舊版「近期有變化」訊號（S1 層，4 個坡面）
    sr = jl(POC / "su_strat" / "strat_result.json")["verdicts_by_stratum"]["S1"]
    out["old_recent_change_S1"] = rec(sr.get("A", 0), sum(sr.values()), "24 坡面分層驗證中的 S1 層（事件有＋近期影像有變化），v1 單元")
    # 3. 高雄 NDRE 持續下降：分層盲判
    key = jl(POC / "ndre_val" / "key.json")
    hum = jl(REPO / "review" / "ndre_val_human.json")
    pools = ndre_pools()
    strata = []
    for h in ("N1", "N2", "N3"):
        ids = [k for k, v in key.items() if v["stratum"] == h and hum.get(k, {}).get("verdict")]
        strata.append({"name": h, "N": pools[h], "n": len(ids), "x": sum(hum[k]["verdict"] == "A" for k in ids)})
    out["ndre_P3_precision_kaohsiung"] = {**stratified(strata), "unweighted": rec(sum(s["x"] for s in strata), sum(s["n"] for s in strata)),
                                          "note": "A＝人工判為後期出現新增或擴大裸露；抽樣框為有效坡面 ≥3 ha、未用於測試包的下降坡面"}
    c = [k for k, v in key.items() if v["stratum"] == "N4" and hum.get(k, {}).get("verdict")]
    out["ndre_control_A_rate_kaohsiung"] = rec(sum(hum[k]["verdict"] == "A" for k in c), len(c), "無下降、無事件、無照片的對照坡面中被判 A 的比例")
    # 4. 花蓮、太魯閣：偵測率與誤報率（全數比對）
    for name in ("taroko", "hualien"):
        d = jl(AREAS / name / "su_ndre" / "su_ndre.json")
        su, ev = d["su"], d["event_ha"]
        ok = [s for s in su if su[s].get("base") is not None]
        evs = [s for s in su if ev.get(s, 0) >= 1.0]          # 含觀測不足者（算未抓到），與 /poc 的 43%、75% 同一定義
        nz = [s for s in ok if ev.get(s, 0) == 0]
        out[f"{name}_event_su_detected"] = rec(sum(bool(su[s]["drop"]) for s in evs), len(evs), "事件崩塌 ≥1 ha 的坡面被抓到下降；全數比對")
        out[f"{name}_false_rate_no_event"] = rec(sum(bool(su[s]["drop"]) for s in nz), len(nz), "無事件坡面被判下降；全數比對")
    (POC / "accuracy_ci.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in out.items():
        if "strata" in v:
            print(f"{k}: 加權 {v['p']:.0%}（95% CI {v['ci95'][0]:.0%}–{v['ci95'][1]:.0%}），未加權 {v['unweighted']['x']}/{v['unweighted']['n']}")
            for s in v["strata"]:
                print(f"   {s['name']} N={s['N']} n={s['n']} x={s['x']} W={s['W']} CI {s['ci95_h']}")
        else:
            print(f"{k}: {v['x']}/{v['n']} = {v['p']:.1%}（95% CI {v['ci95'][0]:.1%}–{v['ci95'][1]:.1%}）")


if __name__ == "__main__":
    main()
