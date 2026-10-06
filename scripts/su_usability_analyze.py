"""實務人員操作測試結果的描述統計（3–5 人，不做統計檢定、不宣稱效率提升百分比）。
輸入：結果資料夾（內含參與者匯出的 usability_*.json，以及專家的 expert_*.json（選用））與 poc/su_usability/key_usability.json。
輸出：終端表格與 <結果資料夾>/summary.json：
  每位參與者、每個條件（原始資料／證據鏈頁）：題數、每題停留秒數中位數、決定分布（建議現勘／可暫緩／資料不足）、信心平均（高3中2低1）；
  兩條件合併：同上；最有幫助的資訊（複選）次數；整體回饋（1–5）與文字回饋；
  專家參考（若有）：各題專家判讀（A–E）的多數與一致程度，並列出「兩條件下的決定 vs 專家判讀」交叉表（只描述，不算準確率——專家 A–E 與「是否現勘」不是同一個問題）。
用法：python scripts/su_usability_analyze.py <結果資料夾>
"""
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KEY = REPO / "data" / "biggis_interp" / "poc" / "su_usability" / "key_usability.json"
CONF = {"高": 3, "中": 2, "低": 1}
DEC = {"go": "建議現勘", "wait": "可暫緩", "na": "資料不足"}


def main():
    d = Path(sys.argv[1])
    key = json.load(open(KEY, encoding="utf-8"))
    P = [json.load(open(f, encoding="utf-8")) for f in sorted(d.glob("usability_*.json"))]
    E = [json.load(open(f, encoding="utf-8")) for f in sorted(d.glob("expert_*.json"))]
    out = {"n_participants": len(P), "n_experts": len(E), "participants": {}, "pooled": {}, "helpful": {}, "feedback": {}, "expert": {}}
    pooled = defaultdict(lambda: {"sec": [], "dec": Counter(), "conf": [], "n": 0})
    helpful = defaultdict(Counter)
    for p in P:
        per = defaultdict(lambda: {"sec": [], "dec": Counter(), "conf": [], "n": 0})
        for c, it in p["items"].items():
            if it["decision"] is None:
                continue
            for tgt in (per[it["condition"]], pooled[it["condition"]]):
                tgt["n"] += 1
                tgt["sec"].append(it["seconds_visible"])
                tgt["dec"][it["decision"]] += 1
                if it["confidence"]:
                    tgt["conf"].append(CONF[it["confidence"]])
            for h in it["helpful"]:
                helpful[it["condition"]][h] += 1
        out["participants"][p["participant"]] = {"group": p["group"], **{k: {"n": v["n"], "median_sec": statistics.median(v["sec"]) if v["sec"] else None, "decisions": {DEC[a]: b for a, b in v["dec"].items()},
                                                                         "mean_conf": round(statistics.mean(v["conf"]), 2) if v["conf"] else None} for k, v in per.items()}}
        out["feedback"][p["participant"]] = p["feedback"]
    out["pooled"] = {k: {"n": v["n"], "median_sec": statistics.median(v["sec"]) if v["sec"] else None, "decisions": {DEC[a]: b for a, b in v["dec"].items()},
                         "mean_conf": round(statistics.mean(v["conf"]), 2) if v["conf"] else None} for k, v in pooled.items()}
    out["helpful"] = {k: dict(v.most_common()) for k, v in helpful.items()}
    if E:
        ex = defaultdict(list)
        for e in E:
            for c, it in e["items"].items():
                if it["verdict"]:
                    ex[c].append(it["verdict"])
        cons = {c: Counter(v).most_common(1)[0][0] for c, v in ex.items()}
        agree = {c: round(Counter(v).most_common(1)[0][1] / len(v), 2) for c, v in ex.items()}
        out["expert"] = {"consensus": cons, "agreement_share": agree, "mean_agreement": round(statistics.mean(agree.values()), 2) if agree else None}
        cross = defaultdict(Counter)
        for p in P:
            for c, it in p["items"].items():
                if it["decision"] and c in cons:
                    cross[(it["condition"], cons[c])][DEC[it["decision"]]] += 1
        out["expert"]["decision_vs_expert"] = {f"{k[0]}|專家{k[1]}": dict(v) for k, v in sorted(cross.items())}
    (d / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1)[:6000])


if __name__ == "__main__":
    main()
