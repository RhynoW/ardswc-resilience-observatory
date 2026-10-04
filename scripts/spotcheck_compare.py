"""比較 AI 判讀與人工判讀（盲測抽查）：完全一致率、有無崩塌一致率（無崩塌 vs 其他）、混淆矩陣、
人工與圖層類別的一致情況（以人工判讀重算，沿用 su_wayback_score／su_uav_score 的定義）。
前置：人工判讀存成 su_validation/verdicts_human.json、uav_validation/verdicts_human.json（或同一個檔放兩邊；缺的項目略過）。
用法：python scripts/spotcheck_compare.py [人工判讀檔路徑]   → poc/spotcheck/compare.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

POC = Path(__file__).resolve().parent.parent / "data" / "biggis_interp" / "poc"


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    human = json.load(open(src, encoding="utf-8")) if src else {}
    out = {}
    for pre, d in (("V", "su_validation"), ("U", "uav_validation")):
        ai = json.load(open(POC / d / "verdicts.json", encoding="utf-8"))
        hp = POC / d / "verdicts_human.json"
        h = {**(json.load(open(hp, encoding="utf-8")) if hp.exists() else {}), **{k: v for k, v in human.items() if k.startswith(pre)}}
        ids = [k for k in h if k.startswith(pre) and k in ai]
        both = [k for k in ids if ai[k]["verdict"] != "無法判讀" and h[k]["verdict"] != "無法判讀"]
        exact = [k for k in both if ai[k]["verdict"] == h[k]["verdict"]]
        binary = [k for k in both if (ai[k]["verdict"] == "無崩塌") == (h[k]["verdict"] == "無崩塌")]
        diff = [{"id": k, "ai": ai[k]["verdict"], "ai_conf": ai[k]["conf"], "human": h[k]["verdict"], "human_conf": h[k]["conf"], "human_note": h[k].get("note", "")} for k in both if ai[k]["verdict"] != h[k]["verdict"]]
        out[d] = {"n_human": len(ids), "n_both_readable": len(both), "ai_unreadable": sum(ai[k]["verdict"] == "無法判讀" for k in ids), "human_unreadable": sum(h[k]["verdict"] == "無法判讀" for k in ids),
                  "exact_agreement": round(len(exact) / len(both), 3) if both else None, "has_landslide_agreement": round(len(binary) / len(both), 3) if both else None,
                  "confusion(ai→human)": {f"{a}→{b}": n for (a, b), n in sorted(Counter((ai[k]["verdict"], h[k]["verdict"]) for k in both).items())},
                  "disagreements": diff}
    p = POC / "spotcheck" / "compare.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({d: {k: v for k, v in o.items() if k != "disagreements"} for d, o in out.items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
