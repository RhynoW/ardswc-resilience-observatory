"""把盲測判讀（verdicts.json）對照 SU 圖層類別（key.json），輸出一致性統計 → su_validation/score.json。

一致性定義：
  完全一致：判讀類別＝圖層類別（圖層「持續裸露」對判讀「持續」）。
  相容（鄰近類別）：擴大↔持續、縮減↔持續、首次出現↔擴大、消失↔縮減；無崩塌只對無崩塌。
  不一致：其他。「無法判讀」不計入分母。
  另算「有無裸露」：圖層該對位年度配對任一期裸露 ≥0.5 ha ＝有；判讀為無崩塌＝無，其餘＝有。
限制：單一判讀者（AI）、樣本小、影像季節與光線不同、10 m 影像看不出小變化。
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "data" / "biggis_interp" / "poc" / "su_validation"
MAP = {"持續裸露": "持續", "擴大": "擴大", "縮減": "縮減", "首次出現": "首次出現", "消失": "消失", "無崩塌": "無崩塌"}
COMPAT = {("擴大", "持續"), ("持續", "擴大"), ("縮減", "持續"), ("持續", "縮減"), ("首次出現", "擴大"), ("擴大", "首次出現"), ("消失", "縮減"), ("縮減", "消失")}


def main():
    key = json.load(open(D / "key.json", encoding="utf-8"))
    VER = sys.argv[1] if len(sys.argv) > 1 else "verdicts.json"      # 可改用人工判讀檔（如 verdicts_human.json）
    ver = json.load(open(D / VER, encoding="utf-8"))
    rows = []
    for vid, k in key.items():
        if vid not in ver:      # 人工判讀檔未完成的項目略過
            continue
        v = ver[vid]
        layer = MAP[k["pair_category"]]
        ya, yb = k["pair_years"]
        bare_layer = max(k["bare_ha"][str(ya)], k["bare_ha"][str(yb)]) >= 0.5
        row = {"v": vid, "su_id": k["su_id"], "method": k["method"], "stratum": k["pair_category"], "pair_years": k["pair_years"],
               "layer_bare_ha": [k["bare_ha"][str(ya)], k["bare_ha"][str(yb)]], "verdict": v["verdict"], "conf": v["conf"], "note": v["note"],
               "dominant_event": k.get("dominant_event"), "category_2021_2024": k["category"]}
        if v["verdict"] == "無法判讀":
            row["match"] = "排除"
        elif v["verdict"] == layer:
            row["match"] = "一致"
        elif (layer, v["verdict"]) in COMPAT:
            row["match"] = "相容"
        else:
            row["match"] = "不一致"
        row["bare_match"] = None if v["verdict"] == "無法判讀" else (bare_layer == (v["verdict"] != "無崩塌"))
        rows.append(row)
    def summ(sel):
        c = Counter(r["match"] for r in sel)
        n = len(sel) - c["排除"]
        b = [r["bare_match"] for r in sel if r["bare_match"] is not None]
        return {"n": len(sel), "excluded": c["排除"], "一致": c["一致"], "相容": c["相容"], "不一致": c["不一致"],
                "exact_rate": round(c["一致"] / n, 3) if n else None, "exact_or_compat_rate": round((c["一致"] + c["相容"]) / n, 3) if n else None,
                "bare_presence_agreement": round(sum(b) / len(b), 3) if b else None}
    out = {"overall": summ(rows), "by_method": {m: summ([r for r in rows if r["method"] == m]) for m in ("Wayback", "S2")},
           "by_layer_category": {s: summ([r for r in rows if r["stratum"] == s]) for s in ("擴大", "首次出現", "縮減", "持續裸露", "消失", "無崩塌")},
           "by_confidence": {c: summ([r for r in rows if r["conf"] == c]) for c in ("高", "中", "低")},
           "confusion(layer→verdict)": {f"{s}→{v}": n for (s, v), n in sorted(Counter((r["stratum"], r["verdict"]) for r in rows).items())},
           "rows": rows}
    (D / ("score.json" if VER == "verdicts.json" else "score_" + Path(VER).stem + ".json")).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
