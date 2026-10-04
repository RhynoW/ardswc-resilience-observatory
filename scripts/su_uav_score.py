"""UAV 對照圖盲測判讀（uav_validation/verdicts.json）對照 SU 圖層類別（key.json）→ score.json。
比對基準：拍攝日可對位年度圖層（aligned）且該對有類別者，用「對位年度配對」類別；否則用 SU 全期類別（2021→2024，標 basis=全期）。
一致性：完全一致／相容（擴大↔持續、縮減↔持續、首次出現↔擴大、消失↔縮減）／不一致；「無法判讀」與全期類別「短暫出現」不計入分母。
另算「有無裸露」一致率。限制：單一判讀者（AI）、樣本 16、UAV 覆蓋偏向災區、河道礫石灘被圖層視為裸露而非崩塌。
"""
import json
import sys
from collections import Counter
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "data" / "biggis_interp" / "poc" / "uav_validation"
MAP = {"持續裸露": "持續"}
COMPAT = {("擴大", "持續"), ("持續", "擴大"), ("縮減", "持續"), ("持續", "縮減"), ("首次出現", "擴大"), ("擴大", "首次出現"), ("消失", "縮減"), ("縮減", "消失")}


def main():
    key = json.load(open(D / "key.json", encoding="utf-8"))
    VER = sys.argv[1] if len(sys.argv) > 1 else "verdicts.json"      # 可改用人工判讀檔（如 verdicts_human.json）
    ver = json.load(open(D / VER, encoding="utf-8"))
    rows = []
    for u, k in key.items():
        if u not in ver:      # 人工判讀檔未完成的項目略過
            continue
        v = ver[u]
        if k["aligned"] and k["pair_layer_category"]:
            layer, basis, yrs = k["pair_layer_category"], "對位年度", k["pair_years"]
            bare = max(k["bare_ha"][str(y)] for y in yrs) >= 0.5
        else:
            layer, basis, yrs = k["layer_category"], "全期", None
            bare = max(k["bare_ha"].values()) >= 0.5
        layer = MAP.get(layer, layer)
        if v["verdict"] == "無法判讀" or layer == "短暫出現":
            m = "排除"
        elif v["verdict"] == layer:
            m = "一致"
        elif (layer, v["verdict"]) in COMPAT:
            m = "相容"
        else:
            m = "不一致"
        rows.append({"u": u, "su_id": k["su_id"], "basis": basis, "layer": layer, "verdict": v["verdict"], "conf": v["conf"], "match": m,
                     "bare_match": None if v["verdict"] == "無法判讀" else (bare == (v["verdict"] != "無崩塌")), "note": v["note"],
                     "dates": [f["date"] for f in k["frames"] if f]})
    def summ(sel):
        c = Counter(r["match"] for r in sel)
        n = len(sel) - c["排除"]
        b = [r["bare_match"] for r in sel if r["bare_match"] is not None]
        return {"n": len(sel), "excluded": c["排除"], "一致": c["一致"], "相容": c["相容"], "不一致": c["不一致"],
                "exact_rate": round(c["一致"] / n, 3) if n else None, "exact_or_compat_rate": round((c["一致"] + c["相容"]) / n, 3) if n else None,
                "bare_presence_agreement": round(sum(b) / len(b), 3) if b else None}
    out = {"overall": summ(rows), "by_basis": {b: summ([r for r in rows if r["basis"] == b]) for b in ("對位年度", "全期")},
           "by_layer": {s: summ([r for r in rows if r["layer"] == s]) for s in sorted({r["layer"] for r in rows})},
           "confusion(layer→verdict)": {f"{a}→{b}": n for (a, b), n in sorted(Counter((r["layer"], r["verdict"]) for r in rows).items())}, "rows": rows}
    (D / ("score.json" if VER == "verdicts.json" else "score_" + Path(VER).stem + ".json")).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False, indent=1))
    for r in rows:
        print(r["u"], r["basis"], r["layer"], "→", r["verdict"], r["conf"], r["match"], r["bare_match"])


if __name__ == "__main__":
    main()
