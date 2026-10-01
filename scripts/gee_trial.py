# -*- coding: utf-8 -*-
"""
Google Earth Engine 試跑：乾季 median composite + NBR 差值，與現有 Sentinel-2 SSIM 並列比較。

動機（2026-09 審查意見）：SSIM 在雲影、色調、對位、季節差異下容易偽陽性。GEE 可取原始 Sentinel-2 SR 反射率，
用 Cloud Score+ 遮雲、各年同季取 median composite（同感測器、同季節），再比光譜指標 NBR——
對色調與季節差異比 RGB 紋理穩健。本腳本只做**小規模試跑**（預設 3 個熱點），驗證做法是否值得擴大到 100 個。

預設對照組（已知 Sentinel Hub + SSIM 的結果，見 data/ge_captures/ardswc_topNN/_change_detect/）：
  #20、#22：2024-02 → 2025-03 出現明顯變化峰值（花蓮秀林，0403 地震後新裸露崩塌）→ 應出現峰值
  #72：整段幾乎無變化（change_score 0.006）→ 應保持平靜
判準：NBR 版本若能在 #20／#22 同一年度對出現峰值、#72 全程低，才有理由擴大；否則不要。

指標定義（全部是「候選訊號」，不是已驗證的崩塌）：
  年度 Y 的 composite = 前一年 11 月到當年 3 月的 median（乾季；避免植生季節差被當成變化）。
  NBR = (B8 − B12) / (B8 + B12)；dNBR = NBR(Y−1) − NBR(Y)；dNBR > --thresh 視為「植生減少／裸露增加」。
  loss_frac = 緩衝區內 dNBR > thresh 的像素比例（分母含無效像素＝只計有雙期有效資料者，見 valid_frac）；
  valid_frac = 兩期 composite 皆有效的像素比例；n_images = 各期 composite 用到的影像張數。
  valid_frac < 0.6 或任一期張數 < --min-images（預設 5）→ 該年度標「資料不足」、不給數值。
  （valid_frac 只看覆蓋、不看張數：1–2 張影像也能蓋滿，但單張 composite 沒有 median 去雜訊的效果，所以另設張數門檻。）

認證：需要 Google Earth Engine 帳號與 Cloud project（非商業使用要在 Cloud console 完成註冊）。
  首次使用：pip install earthengine-api；earthengine authenticate；之後以 --project 或環境變數 GEE_PROJECT 指定專案。
  專案 ID 不寫進原始碼或倉庫。

用法：python scripts/gee_trial.py --project <cloud-project-id> [--ranks 20,22,72] [--years 2017-2025]
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "ardswc_hotspots"
CAPTURES = REPO / "data" / "ge_captures"

S2 = "COPERNICUS/S2_SR_HARMONIZED"
CS_PLUS = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
MIN_VALID_FRAC = 0.6          # 雙期有效像素比例低於此值 → 該年度標「資料不足」
MIN_IMAGES = 5                # 任一期 composite 的影像張數低於此值 → 標「資料不足」（試跑發現 2017–2018 乾季窗只有 1–2 張）


def init_ee(project):
    try:
        import ee
    except ImportError:
        sys.exit("缺少 earthengine-api：pip install earthengine-api")
    try:
        ee.Initialize(project=project)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Earth Engine 初始化失敗：{e}\n請先執行 `earthengine authenticate`，並確認 --project 是已註冊 Earth Engine 的 Cloud project。")
    return ee


def season_composite(ee, geom, year, cs_thresh):
    """年度 Y 的乾季 composite：(Y-1)-11-01 ～ Y-03-31。回傳 (影像, 影像張數 ee.Number)。"""
    col = (ee.ImageCollection(S2).filterBounds(geom).filterDate(f"{year - 1}-11-01", f"{year}-04-01")
           .linkCollection(ee.ImageCollection(CS_PLUS), ["cs_cdf"])
           .map(lambda im: im.updateMask(im.select("cs_cdf").gte(cs_thresh))))
    return col.select(["B8", "B12"]).median(), col.size()


def nbr(img):
    return img.normalizedDifference(["B8", "B12"]).rename("nbr")


def pair_stats(ee, geom, y0, y1, thresh, cs_thresh):
    a, na = season_composite(ee, geom, y0, cs_thresh)
    b, nb = season_composite(ee, geom, y1, cs_thresh)
    na_, nb_ = nbr(a), nbr(b)
    valid = na_.mask().And(nb_.mask())
    d = na_.subtract(nb_)                                   # 正值＝植生減少
    img = ee.Image.cat([
        valid.rename("valid"),
        d.gt(thresh).And(valid).rename("loss"),
        d.lt(-thresh).And(valid).rename("gain"),
        d.updateMask(valid).rename("dnbr"),
    ]).unmask(0)
    r = img.reduceRegion(ee.Reducer.mean(), geom, 10, maxPixels=1e8, bestEffort=False)
    return ee.Dictionary({"valid": r.get("valid"), "loss": r.get("loss"), "gain": r.get("gain"),
                          "n_a": na, "n_b": nb})


def ssim_reference(rank):
    """現有 Sentinel Hub + SSIM 的相鄰配對：[(YYYYMM_a, YYYYMM_b, overall_change_fraction)]。"""
    fp = CAPTURES / f"ardswc_top{rank:02d}" / "_change_detect" / f"ardswc_top{rank:02d}_change_timeline.json"
    if not fp.exists():
        return []
    return [(p["date_a"], p["date_b"], p["overall_change_fraction"])
            for p in json.loads(fp.read_text(encoding="utf-8")).get("pairs", [])]


def main():
    ap = argparse.ArgumentParser(description="GEE 乾季 composite + dNBR 試跑")
    ap.add_argument("--project", default=os.environ.get("GEE_PROJECT"), help="Earth Engine Cloud project ID（或環境變數 GEE_PROJECT）")
    ap.add_argument("--ranks", default="20,22,72")
    ap.add_argument("--years", default="2017-2025", help="起迄年（含），各年用前一年 11 月～當年 3 月")
    ap.add_argument("--radius-m", type=float, default=750.0, help="熱點緩衝區半徑；預設 750 m ＝ Sentinel Hub 1.5 km 見方框的半邊長，使 NBR 與 SSIM 看同一塊範圍（試跑 300 m 時抓不到框內其他坡面的崩塌）")
    ap.add_argument("--min-images", type=int, default=MIN_IMAGES, help="任一期 composite 至少需要的影像張數")
    ap.add_argument("--thresh", type=float, default=0.2, help="dNBR 門檻（>thresh 視為植生減少）；預設 0.2 未經校正，請看輸出分布")
    ap.add_argument("--cs-thresh", type=float, default=0.6, help="Cloud Score+ cs_cdf 下限")
    ap.add_argument("--out", default=str(REPO / "data" / "gee_trial.json"))
    args = ap.parse_args()
    if not args.project:
        sys.exit("請以 --project 或環境變數 GEE_PROJECT 指定 Earth Engine Cloud project ID")
    y_lo, y_hi = (int(x) for x in args.years.split("-"))
    ranks = [int(x) for x in args.ranks.split(",")]

    ee = init_ee(args.project)
    hotspots = {h["rank"]: h for h in json.loads((DATA / "top100_consolidated.json").read_text(encoding="utf-8"))}
    results = []
    for rank in ranks:
        h = hotspots[rank]
        geom = ee.Geometry.Point([h["lon"], h["lat"]]).buffer(args.radius_m)
        ref = {b[:4]: f for _a, b, f in ssim_reference(rank)}      # 以後期年份對照 SSIM 配對
        print(f"\n#{rank} {h.get('county') or ''}{h.get('district') or ''}  SSIM 參考（後期年→變化比例）：{ref}")
        print(f"{'期間':<11}{'張數':>9}{'valid':>8}{'loss':>8}{'gain':>8}  備註")
        rows = []
        for y in range(y_lo + 1, y_hi + 1):
            try:
                s = pair_stats(ee, geom, y - 1, y, args.thresh, args.cs_thresh).getInfo()
            except Exception as e:  # noqa: BLE001
                print(f"{y - 1}→{y}  查詢失敗：{e}")
                continue
            valid = s.get("valid") or 0.0
            few = min(s["n_a"], s["n_b"]) < args.min_images
            insufficient = valid < MIN_VALID_FRAC or few
            row = {"year_a": y - 1, "year_b": y, "n_images_a": s["n_a"], "n_images_b": s["n_b"],
                   "valid_frac": round(valid, 3),
                   "loss_frac": None if insufficient else round((s.get("loss") or 0) / max(valid, 1e-9), 4),
                   "gain_frac": None if insufficient else round((s.get("gain") or 0) / max(valid, 1e-9), 4),
                   "insufficient": insufficient, "insufficient_reason": ("few_images" if few else "low_valid_frac") if insufficient else None, "ssim_ref_by_year_b": ref.get(str(y))}
            rows.append(row)
            note = ("資料不足（影像張數少）" if few else "資料不足（有效像素少）") if insufficient else ""
            print(f"{y - 1}→{y:<6}{s['n_a']:>4}/{s['n_b']:<4}{valid:>8.2f}"
                  f"{'—' if insufficient else format(row['loss_frac'], '.3f'):>8}"
                  f"{'—' if insufficient else format(row['gain_frac'], '.3f'):>8}  {note}")
        results.append({"rank": rank, "lat": h["lat"], "lon": h["lon"], "pairs": rows})

    out = {"method": "dry-season median composite (Nov–Mar) + Cloud Score+ mask + dNBR",
           "params": {"radius_m": args.radius_m, "dnbr_thresh": args.thresh, "cs_thresh": args.cs_thresh,
                      "min_valid_frac": MIN_VALID_FRAC, "min_images": args.min_images, "s2": S2, "cloud_score": CS_PLUS},
           "note": "試跑結果，全部為候選訊號；loss_frac 在 valid_frac 不足時為 null。",
           "results": results}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
