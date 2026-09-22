# -*- coding: utf-8 -*-
"""
Top 100 熱點的 Sentinel-2 影像覆驗（2026-09-22 起取代 GE 快篩，作為變遷證據的主要來源）。

取像：Copernicus Data Space Sentinel Hub WMTS 512×512 圖磚（PopularWebMercator512、tilematrix 14，
約 9.55 m/px），沿用 webapp/change_detect_viewer/sentinel_assist.py 的拼接、浮水印處理與雲量檢查。

每個熱點：
  1. 範圍：以熱點為中心 ±HALF_M（預設 750 m，即 1.5 km 見方，約 157×157 原生像素）。
  2. 日期：每年一期（2017 起）。為了避免季節差異（植生、水稻）被誤判成地貌變化，每年查詢乾季窗口
     （前一年 11/1 – 當年 3/31）雲量最低的場景；該窗口沒有可用場景才退回同年 4–10 月。
     每年最多試 3 個候選，逐張做 AOI 局部雲、無資料檢查。
  3. 全部期別湊齊後，再用「與中位數影像的偏離度」剔除薄雲/霧/雲影離群期別。
  4. 相鄰年份逐對跑 ge_change_detect.detect_change()（與 GE 同一套 SSIM 引擎），
     寫出 _change_detect/<site>_change_timeline.json 與逐對 diff JSON／面板。
  5. change_score = 相鄰年份配對中最大的 overall_change_fraction（與巡查優先級讀取的
     「最大變遷配對」同一定義）。寫回 top100_consolidated.json。

限制（照實揭露）：10 m 解析度只看得到面積級變化（崩塌、裸露、河道、大面積開發）；
每張圖磚左下角的浮水印區改用上層圖磚（19–38 m/px）補回（sentinel_assist._fill_watermark），
該處解析度較粗但各期一致；年度單期無法捕捉年內短期變化。

用法：
  python scripts/s2_verify_hotspots.py              # 從頭跑（會先清掉舊的 ardswc_top* 站點）
  python scripts/s2_verify_hotspots.py --resume     # 跳過已完成的站點
"""
import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "webapp" / "change_detect_viewer"))
import ge_change_detect as CD  # noqa: E402
import sentinel_assist as S2  # noqa: E402

DATA = REPO / "data" / "ardswc_hotspots"
CAPTURES = REPO / "data" / "ge_captures"
HALF_M = 750.0
START_YEAR = 2017
MAX_SCENE_CLOUD = 30.0
MAX_CLOUD_LOCAL = 0.15
MAX_OUTLIER = 0.25
TRIES_PER_YEAR = 3
SSIM_THRESH = 0.45


def _search(lat, lon, d0, d1, log, tag):
    """STAC 查詢＋退避重試（目錄服務偶爾很慢，單一乾季窗口也可能要 10 秒以上）。"""
    for attempt in range(4):
        try:
            return S2.search_scenes(lat, lon, d0, d1, max_cloud=MAX_SCENE_CLOUD, half_m=HALF_M)
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            log(f"  {tag} STAC 查詢失敗（{e}），{15 * (attempt + 1)} 秒後重試")
            time.sleep(15 * (attempt + 1))


def year_candidates(lat, lon, log, tag):
    """逐年查詢乾季窗口（前一年 11/1 – 當年 3/31），每年取雲量最低的前 N 個場景；該窗口沒有可用場景時
    才退回同年 4–10 月。逐窗口小查詢取代一次查十年（後者在 STAC 服務慢時整批逾時）。
    回傳 {year: [scene...]}，year 為乾季所屬的「結束年」。"""
    out, n_scenes = {}, 0
    for y in range(START_YEAR, date.today().year + 1):
        end = min(date(y, 3, 31), date.today())
        scs = _search(lat, lon, date(y - 1, 11, 1), end, log, tag)
        if not scs and date(y, 4, 1) <= date.today():
            scs = _search(lat, lon, date(y, 4, 1), min(date(y, 10, 31), date.today()), log, tag)
        n_scenes += len(scs)
        if scs:
            out[y] = sorted(scs, key=lambda sc: sc["cloud"])[:TRIES_PER_YEAR]
    return out, n_scenes


def process(h, log):
    site = f"ardswc_top{h['rank']:02d}"
    site_dir = CAPTURES / site
    lat, lon = h["lat"], h["lon"]
    cands_by_year, n_scenes = year_candidates(lat, lon, log, f"#{h['rank']}")
    frames = []
    for y, cands in cands_by_year.items():
        for sc in cands:
            try:
                img, bounds = S2.fetch_image(lat, lon, HALF_M, sc["date"], fill_watermark=True)
            except Exception as e:  # noqa: BLE001
                log(f"  #{h['rank']} {sc['date']} 取像失敗：{e}")
                continue
            if S2.blank_fraction(img) > 0.05 or S2.cloud_fraction(img) > MAX_CLOUD_LOCAL:
                continue
            frames.append((sc["date"], img, bounds, sc["cloud"]))
            break
    scores = S2.outlier_scores([f[1] for f in frames])
    good = [f for s, f in zip(scores, frames) if s <= MAX_OUTLIER]
    dropped = [f[0] for s, f in zip(scores, frames) if s > MAX_OUTLIER]

    if site_dir.exists():
        shutil.rmtree(site_dir)
    site_dir.mkdir(parents=True)
    for ymd, img, bounds, _c in good:
        S2.save_frame(site_dir, site, ymd, img, bounds)
    summary = []
    if len(good) >= 2:
        dated = CD._list_dated(site_dir)
        out_dir = site_dir / "_change_detect"
        for i in range(len(dated) - 1):
            (da, pa), (db, pb) = dated[i], dated[i + 1]
            r = CD.detect_change(pa, pb, da, db, out_dir, site, ui_top=0, ui_bottom=0, ui_top_auto=False,
                                 ssim_thresh=SSIM_THRESH, min_region_px=48)
            summary.append({"date_a": da, "date_b": db, "overall_change_fraction": r["overall_change_fraction"],
                            "mean_ssim": r["mean_ssim"], "n_regions": r["n_regions"]})
        (out_dir / f"{site}_change_timeline.json").write_text(json.dumps({
            "site": site, "source": "sentinel-2", "resolution_m": S2.NATIVE_M_PER_PX, "half_m": HALF_M,
            "date_rule": "每年一期：乾季窗口（前一年 11/1–當年 3/31）雲量最低場景，無則同年 4–10 月", "outliers_dropped": dropped,
            "pairs": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    score = max((p["overall_change_fraction"] for p in summary), default=None)
    return h["rank"], n_scenes, [f[0] for f in good], dropped, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    if not S2.is_configured():
        raise SystemExit("未設定 Sentinel Hub Instance ID（SENTINEL_INSTANCE_ID 或 .sentinel_instance_id）")

    hotspots = json.loads((DATA / "top100_consolidated.json").read_text(encoding="utf-8"))
    if not args.resume:
        for d in CAPTURES.glob("ardswc_top*"):
            shutil.rmtree(d)
    todo = [h for h in hotspots if not (args.resume and
            (CAPTURES / f"ardswc_top{h['rank']:02d}" / "_change_detect" /
             f"ardswc_top{h['rank']:02d}_change_timeline.json").exists())]
    results_fp = DATA / "s2_verify_results.json"
    results = json.loads(results_fp.read_text(encoding="utf-8")) if (args.resume and results_fp.exists()) else {}
    log = lambda m: print(m, flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(process, h, log): h for h in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            h = futs[fut]
            try:
                rank, n_sc, kept, dropped, score = fut.result()
                results[str(rank)] = {"scenes": n_sc, "dates": kept, "outliers_dropped": dropped, "change_score": score}
                log(f"[{i}/{len(todo)}] #{rank} 場景 {n_sc}、保留 {len(kept)} 期 {kept[:1]}…{kept[-1:]}、"
                    f"離群剔除 {len(dropped)}、change_score {score}  ({time.time() - t0:.0f}s)")
            except Exception as e:  # noqa: BLE001
                results[str(h["rank"])] = {"error": f"{type(e).__name__}: {e}"}
                log(f"[{i}/{len(todo)}] #{h['rank']} 失敗：{e}")
            results_fp.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    for h in hotspots:
        r = results.get(str(h["rank"]), {})
        h["change_score"] = None if r.get("change_score") is None else round(r["change_score"], 4)
        h["method"] = "sentinel2"
    (DATA / "top100_consolidated.json").write_text(json.dumps(hotspots, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = sum(1 for h in hotspots if h["change_score"] is not None)
    log(f"完成：{ok}/{len(hotspots)} 個熱點有 Sentinel-2 變遷分數（{time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
