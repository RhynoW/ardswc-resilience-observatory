# -*- coding: utf-8 -*-
"""
ge_web_capture_v2.py — 依「中心經緯度 + 目標解析度」擷取 Google Earth Web 最近 N 個歷史日期，
每張輸出「PNG + .jgw world file（含地理座標）」，檔名帶影像日期。

與 v1（ge_web_capture.py，走遍所有日期 / 大量發現）的差異：v2 針對「一個點、看最近幾期」的
輕量需求——指定中心座標與解析度（預設 0.3 m/px），只取**最近 N 個日期**（預設 3），並直接
georef 成帶 .jgw 的截圖，供丟進 QGIS/port_harvest/naval 流程。

流程（重用既有、驗證過的元件）：
  1. 解析度 → 相機距離：GSD≈K·dist（K=0.000444，ge-mosaic 校準）→ dist = gsd / K。
  2. ge_web_capture 的瀏覽器函式：啟用 Flutter a11y semantics、進歷史模式、Ctrl+B→全部隱藏標註。
     （launch 已啟用 chromium 沙箱 → 無 --no-sandbox 警告橫幅；見 v1 同步修正。）
  3. 走到**最新**日期 → 截圖 → 逐步「較舊」擷取，共取最近 N 個日期，存成 <site>_gmap_YYYYMMDD.png。
  4. georef（**預設建構式**）：v2 自己指定了相機中心(lat,lon)、北正上方俯視、GSD≈K·dist，故直接由
     中心+GSD 合成 world file（EPSG:3857），每張輸出 .jgw+.prj——免下載 GMaps、免 SIFT，一定有座標。
     選配 --sift：對特徵豐富的陸地場景額外用 ge_georef 對 GMaps SIFT 精修，通過者覆蓋建構式（fail-closed）。

治理（§2/§14 fail-closed）：擷取/座標皆為自動化幾何，非艦種/機型 ground truth；georef 品質未達標
不靜默輸出錯座標。艦種/機型定型仍由人（§14.1）。

用法：
  python scripts/ge_web_capture_v2.py --lat 21.23615039 --lon 110.43696544 --site zhanjiang_north
  python scripts/ge_web_capture_v2.py --lat 31.3908 --lon 118.4094 --gsd 0.25 --n-dates 5 --site wuhu
執行環境：conda Falcon9_Baseline_Sim（playwright + cv2）。
"""
import argparse
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent  # 可攜路徑（2026-09-05 修正，原硬編碼 F:\ 在容器/HF Space 上不存在）
sys.path.insert(0, str(REPO / "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import ge_web_capture as G                       # _enable_a11y/_enter_historical/_hide_annotations/_read_stepper/_wait_tiles/build_url

K_GSD_PER_DIST = 0.000444                          # GSD(m/px) ≈ K · dist_d（ge-mosaic 3 點校準）
CALIB_VH = 1440                                     # K 的校準 viewport 高（ge-mosaic 預設 2560×1440）
# GE 為透視相機、固定垂直 FOV：同一 dist 下 footprint 固定，viewport 越高 → 每像素 GSD 越細。
# 故實際 GSD = 目標 gsd × (CALIB_VH / vh)：vh=1440→×1（v2 預設，等於目標）、vh=4320（8K）→×1/3。
# 實測 SIFT：8K vs 2560 同景 scale=3.00，若不修正 8K 量船長會是真值 3 倍。
R_MERC = 6378137.0
PRJ_3857 = ('PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",'
            'SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],'
            'UNIT["degree",0.0174532925199433]],PROJECTION["Mercator_1SP"],'
            'PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],'
            'PARAMETER["false_easting",0],PARAMETER["false_northing",0],'
            'UNIT["metre",1],AUTHORITY["EPSG","3857"]]')


def _write_construction_jgw(png, lat, lon, gsd, vw, vh):
    """建構式 world file：v2 自己指定了相機中心(lat,lon)、北正上方俯視、GSD≈K·dist，
    故可直接由中心+GSD 推 world file（EPSG:3857），免 SIFT。作為 SIFT 對位失敗時的 fallback。
    寫 .jgw + .prj + <png>.georef_construction.json（標記為近似，非 SIFT 量測）。"""
    png = Path(png)
    merc_gsd = gsd / max(0.1, math.cos(math.radians(lat)))          # 地面 m/px → 麥卡托 m/px
    mx = R_MERC * math.radians(lon)
    my = R_MERC * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    A, E = merc_gsd, -merc_gsd                                      # 北正上方：右=+X、下=-Y
    C = mx - A * (vw / 2.0)                                         # 左上角像素中心的麥卡托座標
    F = my - E * (vh / 2.0)
    png.with_suffix(".jgw").write_text(
        f"{A:.10f}\n0.0000000000\n0.0000000000\n{E:.10f}\n{C:.6f}\n{F:.6f}\n", encoding="utf-8")
    png.with_suffix(".prj").write_text(PRJ_3857, encoding="utf-8")
    import json as _json
    (png.parent / (png.name + ".georef_construction.json")).write_text(_json.dumps(
        {"method": "construction", "note": "由 v2 指定之相機中心+GSD 推得（非 SIFT 量測，近似）",
         "center_lat": lat, "center_lon": lon, "gsd_m_per_px": gsd, "merc_gsd_m_per_px": round(merc_gsd, 6),
         "vw": vw, "vh": vh}, ensure_ascii=False, indent=1), encoding="utf-8")
    return png.with_suffix(".jgw")


def terrain_elev_m(lat, lon):
    """查地形海拔（公尺）供 GE 相機 look-at 高度用。GE 載 3D 地形，若 look-at 高度設 0（海平面）
    而地表在高原（如西藏 ~4000m），相機(0+dist)會鑽到地表以下 → 截到一片黑。設 look-at=地表海拔即修正。
    用 open-meteo 免金鑰高程 API；失敗回 None（呼叫端退回 0，維持原行為）。"""
    import urllib.request, json as _json
    try:
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lat:.5f}&longitude={lon:.5f}"
        r = _json.load(urllib.request.urlopen(url, timeout=15))
        e = r.get("elevation")
        return float(e[0]) if isinstance(e, list) and e else (float(e) if e is not None else None)
    except Exception:
        return None


def _capture(page, path):
    page.screenshot(path=str(path), timeout=60000)  # 預設 30s 逾時在無 GPU 容器偶爾不夠（見 --headless-container）
    return path.stat().st_size


def _goto_newest(page, max_steps=90):
    """一路點『較新的圖片』走到最新日期（date 不再變即到頂）。回傳最新 date 'YYYY-MM-DD'。"""
    last = None
    for _ in range(max_steps):
        ds, nxt, prv = G._read_stepper(page)
        if not ds or not nxt or ds == last:
            break
        last = ds
        page.mouse.click(*nxt)
        time.sleep(2.0)
    ds, _, _ = G._read_stepper(page)
    return ds


def bbox_from_center(lat, lon, gsd, vw, vh, pad=1.0):
    """中心座標 + 解析度 + viewport → 覆蓋 WGS84 bbox（供 georef GMaps 參考底圖）。"""
    half_w_m = (vw / 2) * gsd * pad
    half_h_m = (vh / 2) * gsd * pad
    dlat = half_h_m / 111320.0
    dlon = half_w_m / (111320.0 * max(0.1, math.cos(math.radians(lat))))
    return (lat - dlat, lon - dlon, lat + dlat, lon + dlon)


def georef_zoom_for(gsd, lat):
    """挑一個 GMaps zoom 使其 GSD 最接近目標 gsd（GMaps GSD=156543.03·cos(lat)/2^z）。"""
    z = math.log2(156543.03 * math.cos(math.radians(lat)) / max(0.05, gsd))
    return int(max(17, min(21, round(z))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--gsd", type=float, default=0.15, help="目標地面解析度 m/px（預設 0.15）→ 決定相機距離")
    ap.add_argument("--n-dates", type=int, default=3, help="擷取最近幾個歷史日期（預設 3，最新往舊）")
    ap.add_argument("--skip-recent", type=int, default=0,
                    help="先跳過最近 N 期不擷取，取更舊的日期窗（如 --skip-recent 5 --n-dates 5 → 第 6~10 新）；"
                         "若跳過途中已到最早日期，該站不擷取（rc=4）")
    ap.add_argument("--min-date", default="",
                    help="影像日期下限 YYYYMMDD；擷取時遇到早於此者即捨棄並停止該站（往舊走故其後皆更舊）")
    ap.add_argument("--site", required=True, help="站點名（輸出檔名/目錄）")
    ap.add_argument("--outdir", default=str(REPO / "data" / "ge_captures"))
    ap.add_argument("--vw", type=int, default=2560)
    ap.add_argument("--vh", type=int, default=1440)
    ap.add_argument("--window-pos", default="0,0")
    ap.add_argument("--wait", type=float, default=6.0, help="每個日期 tiles 載入等待秒數")
    ap.add_argument("--first-wait", type=float, default=24.0, help="首張額外等待（GE 初次載入較久）")
    ap.add_argument("--seed-date", default="20250101", help="起始注入日期（進歷史模式用；GE 會吸附最近可用日期）")
    ap.add_argument("--sift", action="store_true",
                    help="額外用 SIFT 對 GMaps 精修 world file（預設關；預設走建構式：由相機中心+GSD 直接推）")
    ap.add_argument("--no-georef", action="store_true", help="只擷取、不產 .jgw")
    ap.add_argument("--profile", default=str(REPO / "data" / "ge_captures" / "_chrome_profile"))
    ap.add_argument("--cdp", default="",
                    help="連到既有 Chrome 的 CDP 端點（如 http://127.0.0.1:9222）→ 用遠端/另一台的瀏覽器擷取；"
                         "留空＝本機自啟 headful Chrome。遠端瀏覽器需自帶顯示+WebGL（GE 才渲染得出來）。")
    ap.add_argument("--headless-container", action="store_true",
                    help="容器/無顯示環境用（2026-09-05 追加，HF Space 等）：headless=True + Playwright 內建"
                         "Chromium（非 channel=chrome）+ 一次性 context（非 --profile 持久化，無沙箱需求）。"
                         "本機開發預設不開，維持既有 headed+真 Chrome+持久化 profile 行為不變。已實測 HF "
                         "cpu-basic（2 vCPU/無 GPU，software WebGL）可正確載入歷史影像+走日期 stepper；"
                         "viewport 建議調小（如 1920×1080）——實測 2560×1440 在該硬體上偶發 screenshot "
                         "逾時/日期讀取錯位，1920×1080 未見此問題。")
    ap.add_argument("--elev", default="0",
                    help="相機 look-at 海拔(公尺)：'0'(預設,原行為)、'auto'(查地形高程，避免高原鑽地面下截黑)、或指定數值")
    args = ap.parse_args()

    dist = max(1, round(args.gsd / K_GSD_PER_DIST))     # dist 由目標 gsd 決定「取景」（footprint）
    actual_gsd = args.gsd * CALIB_VH / args.vh          # 實際每像素 GSD（隨 viewport 高調整；見 CALIB_VH 說明）
    outdir = Path(args.outdir) / args.site
    outdir.mkdir(parents=True, exist_ok=True)
    if str(args.elev).lower() == "auto":                # 地形感知：look-at 高度=地表海拔，避免高原相機鑽地下截黑
        elev = terrain_elev_m(args.lat, args.lon)
        if elev is None:
            elev = 0.0
            print("[v2] 高程查詢失敗 → look-at 海拔退回 0（高原地區可能截黑，可改 --elev <公尺>）", flush=True)
        else:
            print(f"[v2] 地形高程 {elev:.0f} m（look-at 海拔=此值，相機在其上方 {dist} m）", flush=True)
    else:
        try:
            elev = float(args.elev)
        except ValueError:
            elev = 0.0
    cam = f"{args.lat:.8f},{args.lon:.8f},{elev:.2f}a,{dist}d,35y,0h,0t,0r"
    start_url = G.build_url(f"https://earth.google.com/web/@{cam}", args.seed_date)
    ag_note = "" if abs(actual_gsd - args.gsd) < 1e-6 else f"；viewport {args.vw}×{args.vh} → 實際 GSD≈{actual_gsd:.4f} m/px"
    print(f"[v2] 中心 {args.lat},{args.lon}  gsd={args.gsd} m/px → dist={dist}d  最近 {args.n_dates} 期{ag_note}", flush=True)
    print(f"[v2] URL: {start_url}", flush=True)

    captured = []
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        remote = bool(args.cdp)
        browser = None
        if remote:
            # 遠端瀏覽器方案：連到既有 Chrome（本機或另一台 --remote-debugging-port）→ 用它的分頁擷取。
            # 讓 Flask 端可在無頭/容器/另一台主機，瀏覽器渲染工作外包給有顯示+WebGL 的瀏覽器主機。
            print(f"[v2] 連線遠端瀏覽器 CDP：{args.cdp}", flush=True)
            browser = pw.chromium.connect_over_cdp(args.cdp)
            ctx = browser.contexts[0] if browser.contexts else browser.new_context(
                viewport={"width": args.vw, "height": args.vh})
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            try:                                        # 既有 context 的分頁無法改 viewport（視窗大小已定）→ 盡力而為
                page.set_viewport_size({"width": args.vw, "height": args.vh})
            except Exception:
                print("    （遠端既有視窗，viewport 沿用瀏覽器視窗大小；建議啟動時帶 --window-size）", flush=True)
        elif args.headless_container:
            # 一次性 launch()（非 launch_persistent_context）：容器內無真實 Chrome 安裝、多半也無
            # 沙箱權限（seccomp 限制），headless bundled Chromium 才是唯一可靠選項——已在 HF Space
            # 實測驗證（見上方 --headless-container 說明）。不用 --profile 持久化，每次全新 context
            # 即可（單次擷取任務用完即丟，不需要跨呼叫保留 cookie/consent 狀態）。
            browser = pw.chromium.launch(headless=True)
            ctx = browser.new_context(viewport={"width": args.vw, "height": args.vh})
            page = ctx.new_page()
        else:
            ctx = pw.chromium.launch_persistent_context(
                args.profile, channel="chrome", headless=False,
                viewport={"width": args.vw, "height": args.vh}, device_scale_factor=1.0,
                chromium_sandbox=True,                      # 無 --no-sandbox 警告橫幅
                ignore_default_args=["--enable-automation"],
                args=[f"--window-position={args.window_pos}", "--disable-features=Translate",
                      "--no-first-run", "--no-default-browser-check", "--disable-infobars"])
            page = ctx.pages[0] if ctx.pages else ctx.new_page()

        def _teardown():
            try:
                if remote:
                    browser.close()      # 只斷開 CDP 連線，不殺遠端瀏覽器（連到別人的 Chrome 時尤其重要）
                elif args.headless_container:
                    browser.close()      # launch() 的一次性 browser+context 一併關閉
                else:
                    ctx.close()
            except Exception:
                pass

        print("[v2] ① 開啟 + 定位/Zoom…", flush=True)
        try:
            page.goto(start_url, wait_until="load", timeout=90000)
        except Exception as e:
            print(f"    goto warn: {type(e).__name__}", flush=True)
        page.bring_to_front(); time.sleep(6 + args.first_wait)
        try:                                            # 拖曳觸發 tile 串流
            cx, cy = args.vw // 2, args.vh // 2
            for dx, dy in [(240, 140), (-240, -140)]:
                page.mouse.move(cx, cy); page.mouse.down(); page.mouse.move(cx + dx, cy + dy, steps=10); page.mouse.up(); time.sleep(2.0)
        except Exception:
            pass

        print("[v2] ② 啟用 a11y…", flush=True)
        G._enable_a11y(page); time.sleep(4)
        ds, nxt, prv = G._read_stepper(page)
        if ds is None and G._enter_historical(page):    # 未在歷史模式 → 點『顯示歷史圖像』
            time.sleep(6); ds, nxt, prv = G._read_stepper(page)
        if ds is None:
            print("[v2] ✘ 讀不到歷史日期列（該點可能無 GE 歷史影像）。", flush=True)
            _teardown(); sys.exit(2)

        print("[v2] ③ 關閉標註（Ctrl+B→全部隱藏）…", flush=True)
        G._hide_annotations(page); time.sleep(1.5)

        newest = _goto_newest(page)
        # ④a 先跳過最近 skip_recent 期（不擷取），取更舊的日期窗（如 #6~10）
        skip = max(0, args.skip_recent)
        if skip:
            print(f"[v2] ④a 跳過最近 {skip} 期（取更舊日期窗）…", flush=True)
            last_s = None
            for j in range(skip):
                ds, nxt, prv = G._read_stepper(page)
                if not ds or ds == last_s or not prv:
                    print(f"    第 {j+1}/{skip} 步已無更舊日期（{ds}）→ 該站舊日期不足，不擷取。", flush=True)
                    _teardown(); sys.exit(4)
                last_s = ds
                page.mouse.click(*prv); time.sleep(2.5)
        print(f"[v2] ④ 最新日期 = {newest}；擷取"
              + (f"第 {skip+1}~{skip+args.n_dates} 新" if skip else f"最近 {args.n_dates}")
              + f" 期（最新→較舊）"
              + (f"，日期下限 {args.min_date}" if args.min_date else "") + "…", flush=True)
        last = None
        for i in range(args.n_dates):
            ds, nxt, prv = G._read_stepper(page)
            if not ds or ds == last:
                print(f"    已到最早（{ds}），停止於 {len(captured)} 期。", flush=True); break
            G._wait_tiles(page, args.wait)
            ds2, nxt, prv = G._read_stepper(page); ds2 = ds2 or ds
            ymd = ds2.replace("-", "")
            if args.min_date and ymd < args.min_date:
                print(f"    日期 {ds2} 早於下限 {args.min_date} → 捨棄並停止該站（其後更舊）。", flush=True); break
            png = outdir / f"{args.site}_gmap_{ymd}.png"
            sz = _capture(page, png)
            print(f"    ✔ [{i+1}/{args.n_dates}] {ds2} → {png.name} ({sz:,}B)", flush=True)
            captured.append(ymd); last = ds2
            if i < args.n_dates - 1:
                if not prv:
                    print("    無更舊日期，停止。", flush=True); break
                page.mouse.click(*prv); time.sleep(2.5)
        _teardown()

    if not captured:
        if args.min_date or args.skip_recent:
            print("[v2] ⊘ 指定日期窗內無合格日期（skip-recent/min-date）→ 跳過該站。", flush=True); sys.exit(4)
        print("[v2] ✘ 未擷取任何日期。", flush=True); sys.exit(3)

    if args.no_georef:
        print(f"[v2] 完成（未 georef）：{len(captured)} 張 → {outdir}", flush=True)
        return

    # ── georef ──
    # 預設（建構式）：v2 自己指定了相機中心(lat,lon)、北正上方俯視、GSD≈K·dist，故可直接由中心+GSD
    #   合成 world file，免下載 GMaps、免 SIFT——一定有座標、最快、對水域/特徵少場景同樣可用。
    # --sift（選配）：對特徵豐富的陸地場景想要影像級精度時，額外對 GMaps 做 SIFT 精修，通過者覆蓋掉建構式。
    if args.sift:
        print("[v2] ⑤ georef：--sift → 先建構式打底，再 SIFT 精修（通過者覆蓋）…", flush=True)
        for ymd in captured:                        # 先每張建構式打底（保證都有 .jgw；用實際 GSD）
            _write_construction_jgw(outdir / f"{args.site}_gmap_{ymd}.png",
                                    args.lat, args.lon, actual_gsd, args.vw, args.vh)
        bbox = bbox_from_center(args.lat, args.lon, actual_gsd, args.vw, args.vh)
        zoom = georef_zoom_for(actual_gsd, args.lat)
        print(f"    參考 bbox={tuple(round(v,6) for v in bbox)}  zoom={zoom}", flush=True)
        try:
            import ge_georef
            # SIFT 通過會覆寫 .jgw/.prj 並移除該張的建構式標記；失敗則保留建構式（fail-closed）
            ge_georef.georef_capture_dir(str(outdir), bbox=bbox, zoom=zoom)
            for ymd in captured:                    # SIFT 產出 .jpg 者＝已精修 → 移除建構式標記
                if (outdir / f"{args.site}_gmap_{ymd}.jpg").exists():
                    (outdir / f"{args.site}_gmap_{ymd}.png.georef_construction.json").unlink(missing_ok=True)
        except SystemExit:
            print("[v2]    SIFT 對位未達標（fail-closed）→ 維持建構式 world file。", flush=True)
        except Exception as e:
            print(f"[v2]    SIFT 例外（{type(e).__name__}: {e}）→ 維持建構式 world file。", flush=True)
    else:
        print(f"[v2] ⑤ georef（預設建構式）：由相機中心+實際GSD({actual_gsd:.4f}) 直接推 world file，每張產 .jgw …", flush=True)
        for ymd in captured:
            _write_construction_jgw(outdir / f"{args.site}_gmap_{ymd}.png",
                                    args.lat, args.lon, actual_gsd, args.vw, args.vh)

    n_con = sum(1 for ymd in captured
                if (outdir / f"{args.site}_gmap_{ymd}.png.georef_construction.json").exists())
    n_sift = len(captured) - n_con
    print(f"\n[v2] === 完成 ===  擷取 {len(captured)} 期；.jgw：建構式 {n_con} 張" +
          (f"、SIFT 精修 {n_sift} 張" if args.sift else "（預設建構式）"), flush=True)
    for ymd in captured:
        con = outdir / f"{args.site}_gmap_{ymd}.png.georef_construction.json"
        tag = "+.jgw(建構式)" if con.exists() else "+.jgw(SIFT 精修)"
        print(f"  {ymd}  {args.site}_gmap_{ymd}.png  {tag}", flush=True)
    print(f"[v2] → {outdir}", flush=True)


if __name__ == "__main__":
    main()
