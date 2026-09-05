# -*- coding: utf-8 -*-
"""
ge_change_detect.py — 對同一站點跨歷史日期的 Google Earth Web 截圖，做像素級變遷偵測。

背景（CLAUDE.md「高解析度衛星影像前後期變遷偵測」，2026-09-03 起）：目前唯一有跨日期時間軸的
影像來源是 GE Web（ge-historical-capture / ge-web-capture-v2 系列擷取）；Google Maps/Bing/
ESRI/Apple 這些 slippy-map 圖磚來源只有「當前一張」，沒有歷史日期（見 tile_mosaic.py 開頭
說明）。同站點所有日期共用同一 world file（相機視野/地理 footprint 相同，見 ge_georef.py、
ge_temporal_animate.py），各日期截圖天生像素對位一致，不需要重新 registration。

本工具在此前提上，補上「自動找出哪裡變了」這一步（先前只能靠人眼看 ge-temporal-animation
動畫逐格比對）：
  1. 讀入兩張（或 --all-pairs 逐一相鄰日期）截圖，扣掉 GE UI chrome 帶（頂部工具列/搜尋列、
     底部時間軸/© Google 浮水印——同 naval_grayscale_profile.py 的 ui_top/ui_bottom 慣例）。
     不扣的話，日期文字與時間軸滑塊在不同截圖間必然不同，會製造滿版假陽性。
  2. 平移對位安全網（低解析度 ECC 估位移、縮放回全解析度套用；位移超過 --max-shift-px 視為
     不可靠，不套用、只標記 alignment.uncertain——不盲目 warp 掩蓋真實地物變化）。
  3. skimage structural_similarity 產生逐像素結構差異圖 → 門檻化 → 連通元件 → 篩最小面積 →
     依面積由大到小取前 N 個候選變遷區塊。
  4. 每個候選區塊輸出像素 bbox/centroid；若站點目錄有 .jgw（ge_georef.py 產出），一併換算
     lon/lat（供人工用 GE Web 精準回溯）。
  5. 輸出：四聯圖（日期A｜日期B｜候選框疊圖｜SSIM 差異熱區）PNG + 結構化 JSON。

治理（比照 CLAUDE.md §2 fail-closed 精神）：輸出的變遷區塊是**自動候選、非已驗證事實**——
雲影、隨季節變化的日照角度陰影、影像壓縮雜訊、GE 算圖 LOD 差異都會製造假陽性，人工複核前
不可當定論。門檻一律走 CLI/env（§12），不寫死。兩張影像長寬比不同時 fail-closed 直接拒絕
比對（不盲目 resize 製造假對位）；同 footprint 不同取樣密度（例如一張 8K 一張一般 viewport）
才允許縮到較小者再比對。

用法：
  # 指定兩個日期
  python scripts/ge_change_detect.py --capture-dir data/ge_captures/hetian_flanker_test \
      --date-a 20221112 --date-b 20250101

  # 不給日期 → 預設用最舊 vs 最新
  python scripts/ge_change_detect.py --capture-dir data/ge_captures/<site>

  # 逐一相鄰日期兩兩比對，產出完整變遷時間軸摘要
  python scripts/ge_change_detect.py --capture-dir data/ge_captures/<site> --all-pairs
"""
import argparse
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_DATE_RE = re.compile(r"_gmap_(\d{8})\.png$")
R_MERC = 6378137.0

DEFAULT_UI_TOP = int(os.getenv("GE_CHANGE_UI_TOP", "118"))
DEFAULT_UI_BOTTOM = int(os.getenv("GE_CHANGE_UI_BOTTOM", "44"))
DEFAULT_SSIM_THRESH = float(os.getenv("GE_CHANGE_SSIM_THRESH", "0.35"))
DEFAULT_MIN_REGION_PX = int(os.getenv("GE_CHANGE_MIN_REGION_PX", "80"))
DEFAULT_MAX_SHIFT_PX = int(os.getenv("GE_CHANGE_MAX_SHIFT_PX", "15"))
DEFAULT_TOP_N = int(os.getenv("GE_CHANGE_TOP_N", "30"))
DEFAULT_WIN_SIZE = int(os.getenv("GE_CHANGE_SSIM_WIN", "11"))


def _list_dated(capture_dir):
    out = []
    for p in sorted(glob.glob(str(Path(capture_dir) / "*_gmap_*.png"))):
        m = _DATE_RE.search(p)
        if m:
            out.append((m.group(1), p))
    out.sort(key=lambda t: t[0])
    return out


def _read_jgw(png):
    """同 naval_grayscale_profile.py 的 .jgw 讀法：A,D,B,E,C,F（EPSG:3857 世界檔六參數）。"""
    jgw = Path(png).with_suffix(".jgw")
    if not jgw.exists():
        return None
    try:
        vals = [float(x) for x in jgw.read_text().split()[:6]]
        return vals if len(vals) == 6 else None
    except Exception:
        return None


def px_to_lonlat(jgw_params, px, py):
    if jgw_params is None:
        return None
    A, D, B, E, C, F = jgw_params
    mx = A * px + B * py + C
    my = D * px + E * py + F
    lon = math.degrees(mx / R_MERC)
    lat = math.degrees(2 * math.atan(math.exp(my / R_MERC)) - math.pi / 2)
    if abs(lat) > 85 or abs(lon) > 180:
        return None
    return [round(lon, 7), round(lat, 7)]


def _font(size):
    for p in (r"C:\Windows\Fonts\msjhbd.ttc", r"C:\Windows\Fonts\msjh.ttc",
              r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\simhei.ttf",
              r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _auto_ui_top(gray, ui_top_hint, max_frac=0.08, bright_thresh=200, sustain=10, pad=10):
    """偵測 GE Web 頂部工具列/搜尋列/歷史日期列的實際高度——這個高度隨截圖解析度/DPI 縮放，
    不是固定像素（實測：naval_grayscale_profile.py 的 ui_top=118 是對其他解析度校準的，套到
    ge-web-capture-v2-8k 的 4320px 高截圖會少裁掉整條「歷史圖像 + 日期文字」列，而日期文字
    兩期必然不同，會製造滿版假陽性——見 CLAUDE.md 本輪變遷偵測章節)。
    原理：GE chrome 背景近乎全白/極淺灰，逐列平均亮度顯著高於下方衛星影像；找連續 sustain 列
    平均亮度都低於 bright_thresh 的第一列，視為真正影像起點。找不到清楚斷崖（例如遇到雪地/
    沙漠等本身就很亮的地物）就 fail-safe 退回呼叫端提供的 ui_top_hint，不猜測、不冒進裁切。"""
    h, w = gray.shape
    max_rows = max(1, int(h * max_frac))
    x0, x1 = w // 8, w - w // 8
    means = gray[:max_rows, x0:x1].mean(axis=1)
    run = 0
    for y in range(max_rows):
        if means[y] < bright_thresh:
            run += 1
            if run >= sustain:
                return min(h - 1, y - sustain + 1 + pad)
        else:
            run = 0
    return ui_top_hint


def _water_mask(bgr):
    """粗略水域遮罩（HSV 色相落在藍-青區間 + 有一定飽和度）。刻意寬鬆、非精準水陸分類——唯一目的
    是抑制『兩期皆為水面』時浪紋/日照反光/潮色造成的動態紋理雜訊（實測：亞龍潛艇碼頭 2021-06-30
    水面有強烈對角波紋，與 2025-01-01 平靜水面對比，SSIM 幾乎整片水域判定「不同」，把
    overall_change 灌到 0.92，稀釋了真正的岸際變化訊號）。用 HSV 色相＋飽和度而非原始 BGR 亮度
    門檻——實測深色浪影水域（value~75-90）色相/飽和度仍穩定落在水的範圍，純亮度門檻會漏掉這些
    較暗的水面、讓抑制出現破洞。**刻意不遮蔽水陸交界**：一期是水、另一期是陸（填海造陸/新建碼頭
    延伸入海——正是本工具最該抓到的變遷類型）留給 SSIM 正常判定，不受此遮罩影響。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s = hsv[..., 0], hsv[..., 1]
    return (h >= 85) & (h <= 140) & (s >= 60)


def _align_translation(gray_a, gray_b, max_shift_px, max_side=1024):
    """低解析度 ECC 估平移、縮放回全解析度。回傳 (dx, dy, applied, uncertain)。
    位移超過 max_shift_px 視為不可靠（可能是真實地物變化混淆了 ECC），不套用。"""
    h, w = gray_a.shape
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        sa = cv2.resize(gray_a, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        sb = cv2.resize(gray_b, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        sa, sb = gray_a, gray_b
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    try:
        _, warp = cv2.findTransformECC(sa.astype(np.float32), sb.astype(np.float32), warp,
                                        cv2.MOTION_TRANSLATION, criteria)
        dx, dy = float(warp[0, 2]) / scale, float(warp[1, 2]) / scale
    except cv2.error:
        return 0.0, 0.0, False, True
    if math.hypot(dx, dy) > max_shift_px:
        return dx, dy, False, True
    return dx, dy, True, False


def detect_change(png_a, png_b, date_a, date_b, out_dir, site,
                   ui_top=DEFAULT_UI_TOP, ui_bottom=DEFAULT_UI_BOTTOM, ui_top_auto=True,
                   ssim_thresh=DEFAULT_SSIM_THRESH, min_region_px=DEFAULT_MIN_REGION_PX,
                   top_n=DEFAULT_TOP_N, win_size=DEFAULT_WIN_SIZE,
                   align=True, max_shift_px=DEFAULT_MAX_SHIFT_PX, water_suppress=True):
    im_a = cv2.imread(str(png_a))
    im_b = cv2.imread(str(png_b))
    if im_a is None or im_b is None:
        raise RuntimeError(f"讀圖失敗：{png_a} / {png_b}")

    ha, wa = im_a.shape[:2]
    hb, wb = im_b.shape[:2]
    if (ha, wa) != (hb, wb):
        ar_a, ar_b = wa / ha, wb / hb
        if abs(ar_a - ar_b) > 0.02:
            raise RuntimeError(
                f"兩張影像長寬比不同（{Path(png_a).name}={wa}x{ha} vs {Path(png_b).name}={wb}x{hb}），"
                "footprint 可能不同，fail-closed 拒絕比對")
        if (wa * ha) > (wb * hb):
            im_a = cv2.resize(im_a, (wb, hb), interpolation=cv2.INTER_AREA)
        else:
            im_b = cv2.resize(im_b, (wa, ha), interpolation=cv2.INTER_AREA)

    h, w = im_a.shape[:2]
    if ui_top_auto:
        gray_full_a = cv2.cvtColor(im_a, cv2.COLOR_BGR2GRAY)
        gray_full_b = cv2.cvtColor(im_b, cv2.COLOR_BGR2GRAY)
        # 取兩張裡較大的一個（較保守，寧可多裁不可少裁——少裁會讓日期文字漏進 ROI 製造假陽性）
        top = max(_auto_ui_top(gray_full_a, ui_top), _auto_ui_top(gray_full_b, ui_top))
    else:
        top = ui_top
    top = min(top, h // 3)
    bottom = min(ui_bottom, h // 3)
    roi_a = im_a[top:h - bottom] if bottom > 0 else im_a[top:]
    roi_b = im_b[top:h - bottom] if bottom > 0 else im_b[top:]

    gray_a = cv2.cvtColor(roi_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(roi_b, cv2.COLOR_BGR2GRAY)

    align_info = {"checked": False, "applied": False, "uncertain": False, "shift_px": [0.0, 0.0]}
    if align:
        align_info["checked"] = True
        dx, dy, applied, uncertain = _align_translation(gray_a, gray_b, max_shift_px)
        align_info.update(applied=applied, uncertain=uncertain, shift_px=[round(dx, 2), round(dy, 2)])
        if applied:
            trans = np.float32([[1, 0, dx], [0, 1, dy]])
            roi_b = cv2.warpAffine(roi_b, trans, (roi_b.shape[1], roi_b.shape[0]),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            gray_b = cv2.cvtColor(roi_b, cv2.COLOR_BGR2GRAY)

    win = win_size if win_size % 2 == 1 else win_size + 1
    score, diff = ssim(gray_a, gray_b, full=True, win_size=win, data_range=255)
    changed = np.clip(1.0 - diff, 0.0, 1.0)  # 0=完全相同, 1=完全不同

    # SSIM 用滑動窗卷積計算，陣列四邊會有 padding 造成的邊界假訊號（實測：hetian_flanker_test
    # 兩期真實截圖，未加此濾除前 top-30 候選 100% 全部貼在 ROI 頂邊 y=0，沒有一個是真的地物
    # 變化）。裁掉四邊各 border px（窗半徑 + 緩衝）再進門檻化，不可省略。
    border = win // 2 + 5
    changed[:border, :] = 0
    changed[-border:, :] = 0
    changed[:, :border] = 0
    changed[:, -border:] = 0

    both_water_frac = 0.0
    if water_suppress:
        both_water = _water_mask(roi_a) & _water_mask(roi_b)
        both_water_frac = float(both_water.mean())
        changed[both_water] = 0

    changed_u8 = (changed * 255).astype(np.uint8)

    mask = (changed > ssim_thresh).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    overall_change_fraction = float((mask > 0).mean())

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    jgw = _read_jgw(png_a) or _read_jgw(png_b)

    regions = []
    for i in range(1, n_labels):  # 0 = background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_region_px:
            continue
        x, y, bw, bh = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                         int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        region_mask = labels[y:y + bh, x:x + bw] == i
        mean_score = float(changed[y:y + bh, x:x + bw][region_mask].mean())
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        full_y = y + top
        full_cy = cy + top
        entry = {
            "bbox_px": [x, full_y, bw, bh],
            "area_px": area,
            "mean_change_score": round(mean_score, 4),
            "centroid_px": [round(cx, 1), round(full_cy, 1)],
        }
        if jgw is not None:
            entry["centroid_lonlat"] = px_to_lonlat(jgw, cx, full_cy)
            corners = [(x, full_y), (x + bw, full_y), (x + bw, full_y + bh), (x, full_y + bh)]
            entry["bbox_lonlat"] = [px_to_lonlat(jgw, cxp, cyp) for cxp, cyp in corners]
        else:
            entry["centroid_lonlat"] = None
            entry["bbox_lonlat"] = None
        regions.append(entry)

    regions.sort(key=lambda r: r["area_px"], reverse=True)
    regions = regions[:top_n]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{site}_diff_{date_a}_{date_b}"

    heat = cv2.applyColorMap(changed_u8, cv2.COLORMAP_INFERNO)
    overlay_b = roi_b.copy()
    for r in regions:
        x, y, bw, bh = r["bbox_px"]
        cv2.rectangle(overlay_b, (x, y - top), (x + bw, y - top + bh), (0, 255, 0), max(2, w // 1200))

    panel_w = 900
    scale = panel_w / w
    panel_h = int(roi_a.shape[0] * scale)

    def _panel(img_bgr, label):
        im = cv2.resize(img_bgr, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        d.rectangle([0, 0, panel_w, 38], fill=(0, 0, 0))
        d.text((8, 4), label, font=_font(28), fill=(255, 255, 255))
        return pil

    panels = [_panel(roi_a, date_a), _panel(roi_b, date_b),
              _panel(overlay_b, f"變遷候選 x{len(regions)}"), _panel(heat, "SSIM 差異熱區")]
    combo = Image.new("RGB", (panel_w * 4 + 30, panel_h + 40), (20, 20, 20))
    for i, p in enumerate(panels):
        combo.paste(p, (i * (panel_w + 10), 40))
    ImageDraw.Draw(combo).text(
        (8, 6), f"{site}  {date_a} -> {date_b}  overall_change={overall_change_fraction:.3f}",
        font=_font(24), fill=(255, 255, 255))
    combo.save(out_dir / f"{tag}.png")

    # 個別全解析度面板（供視覺化工具用——combo 只有 900px 寬縮圖，webapp 需要可縮放的原尺寸圖）
    cv2.imwrite(str(out_dir / f"{tag}_before.jpg"), roi_a, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(out_dir / f"{tag}_after.jpg"), roi_b, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(out_dir / f"{tag}_overlay.jpg"), overlay_b, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(out_dir / f"{tag}_heat.jpg"), heat, [cv2.IMWRITE_JPEG_QUALITY, 90])

    result = {
        "site": site, "date_a": date_a, "date_b": date_b,
        "image_a": str(png_a), "image_b": str(png_b),
        "roi": {"ui_top": top, "ui_bottom": bottom, "ui_top_auto": ui_top_auto,
                "width": w, "height": roi_a.shape[0]},
        "panels": {"before": f"{tag}_before.jpg", "after": f"{tag}_after.jpg",
                   "overlay": f"{tag}_overlay.jpg", "heat": f"{tag}_heat.jpg", "combo": f"{tag}.png"},
        "alignment": align_info,
        "water_suppress": {"applied": water_suppress, "both_water_fraction": round(both_water_frac, 4)},
        "ssim_thresh": ssim_thresh, "min_region_px": min_region_px, "win_size": win,
        "mean_ssim": round(float(score), 4),
        "overall_change_fraction": round(overall_change_fraction, 4),
        "n_regions": len(regions),
        "regions": regions,
        "governance_note": ("regions are automated candidate change areas requiring human visual "
                             "confirmation via GE Web (use centroid_lonlat); not verified change facts "
                             "-- clouds/shadows/seasonal light/compression can trigger false positives"),
    }
    with open(out_dir / f"{tag}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main():
    ap = argparse.ArgumentParser(description="GE Web 跨日期像素級變遷偵測")
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--date-a")
    ap.add_argument("--date-b")
    ap.add_argument("--all-pairs", action="store_true", help="逐一相鄰日期兩兩比對，產出完整變遷時間軸")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--ui-top", type=int, default=DEFAULT_UI_TOP,
                     help="頂部 chrome 帶像素高度的 fallback/上限提示；預設由 --ui-top-auto 自動偵測")
    ap.add_argument("--no-ui-top-auto", action="store_true",
                     help="停用自動偵測 GE chrome 帶高度，強制使用 --ui-top 給定的固定值")
    ap.add_argument("--ui-bottom", type=int, default=DEFAULT_UI_BOTTOM)
    ap.add_argument("--ssim-thresh", type=float, default=DEFAULT_SSIM_THRESH)
    ap.add_argument("--min-region-px", type=int, default=DEFAULT_MIN_REGION_PX)
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    ap.add_argument("--win-size", type=int, default=DEFAULT_WIN_SIZE)
    ap.add_argument("--max-shift-px", type=int, default=DEFAULT_MAX_SHIFT_PX)
    ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--no-water-suppress", action="store_true",
                     help="關閉水面動態紋理雜訊抑制（預設開啟；水陸交界變化不受此影響，見腳本內 _water_mask 說明）")
    args = ap.parse_args()

    capture_dir = Path(args.capture_dir)
    site = capture_dir.name
    dated = _list_dated(capture_dir)
    if len(dated) < 2:
        print(f"[error] {capture_dir} 下找到 {len(dated)} 個日期，至少需要 2 個", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else capture_dir / "_change_detect"

    if args.all_pairs:
        pairs = [(dated[i], dated[i + 1]) for i in range(len(dated) - 1)]
    elif args.date_a and args.date_b:
        by_date = dict(dated)
        if args.date_a not in by_date or args.date_b not in by_date:
            print(f"[error] 指定日期不在 {capture_dir} 內。可用日期：{[d for d, _ in dated]}", file=sys.stderr)
            sys.exit(1)
        a, b = sorted([args.date_a, args.date_b])
        pairs = [((a, by_date[a]), (b, by_date[b]))]
    else:
        pairs = [(dated[0], dated[-1])]  # 預設：最舊 vs 最新

    summary = []
    for (date_a, png_a), (date_b, png_b) in pairs:
        print(f"[change-detect] {site}: {date_a} -> {date_b}")
        result = detect_change(png_a, png_b, date_a, date_b, out_dir, site,
                                ui_top=args.ui_top, ui_bottom=args.ui_bottom,
                                ui_top_auto=not args.no_ui_top_auto,
                                ssim_thresh=args.ssim_thresh, min_region_px=args.min_region_px,
                                top_n=args.top_n, win_size=args.win_size,
                                align=not args.no_align, max_shift_px=args.max_shift_px,
                                water_suppress=not args.no_water_suppress)
        print(f"  mean_ssim={result['mean_ssim']} overall_change={result['overall_change_fraction']} "
              f"n_regions={result['n_regions']} alignment={result['alignment']}")
        summary.append({"date_a": date_a, "date_b": date_b,
                         "overall_change_fraction": result["overall_change_fraction"],
                         "mean_ssim": result["mean_ssim"], "n_regions": result["n_regions"]})

    if len(pairs) > 1:
        timeline_path = out_dir / f"{site}_change_timeline.json"
        with open(timeline_path, "w", encoding="utf-8") as f:
            json.dump({"site": site, "pairs": summary}, f, ensure_ascii=False, indent=2)
        print(f"[change-detect] 時間軸摘要 -> {timeline_path}")


if __name__ == "__main__":
    main()
