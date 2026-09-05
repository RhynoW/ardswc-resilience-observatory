# -*- coding: utf-8 -*-
"""
地景複發熱點觀測站 — 台灣長期地質不穩定地點的衛星歷史影像變遷觀測站（port 8072）。

（2026-09-04 起：從原本的通用「衛星影像變遷偵測/日期推估瀏覽器」分家而來。原本的 8072
混合了「任意座標＋六類底圖」通用比對工具與台灣 ARDSWC 災害熱點分析兩個不同目的，使用者
要求拆開——通用工具原樣搬到新 app `webapp/imagery_change_toolkit`（port 8074），本 app
專注呈現台灣觀測站敘事，移除所有非台灣、非本觀測站資料集的比對能力。）

資料管線（見 CLAUDE.md 對應章節、以及本觀測站首頁「方法論」內容）：
  水保署 ARDSWC 歷史災害影像庫（97,500+ 張現場影像）
  → 自適應遞迴網格掃描 GetEventPositionList API（突破單次查詢 500 筆上限）
  → 76,773 筆全台去重事件紀錄
  → 依「不同年份出現次數」（非原始照片數）重新排序 → Top 100 長期複發熱點
  → 對每個熱點跑 Google Earth Web 歷史影像擷取（`ge_web_capture_v2_8k.py`，最長回溯 25 期）
  → `ge_change_detect.py` 的 SSIM 像素級變遷偵測（本 app 唯一 import 的比對引擎，不重寫演算法）
  → 11 個熱點做完整人工目視覆核，記入「深度驗證台帳」（data/ardswc_hotspots/ledger.json）

四個分頁：
  1. 「觀測站首頁」：統計總覽＋76,773 筆事件的分類/年份統計＋可切換底圖的熱點地圖
     （地圖可疊加原始事件點，見 `/api/events`）＋三種地貌演變型態說明。
  2. 「熱點總覽」：100 個熱點的可排序清單，點一筆載入該熱點目前已有的變遷比對面板。
  3. 「深度驗證台帳」：11 個熱點的完整目視覆核紀錄。
  4. 「發現與建言」：政府/民眾/公共政策三方向建言。

治理（§2 fail-closed，同 CLAUDE.md 對照原則）：所有變遷候選區塊/分數皆為自動化建議、
非已驗證事實；深度驗證台帳的「可信」判定僅代表「對位成功＋熱區集中」這兩項技術指標通過，
不代表已排除所有可能成因。單機單使用者，`use_reloader=False`。本 app 只服務 `ardswc_top*`
命名的站點資料——任何非此命名的站點一律拒絕（見 `_is_ardswc_site`）。

底圖來源（2026-09-04 追加）：Google/Bing/ESRI 前端直連；Apple／國土測繪中心 1/50000 地形圖／
正射影像三者需後端代理（`gmaps_tiles.py`，與 `imagery_change_toolkit` 同一份已驗證模組，僅
啟用本 app 需要的三個來源路由——不含百度/騰訊，本觀測站不服務中國大陸座標）。

事件位置標示（2026-09-04 追加）：`_report_marker()` 用該熱點座標＋擷取當下的 `.jgw` 世界檔
（EPSG:3857 六參數仿射）反解成面板影像像素位置，換算方式已用 rank44 案例驗證（見對話紀錄：
熱點座標理論上必落在原始擷取影像正中央，因擷取本來就是以該座標為相機中心，計算結果與此
預期完全吻合）。無 `.jgw`、或反解結果落在影像範圍外時回傳 `None`（fail-closed，不畫錯的點）。
"""
import hashlib
import io
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.utils import safe_join

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(HERE))
import ge_change_detect as CD          # noqa: E402
import gmaps_tiles as GT               # noqa: E402  — 僅用 apple/nlsc_topo/nlsc_photo 三個來源
import cesium_terrain as CT            # noqa: E402  — 可開關等高線圖用（Cesium World Terrain）

REPO = HERE.parent.parent
CAPTURES_ROOT = REPO / "data" / "ge_captures"
DATA_ROOT = REPO / "data" / "ardswc_hotspots"
CONTOUR_CACHE = REPO / "data" / "contour_cache"

app = Flask(__name__)


def _load_json(path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


_SITE_RE = re.compile(r"ardswc_top\d{2,3}(_deephist)?")


def _is_ardswc_site(site):
    """本觀測站只服務 ardswc_top<NN|NNN>[_deephist] 命名的站點——拒絕路徑穿越與非本專案資料。
    rank 可達 100（3 位數，如 ardswc_top100），故位數為 2-3 碼，不可寫死 2 碼（曾是真實 bug：
    rank100 的站名固定寫死 \\d{2} 會被 fullmatch 拒絕，回 403，見對話紀錄的使用者回報）。"""
    if not site or "/" in site or "\\" in site or ".." in site:
        return False
    return bool(_SITE_RE.fullmatch(site))


# ── 熱點座標查表（供事件位置標示使用）───────────────────────────────────────
_HOTSPOTS_CACHE = None


def _hotspots_by_rank():
    global _HOTSPOTS_CACHE
    if _HOTSPOTS_CACHE is None:
        rows = _load_json(DATA_ROOT / "top100_consolidated.json", [])
        _HOTSPOTS_CACHE = {r["rank"]: r for r in rows}
    return _HOTSPOTS_CACHE


def _rank_from_site(site):
    m = re.match(r"ardswc_top(\d{2,3})", site)
    return int(m.group(1)) if m else None


def _lonlat_to_mercator(lon, lat):
    R = 6378137.0
    mx = R * math.radians(lon)
    my = R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return mx, my


def _report_marker(site, date_a, date_b, roi):
    """該熱點座標在指定日期影像面板上的像素位置分數（frac_x/frac_y，0-1）。
    任何一步失敗（無座標、無 jgw、超出範圍）一律回 None，不畫錯誤的標記。"""
    rank = _rank_from_site(site)
    if rank is None:
        return None
    h = _hotspots_by_rank().get(rank)
    if not h or h.get("lon") is None or h.get("lat") is None:
        return None
    capture_dir = CAPTURES_ROOT / site
    png = None
    for d in (date_a, date_b):
        cand = capture_dir / f"{site}_gmap_{d}.png"
        # 只檢查 .jgw 是否存在，不要求原始 png 本身存在——_read_jgw() 只讀 .jgw 文字內容，
        # 從未觸碰 png 像素資料，故公開部署版可以只帶 .jgw 世界檔（幾十 bytes）而不必
        # 附上對應的原始 8K 擷取圖（數 MB～數十 MB），紅點標記功能仍完整可用。
        if cand.with_suffix(".jgw").exists():
            png = cand
            break
    if png is None:
        return None
    jgw = CD._read_jgw(png)
    if jgw is None:
        return None
    A, D, B, E, C, F = jgw
    mx, my = _lonlat_to_mercator(h["lon"], h["lat"])
    det = A * E - B * D
    if det == 0:
        return None
    px = (E * (mx - C) - B * (my - F)) / det
    py = (A * (my - F) - D * (mx - C)) / det
    top = roi.get("ui_top", 0)
    width = roi.get("width")
    height = roi.get("height")
    if not width or not height:
        return None
    panel_x, panel_y = px, py - top
    if not (0 <= panel_x <= width and 0 <= panel_y <= height):
        return None
    return {"frac_x": round(panel_x / width, 5), "frac_y": round(panel_y / height, 5)}


@app.route("/")
def index():
    return render_template("index.html", apple_enabled=GT.apple_is_available())


@app.route("/api/hotspots")
def api_hotspots():
    """100 個複發熱點的完整清單（rank/county/district/座標/複發年數/變遷分數/方法/台帳註記）。"""
    return jsonify(_load_json(DATA_ROOT / "top100_consolidated.json", []))


@app.route("/api/ledger")
def api_ledger():
    """11 個深度驗證熱點的完整目視覆核台帳。"""
    return jsonify(_load_json(DATA_ROOT / "ledger.json", []))


def _file_provenance(path, count=None):
    """單一資料檔的可追溯資訊：短雜湊＋最後修改時間＋筆數。即時計算、不落地存檔——
    這份資料本來就是跨多個工作階段手動逐步累積編輯的活文件（非一次性批次產出），
    寫死的 manifest 檔案反而容易漏更新、造成「manifest 說的版本」與「實際檔案」對不上；
    即時算雖然對這個檔案量級（10MB 級）完全不是效能問題，卻保證「顯示的永遠是真的」。
    `count` 由呼叫端傳入已經算好的筆數（reuse 既有的 `_hotspots_by_rank()`/`_load_events()`
    快取），這裡不再重複 `json.loads` 一次 9.6MB 的 `events_trimmed.json`。"""
    if not path.exists():
        return {"exists": False}
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()[:12]
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return {"exists": True, "sha256_12": digest, "size_bytes": len(raw), "modified": mtime, "count": count}


@app.route("/api/data_status")
def api_data_status():
    """資料可追溯性快照（審查建議 #1 的輕量版）：三份權威資料檔各自的雜湊/修改時間/筆數，
    供首頁顯示「這份分析結果是哪個版本的資料產生的」，也可供之後比對兩次匯出是否為同一批資料。
    刻意不做成離線批次寫檔的 manifest.json——見 `_file_provenance` docstring。"""
    _load_events()
    ledger_rows = _load_json(DATA_ROOT / "ledger.json", [])
    return jsonify({
        "top100_consolidated": _file_provenance(DATA_ROOT / "top100_consolidated.json", len(_hotspots_by_rank())),
        "ledger": _file_provenance(DATA_ROOT / "ledger.json", len(ledger_rows)),
        "events_trimmed": _file_provenance(DATA_ROOT / "events_trimmed.json", len(_EVENTS or [])),
    })


@app.route("/api/hotspot_sites/<int:rank>")
def api_hotspot_sites(rank):
    """該熱點目前有哪些擷取站點可用（深度 25 期 / 快篩 8 期），各自的日期清單。
    優先順序由前端決定（一律優先顯示 _deephist，沒有才用快篩站點）。"""
    out = {}
    for site in (f"ardswc_top{rank:02d}_deephist", f"ardswc_top{rank:02d}"):
        d = CAPTURES_ROOT / site
        dated = CD._list_dated(d) if d.exists() else []
        if len(dated) >= 2:
            out[site] = [x[0] for x in dated]
    return jsonify(out)


@app.route("/api/timeline/<site>")
def api_timeline(site):
    """該站點已計算過的相鄰日期配對摘要（擷取當下即已產生，見 quickbatch/deephist 批次腳本）。"""
    if not _is_ardswc_site(site):
        abort(403)
    p = CAPTURES_ROOT / site / "_change_detect" / f"{site}_change_timeline.json"
    return jsonify(_load_json(p, {"pairs": []}))


# ── 巡查優先級 A–D（2026-09-04 追加）───────────────────────────────────────
# 目的：把「一個籠統的自動分數」轉成「下一步該做什麼」（優先現勘／建議複核／持續監測／
# 資料待確認），供有限的巡查人力排序，而不是要求逐一瀏覽 100 個熱點。
#
# 刻意不做成一個相乘出來的單一分數（例如 recurrence × change × relief × confidence）——
# 那種黑盒分數看起來精確、實際上是拿幾個量綱、可信度都不同的數字硬乘在一起，換算過程
# 使用者完全看不懂為什麼是這個數字。改用「透明規則表」：四個因子各自算出高/中/低/未知，
# 分級邏輯用簡單的條件判斷、每一級都能回答「為什麼」，這也直接對應可解釋性面板的需求。
_TERRAIN_RELIEF_CACHE = None


def _terrain_relief_by_rank():
    """地形起伏（scripts/ardswc_terrain_relief.py 批次算好的結果）；批次尚未跑完的 rank
    就是沒有這筆資料——不假裝有，前端顯示「尚未計算」而不是 0（0 會被誤讀成平地）。"""
    global _TERRAIN_RELIEF_CACHE
    if _TERRAIN_RELIEF_CACHE is None:
        rows = _load_json(DATA_ROOT / "terrain_relief.json", [])
        _TERRAIN_RELIEF_CACHE = {r["rank"]: r for r in rows}
    return _TERRAIN_RELIEF_CACHE


def _dramatic_pair_alignment(rank):
    """讀該熱點目前站點（優先 _deephist）已快取的時間軸，找出 overall_change_fraction
    最高的那組配對，若該配對也有快取的完整 diff（含 alignment 欄位）就一併讀回。
    全程只讀既有磁碟快取，不觸發任何新的 SSIM 運算——維持這個端點是「即時、便宜」的。"""
    for site in (f"ardswc_top{rank:02d}_deephist", f"ardswc_top{rank:02d}"):
        tl = _load_json(CAPTURES_ROOT / site / "_change_detect" / f"{site}_change_timeline.json", None)
        if not tl or not tl.get("pairs"):
            continue
        pairs = tl["pairs"]
        dramatic = max(pairs, key=lambda p: p.get("overall_change_fraction") or -1)
        diff = _load_json(
            CAPTURES_ROOT / site / "_change_detect"
            / f"{site}_diff_{dramatic['date_a']}_{dramatic['date_b']}.json", None)
        alignment = (diff or {}).get("alignment")
        return {"site": site, "pair": dramatic, "alignment": alignment}
    return None


def _band(value, p33, p66):
    if value is None:
        return "未知"
    if value >= p66:
        return "高"
    if value >= p33:
        return "中"
    return "低"


def _confidence_band(rank, ledger_by_rank, dramatic):
    """信心分級的權威順序：人工覆核（若有）> 自動對位狀態 > 無資料。
    自動對位「已套用且不確定」封頂只能到「中」——沒有人看過的結果，不能標成「高」。"""
    l = ledger_by_rank.get(rank)
    if l:
        return {"ok": "高", "warn": "中", "bad": "低"}.get(l.get("verdict"), "中"), "human"
    if dramatic and dramatic.get("alignment"):
        al = dramatic["alignment"]
        if al.get("applied") and not al.get("uncertain"):
            return "中", "auto_aligned"
        return "低", "auto_uncertain"
    return "未知", "no_data"


def _decide_tier(recurrence_band, change_band, confidence_band, relief_band, confidence_source=None):
    """A 的判定核心是「近期變遷證據夠強＋信心夠」；複發年數不是硬性 AND 門檻——
    第一版曾把 recurrence=="高" 當成 A 的必要條件，結果讓本站唯一一個人工覆核「可信」
    的清晰案例（rank44，change_score 0.94、人工確認乾淨，但複發年數只是中等）落到
    B 級，一個已證實乾淨的訊號卻沒被標為優先——用這個已知的真實案例測出來才發現這個
    邏輯漏洞（複發年數在此只能當加分／邊界情況的調節因子，不能當 AND 閘）。

    第二個用已知案例測出來的漏洞：confidence_band=="中" 這個值同時代表兩種性質完全不同
    的情況——(a) 沒有人看過、但自動對位看起來正常（auto_aligned），(b) 人已經看過、
    明確標記「有疑慮」（ledger verdict="warn"）。原本兩者一視同仁，導致 rank28（人工已
    標記「warn」——分數被瀰漫雜訊/色調差異墊高，見台帳 note）在分數夠高時一樣被排進 A
    級「優先現勘」，等於自動邏輯覆蓋掉人已經給出的明確保留意見，直接違背本站「人工覆核
    優先於自動分數」的治理原則。修法：human+warn 一律先落 B，不進 A 快速通道；
    auto_aligned 的「中」則不受此限（沒人看過，本來就只能算自動訊號本身夠不夠強）。"""
    if confidence_band in ("低", "未知") or change_band == "未知":
        return "D", "資料待確認"
    human_warn = (confidence_source == "human" and confidence_band == "中")
    if change_band == "高" and confidence_band in ("高", "中") and not human_warn:
        if confidence_band == "中" and recurrence_band == "低":
            return "B", "建議複核"  # 信心僅中等、複發次數又低，兩個不利因子疊加時先複核較保守
        return "A", "優先現勘"
    if change_band == "中" and recurrence_band == "高" and confidence_band in ("高", "中") and not human_warn:
        if relief_band == "低":
            return "B", "建議複核"  # 地形平緩時，中等分數不直接升 A
        return "A", "優先現勘"
    if change_band in ("高", "中") and confidence_band != "高":
        return "B", "建議複核"  # 有變遷候選但信心不足（未經人工/對位不確定），需要複核而非直接派工
    if recurrence_band in ("高", "中") and change_band == "低":
        return "C", "持續監測"  # 歷史複發，但近期缺乏可靠的變遷證據
    return "B", "建議複核"  # 其餘落在中間地帶，預設走複核，不自動放行到 A


def _priority_for(h, ledger_by_rank, relief_by_rank, score_p33, score_p66, relief_p33, relief_p66):
    rank = h["rank"]
    dramatic = _dramatic_pair_alignment(rank)
    change_band = _band(h.get("change_score"), score_p33, score_p66)
    recurrence_band = _band(h.get("n_distinct_years"), 4, 6)  # 依實際分布 p33≈3/p66≈4，取略高門檻避免「高」被灌水
    relief_row = relief_by_rank.get(rank)
    relief_val = relief_row.get("relief_m") if relief_row else None
    relief_band = _band(relief_val, relief_p33, relief_p66) if relief_val is not None else "未知"
    confidence_band, confidence_source = _confidence_band(rank, ledger_by_rank, dramatic)
    tier, tier_label = _decide_tier(recurrence_band, change_band, confidence_band, relief_band, confidence_source)
    return {
        "rank": rank, "tier": tier, "tier_label": tier_label,
        "factors": {
            "recurrence": {"value": h.get("n_distinct_years"), "band": recurrence_band},
            "recent_change": {"value": h.get("change_score"), "band": change_band},
            "terrain_relief_m": {"value": relief_val, "band": relief_band},
            "confidence": {"band": confidence_band, "source": confidence_source},
        },
        "dramatic_pair": dramatic["pair"] if dramatic else None,
    }


@app.route("/api/priority")
def api_priority():
    """100 個熱點的巡查優先級 A–D，見上方模組註解。全部由既有磁碟快取資料現算，
    不觸發任何新的 SSIM 運算或外部 API 呼叫（地形起伏另由批次腳本預先算好）。"""
    hotspots = _load_json(DATA_ROOT / "top100_consolidated.json", [])
    ledger_by_rank = {l["rank"]: l for l in _load_json(DATA_ROOT / "ledger.json", [])}
    relief_by_rank = _terrain_relief_by_rank()

    scores = sorted(h["change_score"] for h in hotspots if h.get("change_score") is not None)
    reliefs = sorted(r["relief_m"] for r in relief_by_rank.values() if r.get("relief_m") is not None)

    def pctl(arr, p):
        if not arr:
            return None
        return arr[min(int(len(arr) * p), len(arr) - 1)]

    score_p33, score_p66 = pctl(scores, 0.33), pctl(scores, 0.66)
    relief_p33, relief_p66 = pctl(reliefs, 0.33), pctl(reliefs, 0.66)

    out = [_priority_for(h, ledger_by_rank, relief_by_rank, score_p33, score_p66, relief_p33, relief_p66)
           for h in hotspots]
    return jsonify({
        "items": out,
        "band_basis": {
            "change_score_p33": score_p33, "change_score_p66": score_p66,
            "terrain_relief_p33": relief_p33, "terrain_relief_p66": relief_p66,
            "terrain_relief_computed_n": len(reliefs), "terrain_relief_total_n": len(hotspots),
        },
        "governance_note": ("巡查優先級為排序建議，非災害確定性判定；A 級仍需現勘或專業判讀確認，"
                             "D 級代表資料不足以支持任何判斷，不代表風險較低。"),
    })


@app.route("/api/pair/<site>/<date_a>/<date_b>")
def api_pair(site, date_a, date_b):
    """單一日期配對的完整比對結果。優先讀取既有快取的 JSON（擷取當下已產生的面板圖）；
    若這組配對從未算過（例如深度站點只算過頭尾、使用者想看中間某兩期），才即時呼叫
    `ge_change_detect.detect_change()` 現算——與既有快取走同一份函式，結果格式一致。
    另外附加 `report_marker`（該熱點座標在面板影像上的位置，供前端疊標記）。"""
    if not _is_ardswc_site(site):
        abort(403)
    a, b = sorted([date_a, date_b])
    out_dir = CAPTURES_ROOT / site / "_change_detect"
    cached = out_dir / f"{site}_diff_{a}_{b}.json"
    result = _load_json(cached, None)

    if result is None:
        capture_dir = CAPTURES_ROOT / site
        dated = dict(CD._list_dated(capture_dir))
        if a not in dated or b not in dated:
            return jsonify({"error": "指定日期不在此站點的擷取清單中"}), 400
        try:
            result = CD.detect_change(dated[a], dated[b], a, b, out_dir, site)
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    result["report_marker"] = _report_marker(site, a, b, result.get("roi", {}))
    return jsonify(result)


# ── 靜態影像服務（safe_join 擋 ../ 穿越，同 §14.6 B.1 慣例）──────────────────
@app.route("/image/<path:relpath>")
def serve_image(relpath):
    full = safe_join(str(CAPTURES_ROOT), relpath)
    if not full or not Path(full).exists():
        abort(404)
    try:
        Path(full).resolve().relative_to(CAPTURES_ROOT.resolve())
    except ValueError:
        abort(403)
    return send_file(full)


# ── GE Web 回溯（換日期直連 URL，同 §14.7/reference_ge_web_date_url 手法）─────
@app.route("/api/ge_trace")
def api_ge_trace():
    lon, lat = request.args.get("lon"), request.args.get("lat")
    date = request.args.get("date", "")
    dist = request.args.get("dist", "500")
    if not (lon and lat):
        return jsonify({"error": "缺 lon/lat"}), 400
    template = f"https://earth.google.com/web/@{lat},{lon},0.00a,{dist}d,35y,0h,0t,0r"
    try:
        import ge_web_capture as GW  # noqa: PLC0415
        url = GW.build_url(template, date) if (date and len(date) == 8) else template
    except Exception:
        url = template
    return jsonify({"url": url})


# ── 底圖圖磚代理（僅 apple / 國土測繪中心兩層，不含百度/騰訊——本觀測站不服務中國大陸座標）──
@app.route("/api/tile/apple/<int:z>/<int:x>/<int:y>")
def api_apple_tile(z, x, y):
    img = GT.download_tile(x, y, z, source="apple")
    if img is None:
        abort(404)
    from io import BytesIO
    buf = BytesIO(); img.save(buf, format="JPEG", quality=88); buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


@app.route("/api/tile/nlsc_topo/<int:z>/<int:x>/<int:y>")
def api_nlsc_topo_tile(z, x, y):
    img = GT.download_tile(x, y, z, source="nlsc_topo")
    if img is None:
        abort(404)
    from io import BytesIO
    buf = BytesIO(); img.save(buf, format="JPEG", quality=88); buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


@app.route("/api/tile/nlsc_photo/<int:z>/<int:x>/<int:y>")
def api_nlsc_photo_tile(z, x, y):
    img = GT.download_tile(x, y, z, source="nlsc_photo")
    if img is None:
        abort(404)
    from io import BytesIO
    buf = BytesIO(); img.save(buf, format="JPEG", quality=88); buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


@app.route("/api/apple_status")
def api_apple_status():
    return jsonify({"configured": GT.apple_is_available()})


# ── 可開關等高線圖（2026-09-04 追加，Cesium World Terrain）───────────────────
# 技術取自 F:\GitHub\Infrared_Small_Target_Detection 的 CUAS 四格圖輸出：Cesium ion REST
# Terrain API 抓 quantized-mesh 高程 → 內插網格 → OpenCV 逐等高線 threshold-mask 畫線
# （見 scripts/cesium_terrain.py 開頭的改編說明）。與該專案不同的是：這裡疊圖用途是
# Leaflet 透明 overlay tile（PNG，RGBA），不是產生獨立地形面板；且只在中高 zoom（見
# _CONTOUR_MIN_Z/_CONTOUR_MAX_Z）才真的打 API，避免低 zoom 時對大範圍濫發請求。
#
# 授權提醒（同 cesium_terrain.py 文件）：免費層級 Cesium World Terrain 不得商用，本觀測站
# 為研究/展示用途；若日後有商用需求需另行升級 Cesium ion 方案。
_CONTOUR_MIN_Z = int(os.environ.get("CONTOUR_MIN_Z", "11"))
_CONTOUR_MAX_Z = int(os.environ.get("CONTOUR_MAX_Z", "15"))
_CONTOUR_INTERVAL_M = float(os.environ.get("CONTOUR_INTERVAL_M", "50"))
_CONTOUR_INDEX_EVERY = int(os.environ.get("CONTOUR_INDEX_EVERY", "5"))
_CONTOUR_TILE_PX = 256

_contour_source = None  # 延遲初始化：token 不存在時不能讓整個 app 啟動失敗


def _get_contour_source():
    global _contour_source
    if _contour_source is None:
        _contour_source = CT.IonTerrainSource()
    return _contour_source


def _tile_bounds_lonlat(z, x, y):
    """標準 Web Mercator slippy tile -> (west, south, east, north) 經緯度。"""
    n = 2 ** z
    lon_w = x / n * 360.0 - 180.0
    lon_e = (x + 1) / n * 360.0 - 180.0
    lat_n = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_s = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return lon_w, lat_s, lon_e, lat_n


def _transparent_png(px=_CONTOUR_TILE_PX):
    import cv2
    import numpy as np
    canvas = np.zeros((px, px, 4), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", canvas)
    return buf.tobytes() if ok else b""


def _render_contour_tile(z, x, y):
    """回傳該 tile 的透明等高線 PNG bytes。任何一步失敗一律回全透明圖（fail-open——
    沒有等高線疊圖不影響底圖本身可用性，比照本觀測站其餘「無法算出就不畫」的 fail-closed
    精神，只是這裡「不畫」的後果無害，用 fail-open 措辭更精確）。"""
    import cv2
    import numpy as np
    from scipy.ndimage import map_coordinates

    lon_w, lat_s, lon_e, lat_n = _tile_bounds_lonlat(z, x, y)
    center_lat, center_lon = (lat_s + lat_n) / 2.0, (lon_w + lon_e) / 2.0
    mpp = 156543.03392 * math.cos(math.radians(center_lat)) / (2 ** z)
    span_km = (mpp * _CONTOUR_TILE_PX) / 1000.0

    try:
        src = _get_contour_source()
        # 多取 20% 邊界，供重採樣時邊緣不缺資料；res_m 依 tile 實際地面解析度換算，
        # 避免對粗 zoom 也硬要求精細網格（浪費 API 請求）。
        z_grid, gl, go, info = CT.fetch_terrain(
            center_lat, center_lon, span_km=span_km * 1.2, res_m=max(mpp, 15.0), source=src)
    except Exception:
        return _transparent_png()

    render_px = _CONTOUR_TILE_PX * 2  # 2x 超取樣供反鋸齒，最後縮小
    lat_axis = np.linspace(lat_n, lat_s, render_px)   # row 0 = 北（影像上緣）
    lon_axis = np.linspace(lon_w, lon_e, render_px)
    row_f = np.interp(lat_axis, gl, np.arange(len(gl)))
    col_f = np.interp(lon_axis, go, np.arange(len(go)))
    RF, CF = np.meshgrid(row_f, col_f, indexing="ij")
    elev = map_coordinates(z_grid, [RF, CF], order=1, mode="nearest")

    canvas = np.zeros((render_px, render_px, 4), dtype=np.uint8)
    if np.isfinite(elev).any():
        lo = math.floor(float(np.nanmin(elev)) / _CONTOUR_INTERVAL_M) * _CONTOUR_INTERVAL_M
        hi = math.ceil(float(np.nanmax(elev)) / _CONTOUR_INTERVAL_M) * _CONTOUR_INTERVAL_M
        levels = np.arange(lo, hi + _CONTOUR_INTERVAL_M, _CONTOUR_INTERVAL_M)
        elev_u8 = elev  # findContours 需要單通道遮罩，逐 level 產生，不需先轉型
        for lv in levels:
            mask = (elev_u8 >= lv).astype(np.uint8)
            cnts, _hier = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cnts = [c for c in cnts if len(c) > 6]
            if not cnts:
                continue
            is_index = (round(lv / _CONTOUR_INTERVAL_M) % _CONTOUR_INDEX_EVERY == 0)
            # 原本細線(1px)+低 alpha(130) 在 2x 超取樣→INTER_AREA 縮小到 256 的過程中
            # 幾乎被平均掉到接近透明，疊在森林/山地衛星影像上完全看不清（使用者實測回報）。
            # 改法：(a) 先畫一道較粗的白色暈邊(halo)、再疊上實際顏色的線——暈邊在任何
            # 背景色（深綠林地/裸岩/水面）都能撐出對比，不靠單一顏色本身的區分度；
            # (b) 兩者 alpha 都拉高到接近不透明；(c) 線寬加粗，抵銷縮小造成的稀釋。
            main_color = (33, 67, 101, 255) if is_index else (60, 130, 210, 235)  # BGRA
            halo_th = 7 if is_index else 5
            main_th = 3 if is_index else 2
            cv2.drawContours(canvas, cnts, -1, (255, 255, 255, 215), halo_th, cv2.LINE_AA)
            cv2.drawContours(canvas, cnts, -1, main_color, main_th, cv2.LINE_AA)

    small = cv2.resize(canvas, (_CONTOUR_TILE_PX, _CONTOUR_TILE_PX), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", small)
    return buf.tobytes() if ok else _transparent_png()


@app.route("/api/contours/<int:z>/<int:x>/<int:y>")
def api_contour_tile(z, x, y):
    if z < _CONTOUR_MIN_Z:
        # 低 zoom 涵蓋範圍太大，不值得也不該對 Cesium ion 發請求——直接回透明。
        return send_file(io.BytesIO(_transparent_png()), mimetype="image/png")
    z_clamped = min(z, _CONTOUR_MAX_Z)
    if z_clamped != z:
        # 前端 maxNativeZoom 應已擋掉這種情況（改用瀏覽器端放大既有 tile），這裡是保險。
        return abort(404)

    cache_fp = CONTOUR_CACHE / f"{z}_{x}_{y}.png"
    if cache_fp.exists():
        return send_file(str(cache_fp), mimetype="image/png")

    png_bytes = _render_contour_tile(z, x, y)
    try:
        CONTOUR_CACHE.mkdir(parents=True, exist_ok=True)
        cache_fp.write_bytes(png_bytes)
    except OSError:
        pass
    return send_file(io.BytesIO(png_bytes), mimetype="image/png")


@app.route("/api/contours_status")
def api_contours_status():
    """前端用來判斷等高線功能是否可用（token 缺失/Cesium API 打不通時不顯示開關），
    以及告知 attribution/授權限制文字。"""
    try:
        src = _get_contour_source()
        src.endpoint()  # 觸發一次真實驗證（含 token 有效性），不只是檢查檔案存在
        return jsonify({"available": True, "attributions": src.attributions,
                         "commercial_ok": src.commercial_use_allowed()})
    except Exception as e:  # noqa: BLE001
        return jsonify({"available": False, "error": f"{type(e).__name__}: {e}"})


# ── 76,773 筆原始事件：統計/分類/地圖疊點（2026-09-04 追加）─────────────────
# 欄位取自 ARDSWC GetEventPositionList 原始回應，僅保留本觀測站用得到的子集
# （County 欄位全為 null，DisasterYear 亦全為 null，經抽查證實不可用，故不提供依縣市/災害年份
# 篩選；PhotoType/CreateTime_YYYY 為唯二可靠的分類維度）。
_EVENTS = None
_EVENTS_STATS = None

_PHOTO_TYPE_LABELS = {"0": "災害事件", "6": "重要地景", "8": "媒體報導", "10": "出版品照片"}


def _load_events():
    global _EVENTS, _EVENTS_STATS
    if _EVENTS is not None:
        return
    _EVENTS = _load_json(DATA_ROOT / "events_trimmed.json", [])
    from collections import Counter
    pt_counter = Counter(e.get("photo_type") for e in _EVENTS)
    yr_counter = Counter(e.get("year") for e in _EVENTS)
    _EVENTS_STATS = {
        "total": len(_EVENTS),
        "by_photo_type": [
            {"code": k, "label": _PHOTO_TYPE_LABELS.get(k, k or "未分類"), "count": v}
            for k, v in sorted(pt_counter.items(), key=lambda kv: -kv[1])
        ],
        "by_year": [
            {"year": k, "count": v} for k, v in sorted(yr_counter.items(), key=lambda kv: (kv[0] or ""))
        ],
        "note": "County/DisasterYear 欄位於來源資料全為空值，故僅提供 PhotoType（事件分類）與"
                "CreateTime_YYYY（資料建檔年份，非災害實際發生年份）兩個維度；ARDSWC 原始資料"
                "不含可靠的行政區欄位，本觀測站不對 76,773 筆原始事件做縣市分類。",
    }


@app.route("/api/events/stats")
def api_events_stats():
    _load_events()
    return jsonify(_EVENTS_STATS)


@app.route("/api/events")
def api_events():
    """依 photo_type / year 篩選，回傳最多 limit 筆（預設/上限 3000）供地圖疊點——76,773 筆
    全量不適合直接丟給瀏覽器，故一律裁切，並在回應帶 truncated 旗標讓前端知道還有更多未顯示。"""
    _load_events()
    photo_type = request.args.get("photo_type")
    year = request.args.get("year")
    try:
        limit = min(3000, max(1, int(request.args.get("limit", 3000))))
    except ValueError:
        limit = 3000

    matched = []
    for e in _EVENTS:
        if photo_type and e.get("photo_type") != photo_type:
            continue
        if year and e.get("year") != year:
            continue
        matched.append(e)
        if len(matched) >= limit:
            break
    total_matched = sum(
        1 for e in _EVENTS
        if (not photo_type or e.get("photo_type") == photo_type)
        and (not year or e.get("year") == year)
    )
    return jsonify({"points": matched, "returned": len(matched), "total_matched": total_matched,
                     "truncated": total_matched > len(matched)})


if __name__ == "__main__":
    # HF Space 容器內用 PORT 環境變數（預設 7860，HF 慣例）＋監聽 0.0.0.0；
    # 本機開發沒設 PORT 時維持原本 127.0.0.1:8072，行為不變。
    _port = int(os.environ.get("PORT", 8072))
    _host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    app.run(host=_host, port=_port, debug=False, use_reloader=False, threaded=True)
