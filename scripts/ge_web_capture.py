# -*- coding: utf-8 -*-
"""
ge_web_capture.py — 全自動擷取 Google Earth Web 歷史影像（逐日期），供 J7-Wuhu 等定向蒐集。

原理（見 memory reference-ge-web-date-url）：GE Web 的所有相機參數 + 歷史日期都在 URL 裡可控：
  https://earth.google.com/web/@{lat},{lon},{alt}a,{dist}d,{yaw}y,{h}h,{tilt}t,{roll}r/data=<protobuf,含 YYYY-MM-DD>
日期是 data= protobuf(base64url) 內的 ASCII 字串（field2: \\x12\\x0a + 10 bytes），同長度直接替換即可。
打開該 URL 會自動進歷史影像模式並跳到該日期（自動關 3D → 乾淨俯視）。

作法：Playwright 開 headed Chrome（用真 GPU 穩定跑 GE 的 WebGL）→ 逐日期 goto URL → 等 tiles 載入
→ page.screenshot() 直接擷取渲染後的頁面（含 GE 內建時間軸/標籤，無瀏覽器工具列）。device_scale_factor
放大 → 不需虛擬 8K 屏即可高解析。可選 --upload 直接把每張送 gmaps_bbox_SIFT2 的 /api/upload_detect
（OBB+SAM3 偵測 → 產生待複核候選；確認 J7 仍由人做，§2 治理）。

治理：模型只建議、非 ground truth；擷取為當前自動化，確認 J7（三角翼）由人在 8060 UI 完成。

用法：
  # 先擷 1 個日期驗證（capture-only）：
  python scripts/ge_web_capture.py --site wuhuairbase --lat 31.3908 --lon 118.4094 \
      --dates 20050427 --dist 1200 --wait 14 --outdir data/ge_captures
  # 全部 19 個蕪湖日期（從 spec 讀）+ 自動送偵測：
  python scripts/ge_web_capture.py --site wuhuairbase --lat 31.3908 --lon 118.4094 \
      --dates-file data/collection_targets/j7_wuhu_ge_historical.json --dist 1200 --wait 14 --upload
"""
import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

try:                       # Windows cp950 console 無法印 ✔ 等字元 → 強制 utf-8
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(__file__).resolve().parent.parent  # 可攜路徑（2026-09-05 修正，原硬編碼 F:\ 在容器/HF Space 上不存在）
# 使用者提供的「已驗證合理視野」範本 URL（蕪湖 apron，dist≈15784）。只抽換 data= 內的日期即可維持視野。
# GE 預設載入地形 → 相機到地表距離小於海平面距離，dist 太小(如 1200/5000)會太近而糊；此 dist 已驗證正確。
DEFAULT_TEMPLATE = ("https://earth.google.com/web/@31.39041627,118.3970501,17.25758692a,"
                    "15784.67943035d,1y,0h,0t,0r/data=ChYqEAgBEgoyMDA1LTA0LTI3GAFCAggBOgMKATBCAggASg0I"
                    "____________ARAA?authuser=0")


# 位置無關的「歷史影像單圖層 + 日期」canonical data= protobuf（取自 DEFAULT_TEMPLATE，含 2005-04-27）。
# 用於：種子網址是搜尋 URL 或無 data= 的相機 URL 時（如 ge_mosaic_plan 把搜尋 URL 轉成純相機、去掉 data=），
# 仍能靠抽換日期做歷史擷取——附上此 canonical data= 再換日期即可進歷史模式（同 memory reference-ge-web-date-url）。
CANON_HIST_DATA = "ChYqEAgBEgoyMDA1LTA0LTI3GAFCAggBOgMKATBCAggASg0I____________ARAA"


def _split_template(url):
    """拆出 (camera_str, data_b64 or None)。camera 逐字保留，只換 data= 內日期。
    容忍無 /data= 的 URL（回 data_b64=None，由 build_url 補 canonical 歷史 data=）。"""
    cam = url.split("/@", 1)[1].split("/data=", 1)[0].split("?", 1)[0]
    data_b64 = None
    if "/data=" in url:
        data_b64 = url.split("/data=", 1)[1].split("?", 1)[0].split("&", 1)[0]
    return cam, data_b64


def build_url(template, date_yyyymmdd):
    import re
    cam, data_b64 = _split_template(template)
    if data_b64 is None:                       # 種子無 data=（搜尋 URL/純相機）→ 補 canonical 歷史 protobuf 再換日期
        data_b64 = CANON_HIST_DATA
    b = base64.urlsafe_b64decode(data_b64 + "=" * (-len(data_b64) % 4))
    dd = f"{date_yyyymmdd[:4]}-{date_yyyymmdd[4:6]}-{date_yyyymmdd[6:]}"
    b2 = re.sub(rb"\d{4}-\d{2}-\d{2}", dd.encode(), b, count=1)   # 同長度替換 → 視野不變
    data_new = base64.urlsafe_b64encode(b2).decode().rstrip("=")
    return f"https://earth.google.com/web/@{cam}/data={data_new}"


def load_dates(args):
    if args.dates:
        return [d.strip() for d in args.dates.replace(",", " ").split() if d.strip()]
    if args.dates_file:
        j = json.load(open(args.dates_file, encoding="utf-8"))
        return list(j["dates"])
    raise SystemExit("need --dates or --dates-file")


def maybe_upload(png_path: Path, session: str, model: str, wait_done=True, timeout=240):
    """POST 截圖到 gmaps_bbox_SIFT2 /api/upload_detect（OBB+SAM3 → 待複核候選）。
    wait_done=True：輪詢 /api/detect/<job> 至完成才回傳（**序列化 → GPU 安全**，避免多個 SAM3 併發燒卡）。
    回傳簡短狀態字串（含偵測數）。"""
    import requests
    import time as _t
    try:
        with open(png_path, "rb") as f:
            r = requests.post("http://127.0.0.1:8060/api/upload_detect",
                              files={"image": (png_path.name, f, "image/png")},
                              data={"model": model, "session": session}, timeout=120)
        j = r.json()
    except Exception as e:
        return f"POST-ERR {type(e).__name__}"
    if not j.get("ok"):
        return f"ERR {j.get('error')}", []
    jid = j["job_id"]
    if not wait_done:
        return jid, []
    t0 = _t.time()
    while _t.time() - t0 < timeout:
        try:
            s = requests.get(f"http://127.0.0.1:8060/api/detect/{jid}", timeout=30).json()
        except Exception:
            _t.sleep(2); continue
        st = s.get("status")
        if st == "done":
            dets = s.get("result", {}).get("detections", [])
            return f"{jid} ({len(dets)} dets)", dets
        if st == "error":
            return f"{jid} DET-ERR {s.get('error')}", []
        _t.sleep(2)
    return f"{jid} (timeout)", []


# ---- Flutter 無障礙(semantics) DOM 自動化：GE Web 是 Flutter/CanvasKit，啟用 a11y 後 UI 變成
#      可讀/可點的 DOM 語意樹，即可自動探索歷史日期並逐日期截圖（實測可行）。----
_JS_NODES = """() => [...document.querySelectorAll('flt-semantics,[role]')].map(e=>{
  const r=e.getBoundingClientRect();
  return {tx:(e.textContent||'').trim(), al:e.getAttribute('aria-label'), role:e.getAttribute('role'),
          x:Math.round(r.x+r.width/2), y:Math.round(r.y+r.height/2), w:Math.round(r.width)};
}).filter(o=>o.w>0)"""


def _enter_historical(page):
    """若尚未在歷史影像模式（無日期列），點 semantics 的『顯示歷史圖像』開關進入。回傳是否有點。"""
    try:
        for o in page.evaluate(_JS_NODES):
            lab = (o.get('al') or '') + " " + (o.get('tx') or '')
            if ('歷史' in lab or 'historical' in lab.lower()) and o['w'] <= 60 and o['y'] < 120:
                page.mouse.click(o['x'], o['y'])
                return True
    except Exception:
        pass
    return False


def _hide_annotations(page):
    """Ctrl+B 開『基本地圖設定』→ 點『全部隱藏』關閉標註圖層（邊界/地名/地點/道路）→ 關面板。
    這是移除 GE 地圖文字標籤（水道/地名，會被下游偵測誤當目標）最乾淨的做法（使用者 2026-07-26）。
    基本地圖設定為全域、跨日期持續，故只需在開始時做一次。回傳是否點到『全部隱藏』。"""
    try:
        page.keyboard.press("Control+b"); time.sleep(2.5)
        hit = None
        for o in page.evaluate(_JS_NODES):
            lab = (o.get('al') or '') + " " + (o.get('tx') or '')
            if '全部隱藏' in lab or 'hide all' in lab.lower():
                hit = (o['x'], o['y']); break
        if hit:
            page.mouse.click(*hit); time.sleep(1.5)
        page.keyboard.press("Escape"); time.sleep(1.0)      # 關面板
        return hit is not None
    except Exception:
        return False


def _collapse_timeline(page):
    """點 semantics 的『收合』控制，收起歷史時間軸/工具列 → 截圖更乾淨、少 UI 遮擋。
    回傳是否有點到。以文字/aria-label 含『收合』或『collapse』辨識（位置+role 無關語言）。"""
    try:
        for o in page.evaluate(_JS_NODES):
            lab = (o.get('al') or '') + " " + (o.get('tx') or '')
            if '收合' in lab or 'collapse' in lab.lower():
                page.mouse.click(o['x'], o['y'])
                return True
    except Exception:
        pass
    return False


def _enable_a11y(page):
    """點 Flutter 的『Enable accessibility』佔位鈕 → 建立 semantics DOM 樹。"""
    try:
        page.evaluate("""()=>{const c=[...document.querySelectorAll('[aria-label]')]
            .find(e=>/accessibility/i.test(e.getAttribute('aria-label')||'')); if(c){c.click();return 1;} return 0;}""")
        return True
    except Exception:
        return False


_EN_MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


def _read_stepper(page):
    """從 semantics 樹讀歷史日期列：回傳 (date 'YYYY-MM-DD'|None, newer中心xy|None, older中心xy|None)。
    以『按鈕標籤』辨識(與位置無關)：newer=較新的圖片/newer、older=較舊的圖片/older——比純位置穩健：
    實測某些視角(旋轉/近景/搜尋)兩個箭頭都在日期左側，純『左=prev 右=next』會抓不到 next。
    回傳 (date, next=newer, prev=older)：next 走向較新、prev 走向較舊(最早)。

    **英文 locale fallback（2026-09-05 追加）**：容器環境（無 zh-TW 系統 locale）下 GE Web 會用
    英文渲染，且整條歷史工具列（『Older images』/日期/『Newer images』等）被 Flutter 拼成同一個
    accessibility 節點（如 `'...Historical ImageryOlder imagesJan 1, 2020Newer imagesCollapse...'`），
    不再是各自獨立、長度 <16 的短節點——原本的中文數字日期正則＋短節點長度限制會直接找不到、
    整支函式提早回傳 None。故先照舊嘗試中文短節點格式（開發者本機 zh-TW locale 下的既有行為
    完全不變），找不到才在**任何節點**（不限長度）內用英文月份格式（`Jan 1, 2020`）的正則再找
    一次——按鈕比對邏輯本身沿用同一段（找到的節點 y 座標一樣落在按鈕列同一帶，`newer`/`older`
    的英文字串比對本來就已支援，只是先前從未執行到）。"""
    import re
    try:
        nodes = page.evaluate(_JS_NODES)
    except Exception:
        return None, None, None
    dl = None
    for o in nodes:
        m = re.search(r'(\d{4}).(\d{1,2}).(\d{1,2})日?$', o['tx'])
        if m and o['y'] < 180 and len(o['tx']) < 16:
            dl = (o, f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"); break
    if not dl:
        for o in nodes:
            if o['y'] >= 180:
                continue
            m = re.search(r'\b([A-Z][a-z]{2})\.?\s+(\d{1,2}),\s+(\d{4})\b', o['tx'])
            if m and m.group(1) in _EN_MONTHS:
                dl = (o, f"{m.group(3)}-{_EN_MONTHS[m.group(1)]:02d}-{int(m.group(2)):02d}"); break
    if not dl:
        return None, None, None
    o, ds = dl
    newer = older = None
    for b in nodes:
        if b.get('role') != 'button' or abs(b['y'] - o['y']) > 25:
            continue
        lab = (b.get('al') or '') + (b.get('tx') or '')
        if '較新' in lab or 'newer' in lab.lower():
            newer = (b['x'], b['y'])
        elif '較舊' in lab or 'older' in lab.lower():
            older = (b['x'], b['y'])
    if newer is None and older is None:            # fallback：純位置（舊行為，左=older 右=newer）
        btns = [b for b in nodes if b.get('role') == 'button' and abs(b['y'] - o['y']) < 25 and b['w'] <= 48]
        r = min([b for b in btns if b['x'] > o['x']], key=lambda b: b['x'] - o['x'], default=None)
        l = min([b for b in btns if b['x'] < o['x']], key=lambda b: o['x'] - b['x'], default=None)
        newer = (r['x'], r['y']) if r else None
        older = (l['x'], l['y']) if l else None
    return ds, newer, older


def _wait_tiles(page, settle, timeout=45):
    """輪詢 semantics 的載入進度條至 100%/消失，再 settle。回傳是否在時限內完成。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            pct = page.evaluate("""()=>{const p=[...document.querySelectorAll('[role=progressbar]')]
                .map(e=>e.getAttribute('aria-label')||'').find(t=>/\\d+%/.test(t));
                if(!p)return 100; const m=p.match(/(\\d+)%/); return m?+m[1]:100;}""")
        except Exception:
            pct = 100
        if pct is None or pct >= 100:
            break
        time.sleep(1.0)
    time.sleep(settle)
    return True


def _wait_app_ready(page, floor=6.0, cap=28.0):
    """等 GE Flutter app 開機到 semantics 可互動（a11y 已啟用且能讀到歷史日期列）。
    先睡 floor（app 最起碼開機時間），再啟用 a11y 並輪詢日期列出現即提早返回；cap 為上限。
    用於 --dates 路徑取代首張/每張的固定長 sleep（D：app 準備好就提早往下走）。
    回傳讀到的日期字串，或 None（逾 cap 仍讀不到 → 呼叫端改用保守 settle）。"""
    t0 = time.time()
    time.sleep(floor)
    _enable_a11y(page)
    ds = None
    while time.time() - t0 < cap:
        try:
            ds, _, _ = _read_stepper(page)
        except Exception:
            ds = None
        if ds:
            break
        _enable_a11y(page)                     # goto 重載後 a11y 佔位鈕可能稍晚才出現 → 重試
        time.sleep(1.5)
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="wuhuairbase")
    ap.add_argument("--template-url", default=DEFAULT_TEMPLATE,
                    help="已驗證視野的 GE Web URL；只抽換 data= 內日期。預設=蕪湖 apron(dist≈15784)")
    ap.add_argument("--dates", default="")
    ap.add_argument("--dates-file", default="")
    ap.add_argument("--auto-dates", action="store_true",
                    help="啟用 Flutter 無障礙 → 自動探索該地點所有可用歷史日期並逐一截圖（不需 --dates）")
    ap.add_argument("--max-dates", type=int, default=80, help="auto-dates 安全上限")
    ap.add_argument("--skip-first", type=int, default=1,
                    help="auto-dates：跳過最早的 N 張（GE 最早一張通常是 Landsat 資源衛星，解析度太粗、無機體細節）")
    ap.add_argument("--outdir", default=str(REPO / "data" / "ge_captures"))
    ap.add_argument("--wait", type=float, default=14.0, help="每個日期 tiles 載入等待秒數（auto-dates 的 settle；--dates 路徑 a11y 讀不到時的保守 fallback settle）")
    ap.add_argument("--settle", type=float, default=5.0,
                    help="--dates 路徑：進度條輪詢到 100% 後的最後 settle 秒數（事件驅動；a11y 就緒時取代舊的盲等 --wait）")
    ap.add_argument("--first-wait", type=float, default=25.0, help="首個日期額外等待(GE 初次載入較久)")
    ap.add_argument("--vw", type=int, default=3840, help="viewport 寬(CSS px)；Playwright 離屏渲染故可 > 實體螢幕")
    ap.add_argument("--vh", type=int, default=2160, help="viewport 高(CSS px)")
    ap.add_argument("--dsf", type=float, default=1.0,
                    help="device_scale_factor。務必=1：>1 會 upscale GE 的 WebGL canvas 使影像變糊(文字反而變清)。要更高解析改用更大 --vw/--vh")
    ap.add_argument("--window-pos", default="0,0", help="瀏覽器視窗左上角 x,y（決定顯示在哪個螢幕；page.screenshot 與螢幕無關，此僅供你觀看）")
    ap.add_argument("--no-pan", action="store_true",
                    help="不做『拖曳平移 + 微 zoom』觸發高 LOD 的動作（改只靠 goto + 等進度條 100% 載入）。"
                         "近距靜態 nadir 視角常已足夠、且省去平移殘留位移（利置中裁切）。預設仍做平移（其他呼叫端不受影響）。")
    ap.add_argument("--upload", action="store_true", help="擷取後自動送 /api/upload_detect")
    ap.add_argument("--model", default="", help="upload 用的建議模型(空=伺服器最新)")
    ap.add_argument("--profile", default=str(REPO / "data" / "ge_captures" / "_chrome_profile"),
                    help="持久化 Chrome profile（保留登入/cookie，加速後續載入）")
    ap.add_argument("--geo-ref", action="store_true",
                    help="擷取後對截圖做地理定位：以 GMaps 為參考、SIFT 對位最新日期，輸出 .jpg/.jgw/.prj（scripts/ge_georef.py）")
    ap.add_argument("--geo-bbox", default="",
                    help="geo-ref 參考底圖 WGS84 bbox: lat_min,lon_min,lat_max,lon_max（未給則試從 --dates-file 的 collection_bbox 帶入）")
    ap.add_argument("--geo-zoom", type=int, default=19, help="geo-ref GMaps 參考底圖 zoom（19≈0.3m/px）")
    ap.add_argument("--no-collapse", dest="collapse", action="store_false",
                    help="不自動點『收合』收起時間軸（預設會收合讓截圖更乾淨）")
    ap.add_argument("--show-labels", dest="hide_labels", action="store_false",
                    help="不關閉地圖標註（預設 Ctrl+B→全部隱藏，移除水道/地名文字避免誤偵測）")
    ap.set_defaults(collapse=True, hide_labels=True)
    args = ap.parse_args()

    dates = [] if args.auto_dates else load_dates(args)
    outdir = Path(args.outdir) / args.site
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[ge_web_capture] site={args.site} mode={'AUTO-DISCOVER' if args.auto_dates else f'{len(dates)} dates'} "
          f"viewport={args.vw}x{args.vh}×{args.dsf} upload={args.upload}")
    print(f"  template cam: @{_split_template(args.template_url)[0]}")

    def _capture(page, ds_yyyymmdd, results):
        session = f"{args.site}_gmap_{ds_yyyymmdd}"
        png = outdir / f"{session}.png"
        page.screenshot(path=str(png))
        sz = png.stat().st_size
        line = f"    ✔ {ds_yyyymmdd} -> {png.name} ({sz:,} bytes)"
        dets = []
        if args.upload:
            st, dets = maybe_upload(png, session, args.model)
            line += f"  | {st}"
        print(line)
        results.append({"date": ds_yyyymmdd, "png": str(png), "bytes": sz, "session": session, "dets": dets})

    from playwright.sync_api import sync_playwright
    results = []
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            args.profile, channel="chrome", headless=False,
            viewport={"width": args.vw, "height": args.vh},
            device_scale_factor=args.dsf,
            chromium_sandbox=True,                       # 啟用沙箱 → 不注入 --no-sandbox → 移除頂部黃色警告橫幅
            ignore_default_args=["--enable-automation"],  # 去掉「Chrome 受自動化軟體控制」資訊列
            # 註：page.screenshot 擷取的是「頁面內容」而非瀏覽器視窗，瀏覽器工具列本就不入鏡；
            # 放大靠 viewport，不需 --start-fullscreen（該旗標與 persistent context 會衝突致關閉）。
            args=[f"--window-position={args.window_pos}",
                  "--disable-features=Translate", "--no-first-run", "--no-default-browser-check",
                  "--disable-infobars"],
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()

        if args.auto_dates:
            _auto_discover(page, args, outdir, results, _capture)
            ctx.close()
            _finish(args, results)
            return

        for i, d in enumerate(dates):
            url = build_url(args.template_url, d)
            print(f"[{i+1}/{len(dates)}] {d} -> goto")
            try:
                page.goto(url, wait_until="load", timeout=90000)
            except Exception as e:
                print(f"    goto warn: {type(e).__name__} (續等載入)")
            page.bring_to_front()                       # 前景才不會被 Chrome 節流 WebGL 串流
            # D：等 app 開機到 semantics 可讀就提早往下走（取代固定長 sleep）。每個日期 goto 都會重載
            #    GE Flutter app，故每次都需重啟 a11y 才能讀進度條/日期列。
            ready_ds = _wait_app_ready(page, floor=6.0 if i == 0 else 3.0,
                                       cap=(6.0 + args.first_wait) if i == 0 else 16.0)
            # 靜止視角 GE 只給低 LOD。真實「拖曳平移」才會觸發 GE 對新視野串高解析 tile
            # （Playwright 的 mouse.wheel 未必到得了 WebGL canvas）。拖出去再拖回、微調 zoom。（load-bearing，保留）
            # --no-pan：近距靜態 nadir 重截（如 ship-recapture-clips）改只靠 goto + 等進度條 100%，
            #           省去平移的殘留位移（讓目標保持畫面正中，利置中裁切）。
            if not args.no_pan:
                try:
                    cx, cy = args.vw // 2, args.vh // 2
                    for dx, dy in [(260, 150), (-260, -150), (-260, 150), (260, -150)]:
                        page.mouse.move(cx, cy)
                        page.mouse.down(); page.mouse.move(cx + dx, cy + dy, steps=12); page.mouse.up()
                        time.sleep(2.5)
                    page.mouse.wheel(0, -160); time.sleep(2.0)   # 微 zoom in 觸發更高 LOD
                    page.mouse.wheel(0, 160);  time.sleep(1.5)
                except Exception:
                    pass
            # A：事件驅動——輪詢進度條到 100% 才截圖，取代 networkidle(GE 幾乎不觸發)+盲等 sleep(args.wait)。
            #    a11y 就緒時 settle 只需覆蓋最後一批 tile 解碼(args.settle)；未就緒則保守退回 max(settle,wait)。
            _wait_tiles(page, args.settle if ready_ds else max(args.settle, args.wait))
            # 註：GE Web 是 Flutter/CanvasKit — UI 全畫在 <canvas>、無 DOM，故彈窗/收合鈕/時間軸
            #     無法用選擇器點（滑鼠座標事件則到得了 canvas，拖曳已用於觸發 tile 串流）。時間軸/UI
            #     在頂部與角落、不擋中央 apron，OBB 偵測不受影響；放大靠 4K viewport。
            session = f"{args.site}_gmap_{d}"
            png = outdir / f"{session}.png"
            page.screenshot(path=str(png))
            sz = png.stat().st_size
            line = f"    ✔ {png.name}  ({sz:,} bytes)"
            dets = []
            if args.upload:
                st, dets = maybe_upload(png, session, args.model)
                line += f"  | {st}"
            print(line)
            results.append({"date": d, "png": str(png), "bytes": sz, "session": session, "dets": dets})
        ctx.close()
    _finish(args, results)


def _finish(args, results):
    import io
    print(f"\n=== 完成：{len(results)} 張 ===")
    for r in results:
        print(f"  {r['date']}  {r['session']}.png  {r['bytes']:,}B")
    if results and not args.upload:
        print("\n未上傳（capture-only）。到 http://127.0.0.1:8060 逐張 📄 上傳，或重跑加 --upload。")
    # 記錄探索到的日期清單（供稽核/下次直接用）
    if args.auto_dates and results:
        outp = Path(args.outdir) / args.site / "_discovered_dates.json"
        io.open(outp, "w", encoding="utf-8").write(json.dumps(
            {"site": args.site, "dates": [r["date"] for r in results]}, ensure_ascii=False, indent=1))
        print(f"探索到 {len(results)} 個歷史日期 → {outp}")

    # 信心度排序報告：把所有偵測候選（不限某型）依『預測類別 + 信心度』彙整排序，
    # 供快速發現各類別的高可信樣本（§11.1 大量發現可信樣本；仍需人工確認才進訓練）。
    if args.upload:
        from collections import defaultdict
        alld = []
        for r in results:
            for d in (r.get("dets") or []):
                alld.append({"date": r["date"], "session": r["session"], "det_id": d.get("det_id"),
                             "pred": d.get("pred"), "conf": round(float(d.get("conf", 0)), 4),
                             "obb_conf": d.get("obb_conf"), "source": d.get("source"),
                             "review_status": d.get("review_status"), "chip_url": d.get("chip_url"),
                             "seed_votes": d.get("seed_votes")})
        alld.sort(key=lambda x: -x["conf"])
        byc = defaultdict(list)
        for d in alld:
            byc[d["pred"]].append(d)
        by_class = {}
        for c, v in sorted(byc.items(), key=lambda kv: -max(x["conf"] for x in kv[1])):
            by_class[c] = {"n": len(v), "max_conf": v[0]["conf"],
                           "n_high_ge85": sum(1 for x in v if x["conf"] >= 0.85),
                           "n_review_50_85": sum(1 for x in v if 0.5 <= x["conf"] < 0.85),
                           "chips_top": [x["chip_url"] for x in v[:5]]}
        report = {"site": args.site, "model_hint": args.model or "server-latest",
                  "total_dets": len(alld), "by_class": by_class, "ranked": alld}
        outp = Path(args.outdir) / args.site / "_ranked_detections.json"
        io.open(outp, "w", encoding="utf-8").write(json.dumps(report, ensure_ascii=False, indent=1))
        print(f"\n信心度排序報告（{len(alld)} 偵測）→ {outp}")
        print("=== 各類別 by 最高信心（大量發現可信樣本用；仍需人工確認）===")
        for c, s in by_class.items():
            print(f"  {c:22} n={s['n']:4}  max={s['max_conf']:.2f}  high(≥.85)={s['n_high_ge85']:3}  review(.5-.85)={s['n_review_50_85']:3}")
        print("治理：全部為未複核候選（§2）。高信心≠ground truth；蕪湖/閻良混雜多型，登記前仍須逐框確認機翼形狀(§6.5)。")

    # 地理定位：以 GMaps 為參考、SIFT 對位最新日期 → 逐日期輸出 .jpg/.jgw/.prj（§14 幾何、fail-closed）。
    if getattr(args, "geo_ref", False) and results:
        bbox = None
        if args.geo_bbox:
            try:
                bbox = tuple(float(x) for x in args.geo_bbox.split(","))
                assert len(bbox) == 4
            except Exception:
                bbox = None
        if bbox is None and args.dates_file:
            try:
                import ge_georef as _gg
                bbox = _gg._bbox_from_spec(args.dates_file)
            except Exception:
                bbox = None
        if bbox is None:
            print("[geo-ref] 略過：未提供 --geo-bbox，且 --dates-file 無 collection_bbox（fail-closed，不臆測地理範圍）。")
        else:
            try:
                import ge_georef as _gg
                _gg.georef_capture_dir(Path(args.outdir) / args.site, bbox, zoom=args.geo_zoom)
            except Exception as ex:
                print(f"[geo-ref] 失敗（不影響已擷取的 PNG）：{type(ex).__name__}: {ex}")


def _auto_discover(page, args, outdir, results, capture_fn):
    """啟用 Flutter a11y → 走到最早日期 → 逐日期(較新)截圖直到最新。日期由 semantics 樹讀取。"""
    print("[auto] 載入 template + 啟用無障礙 semantics…")
    try:
        page.goto(args.template_url, wait_until="load", timeout=90000)
    except Exception as e:
        print(f"    goto warn: {type(e).__name__}")
    page.bring_to_front()
    time.sleep(6 + args.first_wait)
    _enable_a11y(page)
    time.sleep(4)
    # 首次以拖曳觸發 tile 串流（之後靠進度條輪詢）
    if not args.no_pan:
        try:
            cx, cy = args.vw // 2, args.vh // 2
            for dx, dy in [(240, 140), (-240, -140)]:
                page.mouse.move(cx, cy)
                page.mouse.down(); page.mouse.move(cx + dx, cy + dy, steps=10); page.mouse.up()
                time.sleep(2.0)
        except Exception:
            pass

    ds0, nxt, prv = _read_stepper(page)
    if ds0 is None:
        # search/一般 URL 未在歷史模式 → 點『顯示歷史圖像』進入
        if _enter_historical(page):
            print("[auto] 已點『顯示歷史圖像』進入歷史模式…")
            time.sleep(6)
            ds0, nxt, prv = _read_stepper(page)
    if ds0 is None:
        print("[auto] ✘ 讀不到歷史日期列——確認 URL 能進歷史影像模式、a11y 已啟用。改用 --dates。")
        return
    # 關閉地圖標註圖層（Ctrl+B → 全部隱藏）：移除水道/地名文字，避免被下游偵測誤當目標
    if getattr(args, "hide_labels", True):
        if _hide_annotations(page):
            print("[auto] 已關閉地圖標註圖層（Ctrl+B → 全部隱藏）")
        else:
            print("[auto] 未找到『全部隱藏』（可能面板未開/已關）——不影響擷取")
    # 收合時間軸讓截圖更乾淨；若收合後讀不到日期列(stepper 被藏)則切回展開(fail-safe)
    if getattr(args, "collapse", True) and _collapse_timeline(page):
        time.sleep(2.0)
        tds, _, _ = _read_stepper(page)
        if tds:
            print("[auto] 已收合時間軸（截圖更乾淨，日期列仍可讀）")
        else:
            _collapse_timeline(page); time.sleep(2.0)      # 再點一次切回展開
            print("[auto] 收合後讀不到日期列 → 還原展開（不收合）")
    print(f"[auto] 目前日期 {ds0}；往回走到最早…")
    # 走到最早：一直點 prev 直到日期不再變
    last = None
    for _ in range(args.max_dates):
        ds, nxt, prv = _read_stepper(page)
        if not prv or ds == last:
            break
        last = ds
        page.mouse.click(*prv)
        time.sleep(2.2)
    ds, nxt, prv = _read_stepper(page)
    print(f"[auto] 最早日期 {ds}")
    # 跳過最早的 N 張 Landsat（使用者提示：GE 最早一張通常是 Landsat 資源衛星，解析度太粗、無機體細節）
    for _ in range(max(0, args.skip_first)):
        ds, nxt, prv = _read_stepper(page)
        if not nxt:
            break
        print(f"[auto] 跳過最早(Landsat) {ds}")
        page.mouse.click(*nxt)
        time.sleep(2.2)
    ds, nxt, prv = _read_stepper(page)
    print(f"[auto] 從 {ds} 開始逐日期截圖…")

    # 逐日期往新走、每個截圖
    captured_last = None
    for _ in range(args.max_dates + 5):
        ds, nxt, prv = _read_stepper(page)
        if not ds:
            break
        if ds != captured_last:
            _wait_tiles(page, args.wait)               # 等該日期 tiles 載入
            ds2, nxt, prv = _read_stepper(page)        # 等待後重讀(視角穩定)
            ds = ds2 or ds
            capture_fn(page, ds.replace("-", ""), results)
            captured_last = ds
        if not nxt:
            break
        page.mouse.click(*nxt)                          # → 較新的圖片(下一個日期)
        time.sleep(2.2)
        nd, _, _ = _read_stepper(page)
        if nd == ds:                                    # 下一個沒變 → 已到最新
            break


if __name__ == "__main__":
    main()
