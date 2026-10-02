# -*- coding: utf-8 -*-
"""
產生「人工覆核工作單」：一份離線 HTML，讓覆核人員逐點目視判讀，匯出 data/ardswc_hotspots/ledger.json。

用途（2026-09-22 審查意見）：A 級／優先現勘候選點在決賽前全部完成人工覆核，並把自動化結果與人工判讀
明確分開。本腳本**只呈現證據、不預填任何結論**——判讀欄位一律空白，由覆核人員填寫。

兩組清單：
  1. A 級候選（目前自動分級為 A 者）——要覆核的主要對象。
  2. 對照組：「只依複發性排序」取與 A 級同樣多的前 N 名中、不在 A 級者。量化驗證（/api/validation）
     要回答「加入影像與地形因子後，是否比單純依複發次數更準」，需要兩組都有覆核結果才算得出來。

判讀選項與 ledger.json 現有欄位一致：verdict ∈ ok(可信)／warn(需複查)／bad(偽陽性)；另可勾
treated（已完成治理工程）、improved（現況已改善），這兩項會把 A／B 級降為 C（見 app.py _priority_for）。
覆核標準（沿用 README「系統性陷阱」）：只有在「前後期影像上能指認出明確地貌變化（新崩塌／裸露／
河道改道／大面積開發）」時才標 ok；對位失敗、雲影、季節或色調差異、市區 10 m 解析度紋理雜亂造成的
高分一律不算。看不出變化就標 warn 或 bad，不要為了「系統看起來有用」而放寬。

用法：python scripts/review_queue.py [--out review/review_queue.html]
輸出頁面用相對路徑引用 data/ge_captures 下的面板圖，請保留在倉庫內開啟（預設輸出到 review/）。
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "ardswc_hotspots"
CAPTURES = REPO / "data" / "ge_captures"
sys.path.insert(0, str(REPO / "webapp" / "change_detect_viewer"))
sys.path.insert(0, str(REPO / "scripts"))


def _load(p, default):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def build_items(out_dir):
    import app as A                                     # 沿用線上同一份分級邏輯，清單與網站永遠一致
    hotspots = _load(DATA / "top100_consolidated.json", [])
    by_rank = {h["rank"]: h for h in hotspots}
    pr = {p["rank"]: p for p in A._compute_priority()["items"]}
    a_ranks = sorted(r for r, p in pr.items() if p["tier"] == "A")
    base_top = [h["rank"] for h in sorted(hotspots, key=lambda h: (-(A._recurrence(h) or 0), h["rank"]))][:len(a_ranks)]
    base_only = [r for r in base_top if r not in a_ranks]
    relief = A._terrain_relief_by_rank()
    uav = _uav_nearby(hotspots)
    rel = Path("..") / "data" / "ge_captures"          # 輸出檔在 review/ 時的相對路徑
    try:
        rel = Path(__import__("os").path.relpath(CAPTURES, out_dir)).as_posix()
    except ValueError:
        rel = str(CAPTURES)

    def item(rank, group):
        h, p = by_rank[rank], pr[rank]
        site = f"ardswc_top{rank:02d}"
        tl = _load(CAPTURES / site / "_change_detect" / f"{site}_change_timeline.json", {"pairs": []})
        pairs = []
        for q in tl.get("pairs", []):
            tag = f"{site}_diff_{q['date_a']}_{q['date_b']}"
            dj = _load(CAPTURES / site / "_change_detect" / f"{tag}.json", {})
            boxes = []
            for rg in dj.get("regions") or []:                       # 自動 SSIM 變遷候選區塊（bbox_lonlat 四角 → [W,S,E,N]）
                bb = rg.get("bbox_lonlat")
                if bb:
                    lons, lats = [c[0] for c in bb], [c[1] for c in bb]
                    boxes.append([round(min(lons), 6), round(min(lats), 6), round(max(lons), 6), round(max(lats), 6),
                                  round(rg.get("mean_change_score") or 0, 2)])
            pairs.append({"a": q["date_a"], "b": q["date_b"], "frac": q.get("overall_change_fraction"), "boxes": boxes,
                          "img": {k: f"{rel}/{site}/_change_detect/{tag}_{k}.jpg"
                                  for k in ("before", "after", "overlay", "heat")}})
        dr = p.get("dramatic_pair") or {}
        al = {}
        if dr:
            d = _load(CAPTURES / site / "_change_detect" / f"{site}_diff_{dr['date_a']}_{dr['date_b']}.json", {})
            al = d.get("alignment") or {}
        f = p["factors"]
        return {
            "rank": rank, "group": group, "site": site,
            "place": f"{h.get('county') or ''}{h.get('district') or ''}",
            "lat": h["lat"], "lon": h["lon"],
            "tier": p["tier"], "factors": {
                "recurrence": f["recurrence"]["value"], "recurrence_band": f["recurrence"]["band"],
                "change": f["recent_change"]["value"], "change_band": f["recent_change"]["band"],
                "relief_m": f["terrain_relief_m"]["value"], "confidence": f["confidence"]["band"]},
            "events": [{"y": e["year"], "n": e["name"], "r": e["n_records"]} for e in h.get("events", [])],
            "pairs": pairs, "default_pair": next((i for i, q in enumerate(pairs)
                                                  if dr and q["a"] == dr["date_a"] and q["b"] == dr["date_b"]), 0),
            "alignment": {"applied": al.get("applied"), "uncertain": al.get("uncertain"), "shift_px": al.get("shift_px")},
            "uav": uav.get(rank, []),
            "ev_dates": [{"d": e["first_date"], "n": e["name"]} for e in h.get("events", []) if e.get("first_date")],
            "relief_m": (relief.get(rank) or {}).get("relief_m"),
        }

    return ([item(r, "A") for r in a_ranks], [item(r, "base") for r in base_only])


def _uav_nearby(hotspots, max_m=1500):
    import math
    out = {}
    for fp in sorted((REPO / "webapp" / "change_detect_viewer" / "static" / "uav").glob("*/meta.json")):
        try:
            m = json.loads(fp.read_text(encoding="utf-8"))
            lon, lat = m["center_lonlat"]
        except (OSError, ValueError, KeyError):
            continue
        for h in hotspots:
            d = math.hypot((lon - h["lon"]) * 111320 * math.cos(math.radians(lat)), (lat - h["lat"]) * 110540)
            if d <= max_m:
                out.setdefault(h["rank"], []).append({"id": m["id"], "title": m.get("title", ""), "dist_m": round(d)})
    return out


NLSC_YEARS = [65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 78, 79, 80, 81, 83, 84] + list(range(86, 115))   # TOPO05KPHOTO_NNN（民國年）
NLSC_TILE = "https://wmts.nlsc.gov.tw/wmts/{layer}/default/GoogleMapsCompatible/{z}/{y}/{x}"
NLSC_CACHE = REPO / "review" / "nlsc_coverage_cache.json"


def _nlsc_probe(lat, lon, z=17):
    """1/5000 相片基本圖每年只涵蓋部分圖幅：逐年探測熱點中心圖磚，空圖磚（約 300–900 位元組的透明 PNG）視為無資料。
    回傳有資料的民國年清單。"""
    import math
    from concurrent.futures import ThreadPoolExecutor
    from urllib.request import Request, urlopen
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)

    def one(yr):
        url = NLSC_TILE.format(layer=f"TOPO05KPHOTO_{yr:03d}", z=z, y=y, x=x)
        for _ in range(2):
            try:
                return yr, len(urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30).read()) > 3000
            except Exception:  # noqa: BLE001
                pass
        return yr, None                                     # 探測失敗：不當作無資料，前端仍列出
    with ThreadPoolExecutor(max_workers=12) as ex:
        res = list(ex.map(one, NLSC_YEARS))
    return [yr for yr, ok in res if ok or ok is None]


def _nlsc_years(items_ranks, hotspots_by_rank, probe=True):
    cache = _load(NLSC_CACHE, {})
    out = {}
    for r in items_ranks:
        h = hotspots_by_rank[r]
        key = f"{r}:{h['lat']:.5f},{h['lon']:.5f}"
        if key not in cache:
            cache[key] = _nlsc_probe(h["lat"], h["lon"]) if probe else NLSC_YEARS
            NLSC_CACHE.write_text(json.dumps(cache), encoding="utf-8")
        out[r] = cache[key]
    return out


WB_CACHE = REPO / "review" / "wayback_frames_cache.json"


def _wayback_frames(items, hotspots_by_rank):
    """每個熱點的 Esri Wayback「不同拍攝日」影像（wayback_assist.list_frames，同拍攝日只留一版）；結果快取。"""
    sys.path.insert(0, str(REPO / "webapp" / "change_detect_viewer"))
    import wayback_assist as WB
    from concurrent.futures import ThreadPoolExecutor
    cache = _load(WB_CACHE, {})

    def one(r):
        h = hotspots_by_rank[r]
        key = f"{r}:{h['lat']:.5f},{h['lon']:.5f}"
        if key in cache:
            return r, cache[key]
        try:
            fr = [{"release": f["release"], "date": f["date"], "kind": f["date_kind"], "res": f.get("res_m")}
                  for f in WB.list_frames(h["lat"], h["lon"], 17)]
        except Exception as e:  # noqa: BLE001
            print(f"  #{r} Wayback 清單失敗：{type(e).__name__}", flush=True)
            return r, None
        cache[key] = fr
        WB_CACHE.write_text(json.dumps(cache), encoding="utf-8")
        return r, fr
    with ThreadPoolExecutor(max_workers=4) as ex:
        return dict(ex.map(one, [it["rank"] for it in items]))


def _timeline(nlsc_years, wb_frames):
    """合併成同一條時間軸：1/5000 相片基本圖的年份定位為『該年 1 月 1 日』（西元 = 民國 + 1911）；
    Wayback 用影像拍攝日（查不到拍攝日者用發布日並標 kind）。同日時 NLSC 排在前。"""
    tl = [{"t": "nlsc", "y": y, "date": f"{y + 1911}0101"} for y in nlsc_years]
    tl += [{"t": "wb", "rel": f["release"], "date": f["date"], "kind": f["kind"], "res": f["res"]} for f in (wb_frames or [])]
    return sorted(tl, key=lambda e: (e["date"], e["t"] != "nlsc"))


HTML = r"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>人工覆核工作單</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js"></script>
<style>
:root{--bg:#fafaf7;--fg:#222;--mut:#666;--card:#fff;--bd:#d8d6cf;--ok:#2e7d32;--warn:#b26a00;--bad:#b3261e;--acc:#B85C1E}
@media(prefers-color-scheme:dark){:root{--bg:#1b1b19;--fg:#e8e6df;--mut:#a09d94;--card:#262522;--bd:#44413b}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,"Noto Sans TC",sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--bd);padding:10px 16px;z-index:5;display:flex;gap:16px;flex-wrap:wrap;align-items:center}
main{max-width:1400px;margin:0 auto;padding:16px}
.rules{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:12px 16px;margin-bottom:16px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:8px;margin:0 0 20px;padding:14px 16px}
.card.done{border-left:5px solid var(--ok)}
.meta{color:var(--mut);font-size:13px}
.imgs{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin:8px 0}
.imgs figure{margin:0}.imgs img{width:100%;display:block;border:1px solid var(--bd);cursor:zoom-in}
.imgs figcaption{font-size:12px;color:var(--mut)}
@media(max-width:800px){.imgs{grid-template-columns:repeat(2,1fr)}}
.cltab{border-collapse:collapse;margin:8px 0;font-size:13px}.cltab th,.cltab td{border:1px solid var(--bd);padding:3px 10px;text-align:center}.cltab th{background:var(--bg);font-weight:600}
.ctl{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-top:8px}
textarea{width:100%;min-height:52px;background:var(--bg);color:var(--fg);border:1px solid var(--bd);border-radius:6px;padding:6px;font:inherit}
button{background:var(--acc);color:#fff;border:0;border-radius:6px;padding:6px 14px;font:inherit;cursor:pointer}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;border:1px solid var(--bd);font-size:12px;margin-right:6px}
.warnbox{color:var(--bad)}
h2{margin:28px 0 8px}
header .site-badge .sb-qr svg{width:46px;height:46px;padding:2px}header .site-badge .sb-team{font-size:10px}
details.nl{margin:10px 0;border:1px solid var(--bd);border-radius:8px;padding:6px 10px;background:var(--bg)}
details.nl>summary{cursor:pointer;font-weight:600}
.nlmaps{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px}
.nlmap{height:420px;border:1px solid var(--bd);border-radius:6px}
.nlcap{font-size:12px;color:var(--mut);margin:2px 0}
.nlbar{display:flex;gap:8px;align-items:center;margin-top:8px;flex-wrap:wrap}.nlbar input[type=range]{flex:1;min-width:200px}
.nlbar button{padding:2px 10px}.nlyear{font-weight:700;min-width:210px}
@media(max-width:800px){.nlmaps{grid-template-columns:1fr}}
#lightbox{display:none;position:fixed;inset:0;background:#000d;z-index:20;align-items:center;justify-content:center}
#lightbox img{max-width:96vw;max-height:96vh}
</style></head><body>
<header><b>人工覆核工作單</b><span id="prog"></span><button id="exp">匯出 ledger.json</button>
<span class="meta">判讀只存在你的瀏覽器（localStorage），匯出後覆蓋 data/ardswc_hotspots/ledger.json 才生效。</span><div id="site-badge" style="margin-left:auto"></div></header>
<script>__QRJS__</script>
<script>__BADGEJS__</script>
<main>
<div class="rules"><b>覆核標準：</b>只有在<u>前後期影像上能指認出明確地貌變化</u>（新崩塌／裸露／河道改道／大面積開發）才標「可信」。
對位失敗、雲影、季節或色調差異、市區 10 m 解析度紋理雜亂造成的高分，一律不算。看不出變化就標「需複查」或「偽陽性」。
工作單不預填任何結論；SSIM 分數只是影像變化候選訊號，不等於災害或風險。
可勾「已完成治理工程／現況已改善」——這會把 A／B 級改列 C（持續監測）。</div>
<div id="root"></div></main>
<div id="lightbox" onclick="this.style.display='none'"><img></div>
<script>
const A_ITEMS=__A__, BASE_ITEMS=__BASE__, INIT=__INIT__, CLIM=__CLIM__, OFFICIAL=__OFF__;
const KEY="review_queue_v1";
let st={}; try{st=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
INIT.forEach(l=>{ if(!st[l.rank]) st[l.rank]={verdict:l.verdict,treated:!!l.treated,improved:!!l.improved,note:l.note||"",pair:null,ts:l.reviewed_at||null}; });
const LABEL={ok:"可信",warn:"需複查",bad:"偽陽性"};
function save(){try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){}prog()}
function prog(){const done=r=>st[r.rank]&&st[r.rank].verdict;
  document.getElementById("prog").textContent=`A 級 ${A_ITEMS.filter(done).length}/${A_ITEMS.length}　對照組 ${BASE_ITEMS.filter(done).length}/${BASE_ITEMS.length}`;
  document.querySelectorAll(".card").forEach(c=>c.classList.toggle("done",!!(st[c.dataset.rank]&&st[c.dataset.rank].verdict)))}
function fmt(d){return d.slice(0,4)+"-"+d.slice(4,6)+"-"+d.slice(6)}


// ── 降雨／土壤濕度背景（data/climate_series.json：IMERG 日雨量、SMAP L4 根系層含水量；單點最近像元）──
// ── 官方衛星判釋崩塌（BigGIS；衛星影像判釋，非現地確認）：熱點 1.5 km 內的多邊形；落在「兩期影像之間」的事件標紅 ──
function officialPanel(it,getPi){
  const d=document.createElement("details");d.className="nl of";d.open=true;
  const polys=OFFICIAL&&OFFICIAL.ranks&&OFFICIAL.ranks[it.rank];
  if(!OFFICIAL||!polys){d.innerHTML=`<summary>官方衛星判釋崩塌（BigGIS）</summary><div class="meta">未嵌入官方判釋資料（執行 scripts/fetch_biggis_interp.py 後重建本頁）。</div>`;return d}
  d.innerHTML=`<summary>官方衛星判釋崩塌（BigGIS，1.5 km 內共 ${polys.length} 塊）——兩期影像之間的事件標紅</summary><div class="ofbody"></div>`;
  const body=d.querySelector(".ofbody");
  const area=xy=>{let s=0;for(let i=0;i<xy.length;i+=2){const j=(i+2)%xy.length;s+=xy[i]*xy[j+1]-xy[j]*xy[i+1]}return Math.abs(s)/2/10000};
  function draw(){
    const q=it.pairs[getPi()];if(!q){body.innerHTML='<div class="meta">沒有比對期別</div>';return}
    const R=1500,W=320,sc=W/(2*R),ns="http://www.w3.org/2000/svg";
    const svg=document.createElementNS(ns,"svg");svg.setAttribute("viewBox",`0 0 ${W} ${W}`);svg.setAttribute("width",W);svg.style.background="#f3f1ea";svg.style.border="1px solid #999";
    const mk=(t,a)=>{const e=document.createElementNS(ns,t);for(const k in a)e.setAttribute(k,a[k]);svg.appendChild(e);return e};
    [-1000,-500,0,500,1000].forEach(v=>{mk("line",{x1:W/2+v*sc,y1:0,x2:W/2+v*sc,y2:W,stroke:"#d6d2c4","stroke-width":.6});mk("line",{x1:0,y1:W/2-v*sc,x2:W,y2:W/2-v*sc,stroke:"#d6d2c4","stroke-width":.6})});
    mk("circle",{cx:W/2,cy:W/2,r:1000*sc,fill:"none",stroke:"#8aa","stroke-dasharray":"4 3","stroke-width":.8});
    const win={};let nIn=0;
    polys.forEach(([ei,xy])=>{
      const ed=OFFICIAL.events[ei][0].replace(/-/g,"");const inW=ed>q.a&&ed<=q.b;
      let pts="";for(let i=0;i<xy.length;i+=2)pts+=`${(W/2+xy[i]*sc).toFixed(1)},${(W/2-xy[i+1]*sc).toFixed(1)} `;
      const e=mk("polygon",{points:pts,fill:inW?"#d73027":"#6b7fa3","fill-opacity":inW?.6:.3,stroke:inW?"#a50f15":"#3a4a6b","stroke-width":inW?.9:.5});
      const t=document.createElementNS(ns,"title");t.textContent=`${OFFICIAL.events[ei][0]} ${OFFICIAL.events[ei][1]}（約 ${area(xy).toFixed(2)} 公頃）`;e.appendChild(t);
      if(inW){nIn++;const w=win[ei]||(win[ei]={n:0,ha:0});w.n++;w.ha+=area(xy)}
    });
    mk("line",{x1:W/2-7,y1:W/2,x2:W/2+7,y2:W/2,stroke:"#000","stroke-width":1.4});mk("line",{x1:W/2,y1:W/2-7,x2:W/2,y2:W/2+7,stroke:"#000","stroke-width":1.4});
    const tx=mk("text",{x:6,y:W-6,"font-size":9,fill:"#555"});tx.textContent="北在上；格線 500 m；虛線圓＝1 km；＋＝熱點位置";
    const evs=Object.entries(win).map(([ei,w])=>`${OFFICIAL.events[ei][0]} ${OFFICIAL.events[ei][1].replace(/^[0-9]{8}_/,"")}：${w.n} 塊、約 ${w.ha.toFixed(1)} 公頃`);
    body.innerHTML=`<div class="meta">兩期影像：${fmt(q.a)} → ${fmt(q.b)}。<b>${nIn?`期間內有 ${evs.length} 個官方判釋事件落在 1.5 km 內（紅色）：${evs.join("；")}。`:`期間內沒有官方判釋事件落在 1.5 km 內。`}</b></div>`;
    body.appendChild(svg);
    const n=document.createElement("div");n.className="meta";
    n.innerHTML="判讀提示：紅色＝這兩期之間有官方衛星判釋出的新增崩塌，可作為「兩期影像的新裸露是崩塌」的獨立佐證；藍灰＝其他時間的官方判釋。<b>這是官方衛星判釋，不是現地確認；官方只判釋颱風、豪雨、地震等大事件，所以「沒有紅色」不代表沒有崩塌。</b>資料：農業部農村發展及水土保持署 BigGIS（"+OFFICIAL.fetched+" 抓取；2017 年前的期別沒有 Sentinel-2 比對）。";
    body.appendChild(n)}
  d.refresh=()=>{if(d.open)draw()};d.addEventListener("toggle",()=>{if(d.open)draw()});draw();return d}

function climatePanel(it,getPi){
  const c=CLIM&&CLIM.ranks&&CLIM.ranks[it.rank];
  const d=document.createElement("details");d.className="nl cl";
  if(!c){d.innerHTML=`<summary>降雨／土壤濕度</summary><div class="meta">此熱點無雨量資料（執行 scripts/climate_series.py 產生 data/climate_series.json 後重建本頁）</div>`;return d}
  const t0=Date.parse(CLIM.start+"T00:00:00Z"), DAY=86400000, N=c.rain.length;
  const idx=s8=>Math.round((Date.parse(s8.slice(0,4)+"-"+s8.slice(4,6)+"-"+s8.slice(6)+"T00:00:00Z")-t0)/DAY);
  const idxIso=s10=>Math.round((Date.parse(s10+"T00:00:00Z")-t0)/DAY);
  const dateOf=i=>new Date(t0+i*DAY).toISOString().slice(0,10);
  d.innerHTML=`<summary>降雨／土壤濕度（兩期影像之間）</summary><div class="clbody"></div>`;
  const body=d.querySelector(".clbody");
  function stats(i0,i1){
    const r=[];for(let i=i0;i<=i1;i++)r.push(c.rain[i]>=0?c.rain[i]/10:null);
    const ok=r.filter(v=>v!=null), tot=ok.reduce((a,b)=>a+b,0);
    const roll=k=>{let m=0,w=0;for(let i=0;i<r.length;i++){w+=r[i]||0;if(i>=k)w-=r[i-k]||0;if(w>m)m=w}return m};
    // 警報次數：3 日累積 ≥100 mm 的日子，相鄰（間隔 ≤2 天）合併為一次
    let eps=0,last=-9,w=0;for(let i=0;i<r.length;i++){w+=r[i]||0;if(i>=3)w-=r[i-3]||0;if(w>=100){if(i-last>2)eps++;last=i}}
    const sm=[];for(let i=i0;i<=i1;i++)if(c.sm[i]>=0)sm.push(c.sm[i]/1000);
    return {n:r.length,miss:r.length-ok.length,tot,max1:Math.max(0,...ok),max3:roll(3),max7:roll(7),eps,
      smN:sm.length,smMean:sm.length?sm.reduce((a,b)=>a+b,0)/sm.length:null,smMax:sm.length?Math.max(...sm):null}}
  function render(){
    const q=it.pairs[getPi()];if(!q){body.innerHTML='<div class="meta">沒有比對期別</div>';return}
    const i0=Math.max(0,idx(q.a)), i1=Math.min(N-1,idx(q.b));
    if(i1<=i0){body.innerHTML=`<div class="meta">兩期影像日期（${fmt(q.a)} → ${fmt(q.b)}）不在資料範圍（${CLIM.start} ～ ${CLIM.last_rain_date}）</div>`;return}
    const S=stats(i0,i1), n=i1-i0+1, bucket=Math.max(1,Math.ceil(n/450));
    const W=900,H=170,L=44,R=40,T=10,B=22, pw=W-L-R, ph=H-T-B;
    const bars=[];let mx=1;
    for(let i=i0;i<=i1;i+=bucket){let sum=0;for(let k=i;k<Math.min(i+bucket,i1+1);k++)sum+=c.rain[k]>=0?c.rain[k]/10:0;bars.push([i,sum]);if(sum>mx)mx=sum}
    const X=i=>L+(i-i0)/(n-1||1)*pw, Y=v=>T+ph-v/mx*ph, bw=Math.max(1,pw/bars.length-0.5);
    const smY=v=>T+ph-v/0.6*ph;
    let path="",pen=false;
    for(let i=i0;i<=i1;i++){if(c.sm[i]>=0){path+=(pen?"L":"M")+X(i).toFixed(1)+" "+smY(c.sm[i]/1000).toFixed(1)+" ";pen=true}else pen=false}
    const evs=(it.ev_dates||[]).map(e=>({i:idxIso(e.d),n:e.n,d:e.d})).filter(e=>e.i>=i0&&e.i<=i1);
    const ticks=[];const y0=+dateOf(i0).slice(0,4),y1=+dateOf(i1).slice(0,4);
    for(let y=y0;y<=y1+1;y++){const i=idxIso(y+"-01-01");if(i>=i0+n*0.06&&i<=i1-n*0.06)ticks.push([i,y])}
    body.innerHTML=`<div class="meta">${fmt(q.a)} → ${fmt(q.b)}（${n} 天）　雨量：NASA GPM IMERG（0.1°，逐日）；土壤濕度：SMAP L4 根系層 0–100 cm（9 km，m³/m³）${bucket>1?`　<b>長條＝每 ${bucket} 日合計</b>`:""}</div>
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="兩期影像之間的逐日雨量長條與土壤含水量折線" style="width:100%;height:auto;max-height:200px;background:var(--bg);border:1px solid var(--bd)">
      ${[0,.5,1].map(f=>`<line x1="${L}" x2="${W-R}" y1="${T+ph*(1-f)}" y2="${T+ph*(1-f)}" stroke="#8884" stroke-width="1"/><text x="${L-4}" y="${T+ph*(1-f)+4}" text-anchor="end" font-size="10" fill="#888">${(mx*f).toFixed(0)}</text><text x="${W-R+4}" y="${T+ph*(1-f)+4}" font-size="10" fill="#4a90d9">${(0.6*f).toFixed(2)}</text>`).join("")}
      ${bars.map(([i,v])=>v>0?`<rect x="${(X(i)-bw/2).toFixed(1)}" y="${Y(v).toFixed(1)}" width="${bw.toFixed(1)}" height="${(T+ph-Y(v)).toFixed(1)}" fill="${v>=80?'#d98f35':'#6fa8dc'}"><title>${dateOf(i)}${bucket>1?"起":""}：${v.toFixed(1)} mm</title></rect>`:"").join("")}
      <path d="${path}" fill="none" stroke="#2b6cb0" stroke-width="1.6"/>
      ${evs.map(e=>`<line x1="${X(e.i)}" x2="${X(e.i)}" y1="${T}" y2="${T+ph}" stroke="#c0392b" stroke-dasharray="4 3"><title>通報事件 ${e.d} ${e.n}</title></line>`).join("")}
      ${ticks.map(([i,y])=>`<text x="${X(i)}" y="${H-6}" font-size="10" fill="#888" text-anchor="middle">${y}</text>`).join("")}
      <text x="${L}" y="${H-6}" font-size="10" fill="#888">${dateOf(i0)}</text><text x="${W-R}" y="${H-6}" font-size="10" fill="#888" text-anchor="end">${dateOf(i1)}</text>
      <text x="4" y="12" font-size="10" fill="#888">mm</text><text x="${W-4}" y="12" font-size="10" fill="#4a90d9" text-anchor="end">SM</text>
    </svg>
    <div class="meta">圖例：長條＝雨量（橘＝單日（或區間合計）≥80 mm，約為氣象署「大雨」等級以上）；藍線＝根系層含水量；紅虛線＝本熱點的通報事件日。</div>
    <table class="cltab"><tr><th>期間總雨量</th><th>單日最大</th><th>3 日最大</th><th>7 日最大</th><th>3 日 ≥100 mm 次數</th><th>含水量 均值／最大</th></tr>
    <tr><td>${S.tot.toFixed(0)} mm</td><td>${S.max1.toFixed(0)} mm</td><td>${S.max3.toFixed(0)} mm</td><td>${S.max7.toFixed(0)} mm</td><td>${S.eps}</td>
    <td>${S.smN?`${S.smMean.toFixed(3)} ／ ${S.smMax.toFixed(3)}`:"無資料"}</td></tr></table>
    <div class="meta">${S.miss?`雨量缺 ${S.miss} 天。`:""}${S.smN<n?`SMAP L4 在 GEE 的資料只到 ${CLIM.last_sm_date}，此期間有 ${n-S.smN} 天無含水量。`:""}　判讀提示：兩期影像之間若有 3 日 ≥100 mm 的豪雨（本系統歷史通報中，此級雨量後事件率約為無雨日的 10–80 倍），新裸露較可能是崩塌；若只有乾季或無豪雨，應優先排除季節、河床擺動與工程。解析度 0.1°／9 km，山區單點代表性有限，僅供脈絡，非變遷證據。</div>`}
  d.addEventListener("toggle",()=>{if(d.open)render()});d.refresh=()=>{if(d.open)render()};
  return d}

const NL_TILE=l=>`https://wmts.nlsc.gov.tw/wmts/${l}/default/GoogleMapsCompatible/{z}/{y}/{x}`;
const WB_TILE=r=>`https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/${r}/{z}/{y}/{x}`;
const WB_ATTR='Esri World Imagery Wayback';
const NL_ATTR='&copy; 內政部國土測繪中心 NLSC';
function nlscPanel(it,getPi){
  const ys=it.timeline||[], nN=ys.filter(e=>e.t==="nlsc").length, nW=ys.length-nN;
  const d=document.createElement("details");d.className="nl";
  d.innerHTML=`<summary>NLSC 線上比對：1/5000 相片基本圖（年度滑桿）＋ 通用版正射影像（混合）</summary>
  <div class="nlbar"><button class="pv">◀</button><input type="range" class="sl" min="0" max="${Math.max(0,ys.length-1)}" value="${Math.max(0,ys.length-1)}" ${ys.length?"":"disabled"}><button class="nx">▶</button>
  <span class="nlyear"></span><label class="meta"><input type="checkbox" class="bx1" checked> 左圖變遷候選框</label><label class="meta"><input type="checkbox" class="bx2" checked> 右圖變遷候選框</label><span class="meta">${ys.length?`時間軸共 ${ys.length} 格：1/5000 相片基本圖 ${nN} 個年份（定位為當年 1/1）＋ Esri Wayback ${nW} 個拍攝日`:"此地點無 1/5000 相片基本圖與 Wayback 資料"}</span></div>
  <div class="nlmaps"><div><div class="nlcap">左：1/5000 相片基本圖 ＋ Esri Wayback（共用滑桿；<span class="nly"></span>）</div><div class="nlmap m1"></div></div>
  <div><div class="nlcap">右：通用版正射影像（混合，現況）</div><div class="nlmap m2"></div></div></div>
  <div class="nlcap">兩張地圖同步縮放平移；紅框＝工作單影像範圍約 1.5 km 見方；<span style="color:#1a9b4a">綠框＝自動 SSIM 變遷候選區塊</span>（隨上方「比對期別」切換，<span class="bxn"></span>；僅為候選訊號，需人工確認）。1/5000 相片基本圖每年只涵蓋部分圖幅，空白處代表該年無資料（已依熱點中心預先篩掉無資料年份）。</div>`;
  let inited=false;
  d.addEventListener("toggle",()=>{
    if(!d.open||inited)return;inited=true;
    if(typeof L==="undefined"){d.querySelector(".nlmaps").innerHTML='<div class="warnbox">Leaflet 載入失敗（需要連網取 cdnjs）。</div>';return}
    const c=[it.lat,it.lon], dl=750/111320, dn=750/(111320*Math.cos(it.lat*Math.PI/180));
    const mk=el=>{const m=L.map(el,{center:c,zoom:17,maxZoom:19,zoomControl:true});
      L.rectangle([[it.lat-dl,it.lon-dn],[it.lat+dl,it.lon+dn]],{color:"#e11",weight:2,fill:false}).addTo(m);return m};
    const m1=mk(d.querySelector(".m1")), m2=mk(d.querySelector(".m2"));
    L.tileLayer(NL_TILE("PHOTO_MIX"),{maxZoom:19,attribution:NL_ATTR}).addTo(m2);
    L.tileLayer(NL_TILE("PHOTO_MIX"),{maxZoom:19,opacity:.55}).addTo(m1);       // 無資料處以現況影像淡淡墊底
    let lay=null,syncing=false;
    const g1=L.layerGroup().addTo(m1), g2=L.layerGroup().addTo(m2), cb1=d.querySelector(".bx1"), cb2=d.querySelector(".bx2");
    const drawBoxes=()=>{g1.clearLayers();g2.clearLayers();
      const q=it.pairs[getPi()]||{boxes:[]};
      (q.boxes||[]).forEach((b,i)=>[g1,g2].forEach(g=>L.rectangle([[b[1],b[0]],[b[3],b[2]]],{color:"#1db954",weight:2,fill:false})
        .bindTooltip(`候選 ${i+1}　變化分數 ${b[4]}`).addTo(g)));
      d.querySelector(".bxn").textContent=q.a?`${q.a}→${q.b}，${(q.boxes||[]).length} 個`:"無變遷配對";
      cb1.checked?m1.addLayer(g1):m1.removeLayer(g1);cb2.checked?m2.addLayer(g2):m2.removeLayer(g2)};
    cb1.onchange=()=>cb1.checked?m1.addLayer(g1):m1.removeLayer(g1);
    cb2.onchange=()=>cb2.checked?m2.addLayer(g2):m2.removeLayer(g2);
    d.refresh=drawBoxes;drawBoxes();
    const link=(a,b)=>a.on("move",()=>{if(syncing)return;syncing=true;b.setView(a.getCenter(),a.getZoom(),{animate:false});syncing=false});
    link(m1,m2);link(m2,m1);
    const sl=d.querySelector(".sl"), lab=d.querySelector(".nlyear"), nly=d.querySelector(".nly");
    const show=()=>{
      if(lay)m1.removeLayer(lay);
      if(!ys.length){lab.textContent="—";nly.textContent="無資料";return}
      const e=ys[+sl.value], f=e.date.slice(0,4)+"-"+e.date.slice(4,6)+"-"+e.date.slice(6);
      if(e.t==="nlsc"){lab.textContent=`${f}｜NLSC 民國${e.y}年`;nly.textContent=`NLSC 1/5000 民國 ${e.y} 年（${f}）`;
        lay=L.tileLayer(NL_TILE("TOPO05KPHOTO_"+String(e.y).padStart(3,"0")),{maxZoom:19,attribution:NL_ATTR}).addTo(m1)}
      else{const k=e.kind==="acquired"?"拍攝日":"發布日（查無拍攝日）";lab.textContent=`${f}｜Wayback ${k}`;nly.textContent=`Esri Wayback ${f} ${k}${e.res?"，"+(+e.res).toFixed(1)+" m":""}`;
        lay=L.tileLayer(WB_TILE(e.rel),{maxZoom:19,maxNativeZoom:18,attribution:WB_ATTR}).addTo(m1)}};
    sl.oninput=show;d.querySelector(".pv").onclick=()=>{sl.value=Math.max(0,+sl.value-1);show()};
    d.querySelector(".nx").onclick=()=>{sl.value=Math.min(+sl.max,+sl.value+1);show()};
    show();setTimeout(()=>{m1.invalidateSize();m2.invalidateSize()},100)});
  return d}
function card(it){
  const s=st[it.rank]=st[it.rank]||{verdict:"",treated:false,improved:false,note:"",pair:null,ts:null};
  const pi=s.pair!=null?s.pair:it.default_pair, p=it.pairs[pi];
  const f=it.factors, al=it.alignment;
  const alTxt=al.applied==null?"—":(al.applied&&!al.uncertain?"自動對位良好":"對位不確定／未套用");
  const el=document.createElement("div");el.className="card";el.dataset.rank=it.rank;
  el.innerHTML=`<div><b>#${it.rank}</b> ${it.place} <span class="tag">自動分級 ${it.tier}</span>
  <span class="meta">${it.lat.toFixed(5)}, ${it.lon.toFixed(5)}　<a target="_blank" href="https://www.google.com/maps/@${it.lat},${it.lon},17z/data=!3m1!1e3">Google 衛星</a>　<a target="_blank" href="https://livingatlas.arcgis.com/wayback/#mapCenter=${it.lon.toFixed(5)}%2C${it.lat.toFixed(5)}%2C18&mode=explore">Esri Wayback</a></span></div>
  <div class="meta">獨立災害事件 ${f.recurrence}（${f.recurrence_band}）｜變遷分數 ${f.change??"—"}（${f.change_band}）｜地形起伏 ${it.relief_m??"—"} m｜信心 ${f.confidence}｜${alTxt}${al.shift_px?` 位移 ${al.shift_px.join(", ")} px`:""}</div>
  <div class="meta">事件：${it.events.map(e=>`${e.y} ${e.n}×${e.r}`).join("；")}</div>
  ${it.uav.length?`<div class="meta">附近 UAV 空拍：${it.uav.map(u=>`${u.title}（${u.dist_m} m）`).join("；")}</div>`:""}
  ${it.pairs.length?`<div class="ctl"><label>比對期別 <select class="pair">${it.pairs.map((q,i)=>`<option value="${i}" ${i==pi?"selected":""}>${fmt(q.a)} → ${fmt(q.b)}　${q.frac!=null?(q.frac*100).toFixed(1)+"%":""}</option>`).join("")}</select></label></div>
  <div class="imgs">${["before","after","overlay","heat"].map(k=>`<figure><img loading="lazy" data-k="${k}" src="${p.img[k]}"><figcaption>${{before:"前期",after:"後期",overlay:"變遷候選",heat:"SSIM 熱區"}[k]}</figcaption></figure>`).join("")}</div>`:`<div class="warnbox">沒有比對面板（缺資料）</div>`}
  <div class="ctl">${["ok","warn","bad"].map(v=>`<label><input type="radio" name="v${it.rank}" value="${v}" ${s.verdict==v?"checked":""}> ${LABEL[v]}</label>`).join("")}
  <label><input type="checkbox" class="tr" ${s.treated?"checked":""}> 已完成治理工程</label>
  <label><input type="checkbox" class="im" ${s.improved?"checked":""}> 現況已改善</label></div>
  <textarea placeholder="判讀依據（看到什麼、為何如此判定；例：2024→2025 左上坡面出現新裸露，與 0403 地震相符）">${s.note||""}</textarea>`;
  el.querySelectorAll(`input[name=v${it.rank}]`).forEach(r=>r.onchange=()=>{s.verdict=r.value;s.ts=new Date().toISOString();save()});
  el.querySelector(".tr").onchange=e=>{s.treated=e.target.checked;save()};
  el.querySelector(".im").onchange=e=>{s.improved=e.target.checked;save()};
  el.querySelector("textarea").oninput=e=>{s.note=e.target.value;save()};
  const sel=el.querySelector(".pair"); if(sel) sel.onchange=()=>{s.pair=+sel.value;const q=it.pairs[s.pair];
    el.querySelectorAll(".imgs img").forEach(im=>im.src=q.img[im.dataset.k]);if(el._nl&&el._nl.refresh)el._nl.refresh();if(el._cl&&el._cl.refresh)el._cl.refresh();if(el._of&&el._of.refresh)el._of.refresh();save()};
  {const ctls=el.querySelectorAll(".ctl");el._nl=nlscPanel(it,()=>s.pair!=null?s.pair:it.default_pair);const last=ctls[ctls.length-1];el.insertBefore(el._nl,last);el._cl=climatePanel(it,()=>s.pair!=null?s.pair:it.default_pair);el.insertBefore(el._cl,last);el._of=officialPanel(it,()=>s.pair!=null?s.pair:it.default_pair);el.insertBefore(el._of,last)}
  el.querySelectorAll(".imgs img").forEach(im=>im.onclick=()=>{const lb=document.getElementById("lightbox");lb.querySelector("img").src=im.src;lb.style.display="flex"});
  return el}
const root=document.getElementById("root");
[["A 級候選（主要覆核對象）",A_ITEMS],["對照組：只依複發性排序的前 N 名（不在 A 級者）——供量化驗證比較排序方法",BASE_ITEMS]].forEach(([t,arr])=>{
  const h=document.createElement("h2");h.textContent=`${t}（${arr.length}）`;root.appendChild(h);arr.forEach(it=>root.appendChild(card(it)))});
prog();
document.getElementById("exp").onclick=()=>{
  const all=[...A_ITEMS,...BASE_ITEMS], out=[];
  all.forEach(it=>{const s=st[it.rank]; if(!s||!s.verdict) return;
    const q=it.pairs[s.pair!=null?s.pair:it.default_pair]||{}, al=it.alignment;
    out.push({rank:it.rank,site:it.site,name:it.place,date_a:q.a||null,date_b:q.b||null,score:q.frac??null,
      alignment_applied:!!al.applied,alignment_uncertain:!!al.uncertain,
      verdict:s.verdict,verdict_label:LABEL[s.verdict],treated:!!s.treated,improved:!!s.improved,note:s.note||"",reviewed_at:s.ts})});
  out.sort((a,b)=>a.rank-b.rank);
  const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:"application/json"}));
  a.download="ledger.json";a.click()};
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description="產生人工覆核工作單 HTML")
    ap.add_argument("--out", default=str(REPO / "review" / "review_queue.html"))
    ap.add_argument("--no-wayback", action="store_true", help="時間軸不含 Esri Wayback")
    ap.add_argument("--no-probe", action="store_true", help="不探測 1/5000 相片基本圖各年涵蓋，滑桿列出全部年份（有空白年）")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    a_items, base_items = build_items(out.parent)
    by_rank = {h["rank"]: h for h in _load(DATA / "top100_consolidated.json", [])}
    yrs = _nlsc_years([it["rank"] for it in a_items + base_items], by_rank, probe=not args.no_probe)
    wbf = _wayback_frames(a_items + base_items, by_rank) if not args.no_wayback else {}
    for it in a_items + base_items:
        it["nlsc_years"] = yrs[it["rank"]]
        it["timeline"] = _timeline(yrs[it["rank"]], wbf.get(it["rank"]))
    init = _load(DATA / "ledger.json", [])
    clim = _load(REPO / "data" / "climate_series.json", None)
    if clim:                                   # 只嵌入覆核清單內的熱點
        keep = {str(it["rank"]) for it in a_items + base_items}
        clim["ranks"] = {r: v for r, v in clim["ranks"].items() if r in keep}
    off = None
    pj = REPO / "data" / "biggis_interp" / "polygons.json"
    if pj.exists():                            # 官方衛星判釋崩塌：每個覆核熱點 1.5 km 內的多邊形（以熱點為原點的公尺座標，整數）
        import math
        J = json.loads(pj.read_text(encoding="utf-8"))
        ranks = {}
        for it in a_items + base_items:
            lon0, lat0 = it["lon"], it["lat"]; k = math.cos(math.radians(lat0)); lst = []
            for ei, ar, flat in J["polys"]:
                xy = [((flat[i] - lon0) * 111320 * k, (flat[i + 1] - lat0) * 110570) for i in range(0, len(flat), 2)]
                if min(p[0] for p in xy) > 1600 or max(p[0] for p in xy) < -1600 or min(p[1] for p in xy) > 1600 or max(p[1] for p in xy) < -1600:
                    continue
                lst.append([ei, [int(round(v)) for p in xy for v in p]])
            ranks[str(it["rank"])] = lst
        off = {"fetched": J["meta"]["fetched"], "events": [[e["date"], e["event"]] for e in J["events"]], "ranks": ranks}
    page = (HTML.replace("__A__", json.dumps(a_items, ensure_ascii=False))
                .replace("__BASE__", json.dumps(base_items, ensure_ascii=False))
                .replace("__INIT__", json.dumps(init, ensure_ascii=False))
                .replace("__CLIM__", json.dumps(clim, ensure_ascii=False, separators=(",", ":")))
                .replace("__OFF__", json.dumps(off, ensure_ascii=False, separators=(",", ":"))))
    static = REPO / "webapp" / "change_detect_viewer" / "static"      # 徽章用：內嵌本地 qrcode.js（MIT）與徽章腳本，工作單可離線開啟
    page = page.replace("__QRJS__", (static / "qrcode.js").read_text(encoding="utf-8").replace("</script>", "<\\/script>")).replace("__BADGEJS__", (static / "site_badge.js").read_text(encoding="utf-8"))
    out.write_text(page, encoding="utf-8")
    print(f"A 級 {len(a_items)} 個、對照組 {len(base_items)} 個 → {out}")


if __name__ == "__main__":
    main()
