"""實務人員操作測試組（3–5 人）：比較「原始資料頁」與「證據鏈頁（/su）」對「是否建議現勘」判斷的幫助（依陳委員 2026-10-04 建議）。
設計（交叉）：20 個坡面分成 A、B 兩組各 10 個（10 對配對：同類型、面積、坡度相近）；
  G1：先用「原始資料」判斷 A 組，再用「證據鏈頁」判斷 B 組；G2：先用「原始資料」判斷 B 組，再用「證據鏈頁」判斷 A 組。參與者交替分到 G1／G2。
  原始資料條件：四欄影像圖（Wayback 較早／最新、Sentinel-2 兩期，白線＝有效坡面）＋純資料表（四年圖層裸露 ha、事件目錄列表、照片清單與原始描述；不含分類、層級、提醒）。
  證據鏈條件：同一張影像圖＋嵌入的 /su 證據鏈頁（四類證據、候選層級與原因、偏移提醒）。
  題目：是否建議安排現勘（建議現勘／可暫緩／資料不足無法決定）、信心、最有幫助的資訊（複選）、備註；自動記錄每題在視窗內可見的秒數。
樣本太小（3–5 人）：只做描述統計與回饋，不宣稱效率提升百分比。
輸出（poc/su_usability/，不公開、不推 HF）：kit_G1.html、kit_G2.html（自足，影像內嵌）、expert_review_20.html（同批 20 個坡面的專家盲判表，作為參考答案）、
  key_usability.json（題號→SU、組別、條件；參與者看不到）、pairs.json（配對與配對品質）、README_測試說明.md。
用法：python scripts/su_usability_kit.py
"""
import base64
import json
import random
import sys
import warnings
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from pyproj import Transformer

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "webapp" / "change_detect_viewer"))
import su_strat_panels as P  # noqa: E402
import su_panel_v2 as V2  # noqa: E402
from landslide_incremental import grid  # noqa: E402

POC = REPO / "data" / "biggis_interp" / "poc"
OUT = POC / "su_usability"
SEED = 20261007
LIVE = "https://rhynowu-ardswc-resilience-observatory.hf.space/su#"
TYPES = [("P1 近期照片描述崩塌／落石", 2), ("P2 事件目錄新增或擴大", 2), ("P3 Sentinel-2 近期新增", 2), ("一般判讀（有證據）", 2), ("資料不足（有證據）", 1), ("一般判讀（無證據，陡坡）", 1)]
DEC = [("go", "建議安排現勘"), ("wait", "可暫緩（先不排）"), ("na", "資料不足，無法決定")]
HELP = ["影像圖", "事件目錄", "現地照片", "近期衛星變化", "影像品質／偏移提醒", "候選層級與原因", "其他"]


def jl(p):
    return json.load(open(p, encoding="utf-8"))


def uri(img, w=1800, q=76):
    img = cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])), interpolation=cv2.INTER_AREA)
    return "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()).decode()


def select():
    recs = jl(REPO / "webapp/change_detect_viewer/static/poc/su/su_records.json")
    T = {t["su_id"]: t for t in jl(POC / "su_table_eff.json")}
    mu = {u["su_id"]: u for u in jl(POC / "su_meta_eff.json")["units"]}
    idx = {r[0]: r for r in jl(REPO / "webapp/change_detect_viewer/static/poc/su/su_index.json")["rows"]}
    used = set(v["su_id"] for v in jl(POC / "su_strat" / "key.json").values())          # 已做過分層驗證的 24 個不重複使用
    rng = random.Random(SEED)

    def feat(sid):
        t, u = T[sid], mu[sid]
        return {"su_id": sid, "i": t["i"], "cx": t["cx"], "cy": t["cy"], "eff_ha": u["eff_area_ha"], "slope": u["mean_slope_deg"], "cat": t["category"], "tier": idx[sid][1]}
    pools = {}
    sid_ok = lambda sid: sid in T and sid not in used and mu[sid]["eff_area_ha"] >= 1.5
    pri_ok = lambda sid: ok(sid) and recs[sid]["s2_invalid"] < 0.15
    ok = lambda sid: sid in T and sid not in used and mu[sid]["eff_area_ha"] >= 3.0 and mu[sid]["eff_frac"] >= 0.4 and "c" not in idx[sid][3]      # 排除 Sentinel-2 視窗被浮水印遮蔽 ≥30% 者（資料不足類型另計）
    pools["P1 近期照片描述崩塌／落石"] = [s for s, r in recs.items() if pri_ok(s) and r["tier"] == "優先人工判讀" and any(w.startswith("P1") for w in r["why"])]
    pools["P2 事件目錄新增或擴大"] = [s for s, r in recs.items() if pri_ok(s) and r["tier"] == "優先人工判讀" and any(w.startswith("P2") for w in r["why"]) and not any(w.startswith("P1") for w in r["why"])]
    pools["P3 Sentinel-2 近期新增"] = [s for s, r in recs.items() if pri_ok(s) and r["tier"] == "優先人工判讀" and any(w.startswith("P3") for w in r["why"]) and len(r["why"]) == 1]
    pools["一般判讀（有證據）"] = [s for s, r in recs.items() if pri_ok(s) and r["tier"] == "一般判讀"]
    pools["資料不足（有證據）"] = [s for s, r in recs.items() if sid_ok(s) and r["tier"] == "資料不足" and mu[s]["eff_area_ha"] >= 1.5]
    pools["一般判讀（無證據，陡坡）"] = [s for s, i_ in idx.items() if i_[1] == "一般判讀" and i_[2] == 0 and ok(s) and (mu[s]["mean_slope_deg"] or 0) >= 25 and mu[s]["eff_area_ha"] >= 5.0]
    chosen, pairs = [], []
    far = lambda f: all(((f["cx"] - c["cx"]) ** 2 + (f["cy"] - c["cy"]) ** 2) ** 0.5 >= 1000 for c in chosen)
    for name, npair in TYPES:
        pool = [feat(s) for s in pools[name]]
        rng.shuffle(pool)
        for _ in range(npair):
            anchor = next((f for f in pool if far(f)), None)
            if anchor is None:
                continue
            chosen.append(anchor)
            pool.remove(anchor)
            cand = [f for f in pool if far(f)]
            cand.sort(key=lambda f: (f["cat"] != anchor["cat"], abs(np.log(f["eff_ha"] / anchor["eff_ha"])) + abs((f["slope"] or 0) - (anchor["slope"] or 0)) / 10))
            partner = cand[0]
            chosen.append(partner)
            pool.remove(partner)
            a_set = rng.choice("AB")
            pairs.append({"type": name, "A": (anchor if a_set == "A" else partner), "B": (partner if a_set == "A" else anchor)})
    return pairs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = select()
    print("配對", len(pairs))
    # 配對品質
    q = {"n_pairs": len(pairs), "same_category_share": round(float(np.mean([p["A"]["cat"] == p["B"]["cat"] for p in pairs])), 2),
         "mean_abs_logratio_eff_ha": round(float(np.mean([abs(np.log(p["A"]["eff_ha"] / p["B"]["eff_ha"])) for p in pairs])), 2), "mean_abs_slope_diff": round(float(np.mean([abs((p["A"]["slope"] or 0) - (p["B"]["slope"] or 0)) for p in pairs])), 1)}
    print(q)
    sus = [(p["A"], "A", p["type"]) for p in pairs] + [(p["B"], "B", p["type"]) for p in pairs]
    # 影像圖
    lab10 = np.load(POC / "su_lab10_eff.npy")
    tf, _ = grid()
    to_ll = Transformer.from_crs(3826, 4326, always_xy=True)
    to_m = Transformer.from_crs(3826, 3857, always_xy=True)
    arrs = {}
    for d_ in ("20240404", "20250325"):
        a_ = np.load(POC / "s2" / f"rgb_{d_}.npy")
        a_[~np.load(POC / "s2" / f"valid_{d_}.npy")] = 128
        arrs[d_] = a_
    recs = jl(REPO / "webapp/change_detect_viewer/static/poc/su/su_records.json")
    from concurrent.futures import ThreadPoolExecutor

    official = {k: np.load(POC / "incremental_masks.npz")[k] for k in ("2021", "2024")}

    def one(item):
        f, s, typ = item
        t = dict(f)
        t["lon"], t["lat"] = to_ll.transform(f["cx"], f["cy"])
        cells = V2.build_cells(t)
        return f["su_id"], V2.compose(cells, t, lab10, tf, to_m, official), V2.compose(cells, t, lab10, tf, to_m, None), t
    with ThreadPoolExecutor(3) as ex:
        res = list(ex.map(one, sus))
    img = {sid: im for sid, im, _, _ in res}
    img_e = {sid: ie for sid, _, ie, _ in res}
    meta = {sid: t for sid, _, _, t in res}
    # 題號（兩組共用 T01–T20 的隨機對照；G1／G2 的題目順序各自隨機）
    rng = random.Random(SEED + 1)
    order = list(range(len(sus)))
    rng.shuffle(order)
    code = {sus[j][0]["su_id"]: f"T{n:02d}" for n, j in enumerate(order, 1)}
    setof = {f["su_id"]: s for f, s, _ in sus}
    typeof = {f["su_id"]: t for f, _, t in sus}
    key = {code[sid]: {"su_id": sid, "set": setof[sid], "type": typeof[sid], "tier": recs[sid]["tier"] if sid in recs else "一般判讀（無證據）", "pair_partner": None} for sid in code}
    for p in pairs:
        key[code[p["A"]["su_id"]]]["pair_partner"] = code[p["B"]["su_id"]]
        key[code[p["B"]["su_id"]]]["pair_partner"] = code[p["A"]["su_id"]]
    (OUT / "key_usability.json").write_text(json.dumps(dict(sorted(key.items())), ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "pairs.json").write_text(json.dumps({"quality": q, "pairs": [{"type": p["type"], "A": code[p["A"]["su_id"]], "B": code[p["B"]["su_id"]]} for p in pairs]}, ensure_ascii=False, indent=1), encoding="utf-8")

    def sheet(sid):                                         # 原始資料條件：純資料表（無分類、無層級、無提醒）
        r = recs.get(sid)
        t = None
        if r is None:
            tt = next(x for x in jl(POC / "su_table_eff.json") if x["su_id"] == sid)
            bare = tt["bare_ha"]
            ev, ph = [], []
        else:
            bare, ev, ph = r["bare"], r["events"], r["photo_list"]
        h = "<table><tr><th>年度圖層</th>" + "".join(f"<th>{'2024年' if y == '2024' else y}</th>" for y in ("2021", "2022", "2023", "2024")) + "</tr><tr><td>裸露面積 ha（單元內；2024年＝113年度圖層，影像約2025-03～04）</td>" + "".join(f"<td>{bare.get(y, 0)}</td>" for y in ("2021", "2022", "2023", "2024")) + "</tr></table>"
        h += "<p><b>事件目錄（2021-07～2025-03，多邊形落在此單元者）</b></p>" + ("<ul>" + "".join(f"<li>{e[0]} {e[1]}（{e[2]} ha）</li>" for e in ev) + "</ul>" if ev else "<p class='m'>無</p>")
        h += "<p><b>歷史照片（災害事件／媒體報導，最多 10 筆）</b></p>" + ("<ul>" + "".join(f"<li>{p_[0]}：<a href='https://photo.ardswc.gov.tw/api/Media/{p_[2]}' target='_blank'>{p_[3] or p_[2]}</a></li>" for p_ in ph) + "</ul>" if ph else "<p class='m'>無</p>")
        return h
    items = {}
    for sid, c in code.items():
        s = setof[sid]
        mu_ = next(u for u in jl(POC / "su_meta_eff.json")["units"] if u["su_id"] == sid)
        items[sid] = {"code": c, "img": uri(img[sid], 1520, 70), "img_e": uri(img_e[sid], 1520, 70), "sheet": sheet(sid), "su": sid, "area": f"有效坡面 {mu_['eff_area_ha']} ha"}
    html_kit = lambda grp: kit(grp, items, setof)
    for grp in ("G1", "G2"):
        (OUT / f"kit_{grp}.html").write_text(html_kit(grp), encoding="utf-8")
    (OUT / "expert_review_20.html").write_text(expert(items), encoding="utf-8")
    (OUT / "README_測試說明.md").write_text(readme(q), encoding="utf-8")
    for p in sorted(OUT.iterdir()):
        print(p.name, round(p.stat().st_size / 1e6, 2), "MB")


CSS = """body{font:15px/1.65 system-ui,'Noto Sans TC',sans-serif;margin:0;background:#f6f5f2;color:#222}header{position:sticky;top:0;background:#222;color:#fff;padding:8px 16px;z-index:9}header button{margin-left:12px}main{max-width:1500px;margin:auto;padding:12px 16px}
.it{background:#fff;border:1px solid #ddd;padding:10px;margin:0 0 18px}.it img{width:100%;display:block}.ph{background:#e8f0fc;padding:6px 10px;margin:18px 0 8px;border-left:5px solid #0b5fd4;font-weight:600}table{border-collapse:collapse;font-size:13.5px}th,td{border:1px solid #ddd;padding:3px 8px}.m{color:#777}
label{margin-right:14px;white-space:nowrap}.done{border-left:6px solid #2a7}iframe{width:100%;height:760px;border:1px solid #ccc;margin-top:8px}textarea,input[type=text]{font:inherit;padding:4px 8px;width:60%}.q{margin-top:6px}"""


def kit(grp, items, setof):
    phases = {"G1": [("A", "orig"), ("B", "tool")], "G2": [("B", "orig"), ("A", "tool")]}[grp]
    rng = random.Random(SEED + (7 if grp == "G1" else 8))
    body = []
    for n, (sname, cond) in enumerate(phases, 1):
        sids = [s for s in items if setof[s] == sname]
        rng.shuffle(sids)
        if cond == "orig":
            head = f"第 {n} 階段（共 2 階段）：使用「原始資料」判斷 10 個坡面。每題提供多時點影像圖（見上方圖例）與一份資料表（四年圖層裸露、事件目錄、照片清單）。"
        else:
            head = f"第 {n} 階段（共 2 階段）：使用「證據鏈頁」判斷另外 10 個坡面。每題提供同樣的多時點影像圖，以及一個嵌入的證據鏈頁（需連網；也可點「在新分頁開啟」）。頁面上的候選層級是 POC 規則、尚未驗證，請依你的專業判斷，不必照著選。"
        body.append(f'<div class="ph">{head}</div>')
        for s in sids:
            it = items[s]
            c = it["code"]
            extra = f'<div class="sheet">{it["sheet"]}</div>' if cond == "orig" else f'<p><a href="{LIVE}{it["su"]}" target="_blank" rel="noopener">在新分頁開啟證據鏈頁（{it["su"]}）</a></p><iframe src="{LIVE}{it["su"]}" title="證據鏈頁 {c}" loading="lazy"></iframe>'
            dec = "".join(f'<label><input type="radio" name="d_{c}" value="{k}">{t}</label>' for k, t in DEC)
            conf = "".join(f'<label><input type="radio" name="c_{c}" value="{v}">{v}</label>' for v in ("高", "中", "低"))
            hp = "".join(f'<label><input type="checkbox" name="h_{c}" value="{h}">{h}</label>' for h in HELP if (cond == "tool" or h != "候選層級與原因"))
            body.append(f'<div class="it" id="it_{c}" data-code="{c}" data-cond="{cond}"><b>{c}</b>　<span class="m">{it["area"]}</span><img src="{it["img"]}" alt="{c} 四欄影像：Wayback 較早、最新，Sentinel-2 兩期">{extra}'
                        f'<p class="q"><b>是否建議安排現勘？</b>　{dec}</p><p class="q">信心：{conf}</p><p class="q">最有幫助的資訊（可複選）：{hp}</p><p class="q">備註：<input type="text" id="n_{c}"></p></div>')
    fin = '<div class="ph">最後：整體回饋</div><div class="it"><p>依你的使用經驗（1＝完全沒幫助，5＝很有幫助）：<br>原始資料條件對決定是否現勘的幫助：' + "".join(f'<label><input type="radio" name="u_orig" value="{k}">{k}</label>' for k in range(1, 6)) + '<br>證據鏈頁條件的幫助：' + "".join(f'<label><input type="radio" name="u_tool" value="{k}">{k}</label>' for k in range(1, 6)) + '</p><p>最令你困惑的地方：<br><textarea id="f1" rows="2"></textarea></p><p>還需要哪些資訊或功能：<br><textarea id="f2" rows="2"></textarea></p><p>職務與巡查／判釋年資（選填，不會公開個人身分）：<input type="text" id="f3"></p></div>'
    js = """
const PH=document.querySelectorAll('.it[data-code]');const T={};PH.forEach(e=>T[e.dataset.code]=0);let vis=new Set();
const io=new IntersectionObserver(es=>es.forEach(e=>{const c=e.target.dataset.code;if(e.intersectionRatio>=0.4)vis.add(c);else vis.delete(c)}),{threshold:[0,0.4,0.8]});PH.forEach(e=>io.observe(e));
setInterval(()=>{if(document.hidden)return;vis.forEach(c=>T[c]+=1);prog()},1000);
function prog(){let n=0;PH.forEach(e=>{const c=e.dataset.code;const d=document.querySelector('input[name=d_'+c+']:checked'),f=document.querySelector('input[name=c_'+c+']:checked');const ok=d&&f;e.classList.toggle('done',!!ok);if(ok)n++});document.getElementById('p').textContent='　'+n+' / '+PH.length+' 已完成'}
document.addEventListener('change',prog);
function exp(){const pid=document.getElementById('pid').value.trim();if(!pid){alert('請先填參與者代號');return}
const items={};PH.forEach(e=>{const c=e.dataset.code;const d=document.querySelector('input[name=d_'+c+']:checked'),f=document.querySelector('input[name=c_'+c+']:checked');
items[c]={condition:e.dataset.cond,decision:d?d.value:null,confidence:f?f.value:null,helpful:[...document.querySelectorAll('input[name=h_'+c+']:checked')].map(x=>x.value),note:document.getElementById('n_'+c).value,seconds_visible:T[c]}});
const u=n=>{const x=document.querySelector('input[name='+n+']:checked');return x?+x.value:null};
const o={participant:pid,group:GROUP,exported:new Date().toISOString(),items,feedback:{useful_orig:u('u_orig'),useful_tool:u('u_tool'),confusing:document.getElementById('f1').value,wanted:document.getElementById('f2').value,role:document.getElementById('f3').value}};
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(o,null,1)],{type:'application/json'}));a.download='usability_'+pid+'_'+GROUP+'.json';a.click()}
"""
    return f"""<!doctype html><meta charset="utf-8"><title>坡面判讀操作測試（{grp}）</title><style>{CSS}</style>
<header><b>坡面判讀操作測試</b><span id="p"></span>　參與者代號：<input type="text" id="pid" size="8" placeholder="如 P01"><button onclick="exp()">匯出結果 JSON</button></header><main>
<p>這是一個測試，<b>不是考試，也沒有標準答案</b>。情境：您是巡查規劃人員，要決定下面每個坡面是否建議安排現勘。請依您的專業判斷作答，沒把握就選「資料不足，無法決定」並標示信心。頁面會自動記錄每題在視窗內的停留秒數（只用於比較兩種資料呈現方式），不記錄其他個資。官方判釋是衛星判釋、不是現地確認。完成後按右上角「匯出結果 JSON」，把檔案寄回。</p><p><b>影像圖怎麼看：</b>每個坡面一張圖，由舊到新排成兩列、每列四格（Wayback 0.3–0.5 m 與 Sentinel-2 10 m，標題有日期與來源；Sentinel-2 已做對比增強，僅供目視，原本被浮水印遮住的區域改用較粗的圖磚補回，會比較模糊）。<b>白線</b>＝我們切出的坡面單元中坡度 ≥15° 的「有效坡面」範圍，是統計單元的外框，<b>不是裸露地的邊界</b>；<b>紅線</b>＝官方 2024 年度崩塌地圖層的裸露範圍（即 113 年度圖層，影像約 2025-03～04）；<b>黃線</b>＝2021 年度圖層的裸露範圍。Wayback 影像對地形資料常有 10–25 m 偏移，線與影像可能錯位，請以大致位置判斷。</p>
{"".join(body)}{fin}</main><script>const GROUP="{grp}";{js}</script>"""


def expert(items):
    OPT = P.OPT
    rows = "".join(f'<div class="it" id="it_{it["code"]}" data-code="{it["code"]}"><b>{it["code"]}</b><img src="{it["img_e"]}" alt="{it["code"]} 多時點影像"><p class="q">' + "".join(f'<label><input type="radio" name="v_{it["code"]}" value="{k}"><b>{k}</b> {t}</label>' for k, t in OPT) + '</p><p class="q">信心：' + "".join(f'<label><input type="radio" name="c_{it["code"]}" value="{v}">{v}</label>' for v in ("高", "中", "低")) + f'　備註：<input type="text" id="n_{it["code"]}"></p></div>' for it in sorted(items.values(), key=lambda x: x["code"]))
    js = """const IDS=[...document.querySelectorAll('.it')].map(e=>e.dataset.code);
function exp(){const pid=document.getElementById('pid').value.trim();if(!pid){alert('請先填判讀者代號');return}const o={rater:pid,exported:new Date().toISOString(),items:{}};IDS.forEach(i=>{const v=document.querySelector('input[name=v_'+i+']:checked'),c=document.querySelector('input[name=c_'+i+']:checked');o.items[i]={verdict:v?v.value:null,conf:c?c.value:null,note:document.getElementById('n_'+i).value}});const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(o,null,1)],{type:'application/json'}));a.download='expert_'+pid+'.json';a.click()}
document.addEventListener('change',()=>{let n=0;IDS.forEach(i=>{if(document.querySelector('input[name=v_'+i+']:checked')&&document.querySelector('input[name=c_'+i+']:checked'))n++;document.getElementById('it_'+i).classList.toggle('done',!!(document.querySelector('input[name=v_'+i+']:checked')&&document.querySelector('input[name=c_'+i+']:checked')))});document.getElementById('p').textContent='　'+n+' / '+IDS.length})"""
    return f"""<!doctype html><meta charset="utf-8"><title>專家盲判（20 個坡面）</title><style>{CSS}</style><header><b>專家盲判（20 個坡面）</b><span id="p"></span>　判讀者代號：<input type="text" id="pid" size="8" placeholder="如 E01"><button onclick="exp()">匯出 JSON</button></header><main>
<p>每張圖是同一個坡面在多個時間點的影像，由舊到新排成兩列、每列四格：Wayback（0.3–0.5 m）與 Sentinel-2（10 m，已做對比增強、僅供目視；原本被浮水印遮住的區域改用較粗圖磚補回），標題有日期與來源。白線＝≥15° 有效坡面的外框（不是裸露地邊界）。請只依影像判斷，不要參考任何系統推薦。Wayback 影像對 DTM／Sentinel-2 常有 10–25 m 的偏移，請以輪廓大致位置判斷。選項：A 近期有新增／擴大崩塌；B 有崩塌但近期無變化；C 無崩塌；D 有裸露但非崩塌（河床、道路、工程、農地）；E 無法判讀。</p>{rows}</main><script>{js}</script>"""


def readme(q):
    return f"""# 實務人員操作測試說明

目的：比較「原始資料」與「證據鏈頁（/su）」對「是否建議安排現勘」判斷的幫助。**只做描述統計與回饋；樣本 3–5 人，不宣稱效率提升百分比。**

## 設計（交叉）
- 20 個坡面分成 A、B 兩組各 10 個（10 對配對：同類型、面積與坡度相近；配對品質：{q}）。
- 參與者交替分到 G1／G2：G1 先用「原始資料」做 A 組、再用「證據鏈頁」做 B 組；G2 先用「原始資料」做 B 組、再用「證據鏈頁」做 A 組。這樣每人兩種條件都做，且兩組題目輪流用兩種條件，抵銷題組難度差異。
- 原始資料條件：四欄影像圖＋純資料表（四年圖層裸露、事件目錄、照片清單與原始描述）；證據鏈條件：同一張影像圖＋嵌入的 /su 頁（四類證據、候選層級與原因、偏移提醒）。

## 執行
1. 把 `kit_G1.html` 或 `kit_G2.html` 寄給參與者（自足檔案，影像內嵌；證據鏈階段需連網）。每位參與者只用其中一份。
2. 請參與者填「參與者代號」（如 P01），依序完成兩個階段，按「匯出結果 JSON」寄回。
3. 同一批 20 個坡面另請 2–3 位專家用 `expert_review_20.html` 盲判（A–E），作為參考答案。
4. 回收後：`python scripts/su_usability_analyze.py <結果資料夾>`（描述統計：每題秒數中位數、決定分布、信心、最有幫助的資訊、整體回饋；對照專家共識）。

## 注意
- 告知參與者：這不是考試，沒有標準答案；候選層級是 POC 規則、尚未驗證，不必照著選。
- 不收集個資；「職務與年資」選填。
- `key_usability.json`（題號→坡面、組別、層級）與 `pairs.json` 不要給參與者。
- 證據鏈頁的網址為 Hugging Face Space；若 Space 離線，該階段無法作答。
- 限制：每人 20 題、停留秒數是「題目在視窗內可見」的時間，不等於實際思考時間；配對雖相近但仍有差異；3–5 人無法做統計檢定。
產製日期：{date.today().isoformat()}
"""


if __name__ == "__main__":
    main()
