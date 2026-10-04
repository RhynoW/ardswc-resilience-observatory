"""產生人工抽查表（盲測）：把 su_validation 的 V01–V30 與 uav_validation 的 U01–U16 盲測圖放進一頁，
讓人工獨立判讀；不顯示 AI 判讀，也不顯示官方圖層類別。輸出 poc/spotcheck/index.html（用瀏覽器開）。
判讀存在瀏覽器 localStorage，按「匯出」下載 verdicts_human.json，再放回對應資料夾後用
  python scripts/su_wayback_score.py verdicts_human.json  /  python scripts/su_uav_score.py verdicts_human.json
重新評分，或 python scripts/spotcheck_compare.py 比較 AI 與人工判讀。
"""
import json
from pathlib import Path

POC = Path(__file__).resolve().parent.parent / "data" / "biggis_interp" / "poc"
VERD = ["持續", "擴大", "縮減", "首次出現", "消失", "無崩塌", "無法判讀"]
HELP = {"持續": "前後期都有裸露，大小差不多", "擴大": "裸露範圍明顯變大", "縮減": "裸露範圍明顯變小（植生恢復）", "首次出現": "前期沒有、後期才出現裸露",
        "消失": "前期有、後期已無裸露", "無崩塌": "各期都沒有崩塌裸露（植生、道路、河床不算）", "無法判讀": "雲、陰影、解析度或日期不足"}


def items():
    out = []
    for pre, d, title in (("V", "su_validation", "Wayback／Sentinel-2 坡面盲測"), ("U", "uav_validation", "UAV 對照盲測")):
        key = json.load(open(POC / d / "key.json", encoding="utf-8"))
        for k in key:
            out.append((k, f"../{d}/{k}.png", title))
    return out


def main():
    its = items()
    parts = ["""<!doctype html><meta charset="utf-8"><title>盲測抽查表</title>
<style>body{font:15px/1.6 system-ui,'Noto Sans TC',sans-serif;margin:0;background:#f6f5f2;color:#222}header{position:sticky;top:0;background:#222;color:#fff;padding:8px 16px;z-index:9;display:flex;gap:16px;align-items:center}
header button{padding:4px 12px}main{max-width:1500px;margin:auto;padding:12px 16px}.it{background:#fff;border:1px solid #ddd;margin:0 0 18px;padding:10px 12px}.it img{width:100%;display:block;cursor:zoom-in}
.row{display:flex;flex-wrap:wrap;gap:6px 14px;margin-top:8px;align-items:center}label{white-space:nowrap}h2{margin:24px 0 6px}.done{border-left:6px solid #2a7}.help{color:#555;font-size:13px}input.note{flex:1;min-width:240px}</style>
<header><b>盲測抽查表</b><span id="prog"></span><button onclick="exp()">匯出 verdicts_human.json</button><button onclick="if(confirm('清除全部判讀？')){localStorage.removeItem('spot');location.reload()}">清除</button></header>
<main><p class="help">每張圖是同一個坡面（白線＝坡面外框）在不同日期的影像（左舊右新，標題有日期）。請只依影像判斷，不要參考任何其他資訊。<br>"""]
    parts.append("；".join(f"<b>{k}</b>＝{HELP[k]}" for k in VERD) + "。信心：高＝很確定、中、低＝不太確定。UAV 圖為 0.6 m 航拍。</p>")
    last = None
    for k, src, title in its:
        if title != last:
            parts.append(f"<h2>{title}</h2>")
            last = title
        radios = "".join(f'<label><input type="radio" name="v_{k}" value="{v}" onchange="sv(\'{k}\')">{v}</label>' for v in VERD)
        conf = "".join(f'<label><input type="radio" name="c_{k}" value="{c}" onchange="sv(\'{k}\')">{c}</label>' for c in ("高", "中", "低"))
        parts.append(f'<div class="it" id="it_{k}"><b>{k}</b><a href="{src}" target="_blank"><img src="{src}" loading="lazy"></a><div class="row">{radios}<span>｜信心</span>{conf}<input class="note" id="n_{k}" placeholder="備註（選填）" oninput="sv(\'{k}\')"></div></div>')
    ids = json.dumps([k for k, _, _ in its])
    parts.append("""</main><script>
const IDS=%s;let S={};try{S=JSON.parse(localStorage.getItem('spot')||'{}')}catch(e){}
function sv(k){const v=document.querySelector('input[name=v_'+k+']:checked'),c=document.querySelector('input[name=c_'+k+']:checked');S[k]={verdict:v?v.value:null,conf:c?c.value:null,note:document.getElementById('n_'+k).value};try{localStorage.setItem('spot',JSON.stringify(S))}catch(e){}prog()}
function prog(){let n=0;IDS.forEach(k=>{const s=S[k];const d=s&&s.verdict&&s.conf;document.getElementById('it_'+k).classList.toggle('done',!!d);if(d)n++});document.getElementById('prog').textContent=n+' / '+IDS.length+' 已完成'}
IDS.forEach(k=>{const s=S[k];if(!s)return;if(s.verdict){const e=document.querySelector('input[name=v_'+k+'][value='+s.verdict+']');if(e)e.checked=true}if(s.conf){const e=document.querySelector('input[name=c_'+k+'][value='+s.conf+']');if(e)e.checked=true}document.getElementById('n_'+k).value=s.note||''});prog();
function exp(){const o={_note:'人工判讀（盲測抽查，'+new Date().toISOString().slice(0,10)+'）'};IDS.forEach(k=>{const s=S[k];if(s&&s.verdict&&s.conf)o[k]={verdict:s.verdict,conf:s.conf,note:s.note||''}});
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(o,null,1)],{type:'application/json'}));a.download='verdicts_human.json';a.click()}
</script>""" % ids)
    out = POC / "spotcheck" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text("".join(parts), encoding="utf-8")
    print(out, len(its), "張")


if __name__ == "__main__":
    main()
