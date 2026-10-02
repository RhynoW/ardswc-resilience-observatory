/* 各頁右上角徽章：二維條碼（按下回到首頁）＋「TASTI_SatDashboard」字樣（按下另開新視窗到隊伍的 Hugging Face Space）。
 * 用法：頁面放 <div id="site-badge"></div>，並先載入 /static/qrcode.js（本地 MIT 檔，離線可產生）再載入本檔。
 * 二維條碼編碼的是首頁網址：http(s) 開啟時用目前網站的根網址；以 file:// 離線開啟（例如人工覆核工作單）時用公開網站網址。 */
(function () {
  var TEAM = "TASTI_SatDashboard";
  var TEAM_URL = "https://huggingface.co/spaces/RhynoWu/ATRDC-SatDashboard-i18n";
  var PUBLIC_HOME = "https://rhynowu-ardswc-resilience-observatory.hf.space/";
  var HOME = /^https?:$/.test(location.protocol) ? location.origin + "/" : PUBLIC_HOME;
  var CSS =
    ".site-badge{display:flex;flex-direction:column;align-items:center;gap:3px;line-height:1.2;flex:none}" +
    ".site-badge .sb-qr{display:block;border-radius:3px}" +
    ".site-badge .sb-qr svg{width:72px;height:72px;background:#fff;padding:3px;border:1px solid #bbb;border-radius:3px;display:block}" +
    ".site-badge .sb-team{font:700 11px/1.2 ui-monospace,Menlo,Consolas,monospace;color:inherit;text-decoration:none;letter-spacing:.02em}" +
    ".site-badge .sb-team:hover{text-decoration:underline}" +
    ".site-badge a:focus-visible{outline:3px solid #1a73e8;outline-offset:2px}" +
    ".site-badge .sb-vh{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}" +
    "@media print{.site-badge{display:none}}";

  function mount() {
    var host = document.getElementById("site-badge");
    if (!host) return;
    var st = document.createElement("style");
    st.textContent = CSS;
    document.head.appendChild(st);
    host.className = (host.className ? host.className + " " : "") + "site-badge";

    var qa = document.createElement("a");
    qa.className = "sb-qr";
    qa.href = HOME;
    qa.title = "回到首頁";
    qa.setAttribute("aria-label", "回到首頁（二維條碼，掃描可開啟首頁）");
    try {
      var q = qrcode(0, "M");
      q.addData(HOME);
      q.make();
      qa.innerHTML = q.createSvgTag({ cellSize: 3, margin: 0, scalable: true });
    } catch (e) {
      qa.textContent = "首頁";
    }

    var t = document.createElement("a");
    t.className = "sb-team";
    t.href = TEAM_URL;
    t.target = "_blank";
    t.rel = "noopener";
    t.textContent = TEAM;
    var v = document.createElement("span");
    v.className = "sb-vh";
    v.textContent = "（在新分頁開啟）";
    t.appendChild(v);

    host.appendChild(qa);
    host.appendChild(t);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount);
  else mount();
})();
