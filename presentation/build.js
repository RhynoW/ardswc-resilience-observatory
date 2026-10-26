const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { applyTheme } = require("C:/Users/user/.claude/skills/synced/c8fdb94f-bdfa-4737-b788-9b2d7d63db72_abd4613a-d1c0-4c62-a729-52c56ffb4d41/pptx/scripts/apply_theme.js");

const FONT = "Microsoft JhengHei";
const THEME = {
  name: "SlopeSentinel",
  headFontFace: FONT,
  bodyFontFace: FONT,
  colors: {
    dk1: "1E2B22", lt1: "FFFFFF", dk2: "2C5F2D", lt2: "EEF3EC",
    accent1: "2C5F2D", accent2: "B85042", accent3: "97BC62", accent4: "5E6B63", accent5: "D9A441", accent6: "4A6FA5",
    hlink: "2C5F2D", folHlink: "5E6B63",
  },
};
const HEX = THEME.colors;

async function icon(Comp, hex) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + hex, size: 256 }));
  const buf = await sharp(Buffer.from(svg)).resize(256, 256).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
  pres.title = "地景複發熱點觀測站";
  pres.author = "TASTI_SatDashboard";
  pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
  const C = pres.SchemeColor;

  pres.defineSlideMaster({
    title: "TITLE_DARK",
    background: { color: C.text2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.3, w: 11.7, h: 1.3, fontSize: 54, bold: true, color: C.background1, valign: "middle", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 2.75, w: 11.7, h: 1.3, fontSize: 26, color: C.background2, valign: "top", margin: 0 }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENT",
    background: { color: C.background1 },
    margin: [0.5, 0.6, 0.5, 0.6],
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 32, bold: true, color: C.text2, valign: "middle", margin: 0 }, text: "" } },
      { text: { text: "TASTI_SatDashboard・OSSInt 2026 決賽", options: { x: 0.6, y: 7.0, w: 6, h: 0.3, fontSize: 10, color: C.accent4, margin: 0 } } },
    ],
    slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent4, align: "right" },
  });

  const ic = {
    cat: await icon(fa.FaSatellite, HEX.accent1), cam: await icon(fa.FaCamera, HEX.accent1), img: await icon(fa.FaLayerGroup, HEX.accent1),
    robot: await icon(fa.FaSearchLocation, HEX.lt1), link: await icon(fa.FaLink, HEX.lt1), user: await icon(fa.FaUserCheck, HEX.lt1),
    open: await icon(fa.FaCodeBranch, HEX.accent1), people: await icon(fa.FaUsers, HEX.accent1), map: await icon(fa.FaMapMarkedAlt, HEX.accent1),
    ruler: await icon(fa.FaRulerCombined, HEX.accent1), copy: await icon(fa.FaClone, HEX.accent1), db: await icon(fa.FaDatabase, HEX.accent1),
    eye: await icon(fa.FaEye, HEX.accent1), home: await icon(fa.FaHome, HEX.accent1), hands: await icon(fa.FaHandsHelping, HEX.accent1),
  };

  // 1 封面
  pres.addSection({ title: "開場" });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "開場" });
  s.addText("地景複發熱點觀測站", { placeholder: "title" });
  s.addText([{ text: "坡地韌性哨兵", options: { bold: true, color: C.background1, breakLine: true } }, { text: "把事件目錄、現地照片與衛星影像，整理成每個坡面的證據鏈", options: { fontSize: 18 } }], { placeholder: "body" });
  const flow = [["機器幫忙決定", "先看誰", ic.robot], ["證據鏈告訴你", "為什麼", ic.link], ["最後由人決定", "要不要去", ic.user]];
  flow.forEach(([a, b, im], k) => {
    const x = 0.8 + k * 3.95;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 4.45, w: 3.6, h: 1.25, fill: { color: C.accent1 }, line: { color: C.accent3, width: 1 }, rectRadius: 0.12, objectName: `flow${k}` });
    s.addImage({ data: im, x: x + 0.25, y: 4.75, w: 0.6, h: 0.6, objectName: `flowIcon${k}` });
    s.addText([{ text: a, options: { fontSize: 16, color: C.background2, breakLine: true } }, { text: b, options: { fontSize: 24, bold: true, color: C.background1 } }],
      { x: x + 1.05, y: 4.55, w: 2.45, h: 1.05, valign: "middle", margin: 0, isTextBox: true, objectName: `flowText${k}` });
  });
  s.addText("TASTI_SatDashboard｜開源空間資訊於國家韌性應用黑客松 OSSInt 2026 決賽　　rhynowu-ardswc-resilience-observatory.hf.space",
    { x: 0.8, y: 6.55, w: 11.7, h: 0.4, fontSize: 12, color: C.background2, margin: 0, isTextBox: true });
  s.addNotes("我們是 TASTI_SatDashboard。這個系統不是要取代崩塌判釋，而是把水保署已經有的事件型崩塌目錄、歷史現地照片和近期衛星影像，整理成每個坡面的證據鏈。一句話：機器幫忙決定先看誰，證據鏈告訴你為什麼，最後由人決定要不要去。（約 20 秒）");

  // 2 問題
  pres.addSection({ title: "問題" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "問題" });
  s.addText("巡查先看哪裡？證據散在三個地方", { placeholder: "title" });
  const src = [
    [ic.cat, "事件型崩塌目錄", "衛星判釋每次事件「新生或擴大」的部分；只涵蓋看得到的大事件"],
    [ic.cam, "現地歷史照片", "補小型崩塌、落石、道路邊坡；多拍在道路可達處"],
    [ic.img, "近期衛星影像", "Sentinel-2 約每 5 天一次；Wayback 解析度高、拍攝不定期"],
  ];
  src.forEach(([im, h, d], k) => {
    const x = 0.6 + k * 4.1;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.55, w: 3.8, h: 2.6, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.12, objectName: `src${k}` });
    s.addImage({ data: im, x: x + 0.3, y: 1.85, w: 0.65, h: 0.65, objectName: `srcIcon${k}` });
    s.addText(h, { x: x + 0.3, y: 2.7, w: 3.2, h: 0.5, fontSize: 20, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: `srcH${k}` });
    s.addText(d, { x: x + 0.3, y: 3.25, w: 3.2, h: 1.15, fontSize: 15, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: `srcD${k}` });
  });
  const stats = [["約 1,800 km²", "高雄 POC（桃源・那瑪夏・六龜・甲仙）"], ["13,175", "個坡面單元"], ["8,856", "張現地照片落在 POC 內"]];
  stats.forEach(([n, l], k) => {
    const x = 0.6 + k * 4.1;
    s.addText(n, { x, y: 4.6, w: 3.8, h: 0.8, fontSize: 36, bold: true, color: C.accent2, margin: 0, isTextBox: true, objectName: `stat${k}` });
    s.addText(l, { x, y: 5.4, w: 3.8, h: 0.5, fontSize: 14, color: C.accent4, margin: 0, isTextBox: true, objectName: `statL${k}` });
  });
  s.addNotes("巡查人力有限，但山區坡面上萬個。要判斷先去哪裡，證據散在三個地方：事件型崩塌目錄、現地歷史照片、近期衛星影像，各有盲點。以高雄 POC 為例，約 1,800 平方公里、1 萬 3 千個坡面、近 9 千張現地照片。（約 40 秒）");

  // 3 技術亮點
  pres.addSection({ title: "技術亮點" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "技術亮點" });
  s.addText("技術亮點：免費衛星＋開放資料，每個數字都量過", { placeholder: "title" });
  const tech = [
    ["13,175", "個坡面單元，不是網格", "集水區重疊法切分；12,730 個有 ≥15° 有效坡面。舊版 85% 不連續 → 0%"],
    ["75%・43%", "大事件後抓到崩塌坡面", "Sentinel-2 紅邊時間序列（花蓮・太魯閣）；沒事件的坡面誤報約 2%"],
    ["35", "組沒有 GPS 的 UAV 照片自動對位", "LoFTR 特徵匹配＋20 m DTM，貼回地形；含 2026-10-15 六龜寶來"],
  ];
  tech.forEach(([n, h, d], k) => {
    const x = 0.6 + k * 4.1;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.35, w: 3.9, h: 5.45, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.1, objectName: `techBg${k}` });
    if (k === 0) s.addImage({ path: "v2_only.png", x: x + 0.15, y: 1.5, w: 3.6, h: 2.75, sizing: { type: "cover", w: 3.6, h: 2.75 }, objectName: "techImg0" });
    if (k === 2) s.addImage({ path: "uav_baolai3d.png", x: x + 0.15, y: 1.5, w: 3.6, h: 2.75, sizing: { type: "cover", w: 3.6, h: 2.75 }, objectName: "techImg2" });
    s.addText(n, { x: x + 0.25, y: 4.35, w: 3.4, h: 0.7, fontSize: 32, bold: true, color: k == 1 ? C.accent2 : C.text2, margin: 0, isTextBox: true, objectName: `techN${k}` });
    s.addText(h, { x: x + 0.25, y: 5.05, w: 3.4, h: 0.45, fontSize: 16, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: `techH${k}` });
    s.addText(d, { x: x + 0.25, y: 5.55, w: 3.4, h: 1.15, fontSize: 13, color: C.accent4, margin: 0, valign: "top", isTextBox: true, objectName: `techD${k}` });
  });
  s.addChart(pres.charts.BAR, [{ name: "被抓到的比例", labels: ["<5%", "5–10%", "10–20%", "20–40%", "≥40%"], values: [7, 15, 42, 75, 93] }],
    { x: 4.85, y: 1.5, w: 3.6, h: 2.75, barDir: "col", showTitle: true, title: "太魯閣：崩塌占坡面越大越抓得到", titleFontSize: 11, titleColor: HEX.dk1, titleFontFace: "+mn-lt",
      chartColors: [HEX.accent1], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0"%"', dataLabelFontSize: 10, dataLabelColor: HEX.dk1, dataLabelFontFace: "+mn-lt",
      catAxisLabelFontSize: 9, valAxisHidden: true, catAxisLabelColor: HEX.accent4, catAxisLabelFontFace: "+mn-lt", valAxisMinVal: 0, valAxisMaxVal: 110,
      valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false, barGapWidthPct: 50, plotArea: { fill: { color: HEX.lt1 } }, objectName: "techChart" });
  s.addNotes("技術亮點有三個。第一，以坡面為單位：依輔導委員建議用集水區重疊法切出 1 萬 3 千個坡面，每個都是完整區塊。第二，用免費的 Sentinel-2 紅邊時間序列，在花蓮、太魯閣大事件後抓到 75% 和 43% 的崩塌坡面，沒有事件的坡面誤報約 2%；中間這張圖顯示，崩塌占坡面越大越抓得到。第三，水保署歷史影像平台上的 UAV 照片沒有 GPS，我們用特徵匹配和 20 m 地形自動對位，目前 35 組，包括 10 月 15 日剛上傳的六龜寶來。重點是：每個數字我們都量過準確度。（約 50 秒）");

  // 4 架構創新
  pres.addSection({ title: "架構創新" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "架構創新" });
  s.addText("架構創新：證據鏈分開呈現，不給總分，最後由人決定", { placeholder: "title" });
  const flow2 = [["開放資料", "事件目錄・年度圖層・照片・Sentinel-2・DEM・OSM"], ["坡面單元", "SUv2，版本化 ID"], ["證據鏈", "五類證據，各附限制"], ["三種候選層級", "優先・一般・資料不足"], ["人決定", "要不要現勘；結果回饋規則"]];
  flow2.forEach(([h, d], k) => {
    const x = 0.6 + k * 2.47, last = k === 4;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.4, w: 2.27, h: 1.45, fill: { color: last ? C.accent2 : C.accent1 }, line: { color: last ? C.accent2 : C.accent1 }, rectRadius: 0.1, objectName: `flw${k}` });
    s.addText([{ text: h, options: { fontSize: 17, bold: true, color: C.background1, breakLine: true } }, { text: d, options: { fontSize: 12, color: C.background2 } }],
      { x: x + 0.15, y: 1.45, w: 1.97, h: 1.35, valign: "middle", margin: 0, isTextBox: true, objectName: `flwT${k}` });
    if (k < 4) s.addText("▶", { x: x + 2.24, y: 1.95, w: 0.26, h: 0.35, fontSize: 12, color: C.accent4, align: "center", margin: 0, isTextBox: true, objectName: `flwA${k}` });
  });
  const evid = [["①", "事件目錄與年度圖層", "主骨架"], ["②", "現地照片", "補充，與①不相加"], ["③", "近期衛星變化", "Sentinel-2 紅邊時間序列"], ["④", "影像品質", "雲、陰影、觀測筆數、偏移"], ["⑤", "崩了會碰到什麼", "道路、聚落、土石流潛勢溪流"]];
  s.addText("每個坡面的五類證據", { x: 0.6, y: 3.1, w: 5.6, h: 0.4, fontSize: 15, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: "evLead" });
  evid.forEach(([n, h, d], k) => {
    const y = 3.6 + k * 0.62;
    s.addShape(pres.shapes.OVAL, { x: 0.6, y: y + 0.04, w: 0.46, h: 0.46, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: `ev2D${k}` });
    s.addText(n, { x: 0.6, y: y + 0.04, w: 0.46, h: 0.46, fontSize: 13, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `ev2N${k}` });
    s.addText([{ text: h + "　", options: { bold: true, color: C.text1 } }, { text: d, options: { color: C.accent4 } }], { x: 1.2, y, w: 5.0, h: 0.55, fontSize: 14, valign: "middle", margin: 0, isTextBox: true, objectName: `ev2T${k}` });
  });
  const prin = [["不給總分", "分數會掩蓋「為什麼」；各類證據並列，由判讀者權衡。"], ["「資料不足」也是結果", "雲、陰影、面積太小時明說看不到，不把沒資料當沒問題。"], ["可稽核", "盲判題目與結果附 SHA-256 清單；高解析影像只決定「怎麼看」，不改衛星數值。"]];
  prin.forEach(([h, d], k) => {
    const y = 3.1 + k * 1.25;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.6, y, w: 6.1, h: 1.1, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.1, objectName: `prin${k}` });
    s.addText([{ text: h, options: { fontSize: 16, bold: true, color: C.text2, breakLine: true } }, { text: d, options: { fontSize: 13, color: C.text1 } }],
      { x: 6.85, y: y + 0.05, w: 5.7, h: 1.0, valign: "middle", margin: 0, isTextBox: true, objectName: `prinT${k}` });
  });
  s.addNotes("架構上，資料全部是開放資料，切成坡面單元後，每個坡面掛五類證據：事件目錄是主骨架，照片是補充，衛星只回答最近還有沒有變化，再加上影像品質提醒，以及崩了會碰到什麼——道路、聚落和土石流潛勢溪流。我們刻意不給總分，因為分數會掩蓋為什麼；看不到就標資料不足；驗證紀錄可以稽核。最後只給三種候選層級，由人決定要不要去。（約 45 秒）");

  // 5 韌性呈現（太魯閣案例＋坡面履歷）
  pres.addSection({ title: "韌性呈現" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "韌性呈現" });
  s.addText("韌性呈現：每個坡面一份履歷——壞過、在變、會碰到誰", { placeholder: "title" });
  s.addImage({ path: "taroko_s2.gif", x: 0.6, y: 1.4, w: 4.75, h: 4.75, objectName: "tarokoGif" });
  s.addText("太魯閣 0403 地震：白線＝坡面；橘線＝事件目錄的地震崩塌（官方判釋）", { x: 0.6, y: 6.25, w: 4.75, h: 0.5, fontSize: 12, color: C.accent4, margin: 0, valign: "top", isTextBox: true, objectName: "gifCap" });
  const ts = JSON.parse(require("fs").readFileSync("taroko_ndre_series.json", "utf8"));
  const lab4 = ts.months.map(m => m.slice(2).replace("-", "/"));
  s.addChart(pres.charts.LINE, [
    { name: "此坡面", labels: lab4, values: ts.su.map(v => v == null ? null : v) },
    { name: "全區中位數", labels: lab4, values: ts.region },
  ], { x: 5.75, y: 1.35, w: 6.95, h: 2.8, showTitle: true, title: "此坡面 NDRE：地震後 0.53 → 0.20，之後沒有回來", titleFontSize: 13, titleColor: HEX.dk1, titleFontFace: "+mn-lt",
       chartColors: [HEX.accent2, HEX.accent4], lineSize: 2.5, lineDataSymbol: "none", displayBlanksAs: "span",
       showLegend: true, legendPos: "b", legendFontSize: 10, legendFontFace: "+mn-lt", catAxisLabelFontSize: 9, valAxisLabelFontSize: 9, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
       catAxisLabelColor: HEX.accent4, valAxisLabelColor: HEX.accent4, valAxisMinVal: 0, valAxisMaxVal: 0.8, valAxisMajorUnit: 0.2, valGridLine: { color: "DDE5DF", size: 0.5 }, catGridLine: { style: "none" },
       catAxisLabelFrequency: 3, objectName: "ndreChart" });
  const res = [["229／330", "太魯閣紅邊持續下降的坡面出現在地震當月——事件後幾週內列出先看的坡面"], ["200／411", "高雄優先坡面外推 100 m 內有道路；25 個有建物（OpenStreetMap）"], ["67", "個優先坡面位在土石流潛勢溪流集水區內，崩落土砂可能進入溪流（水保署）"]];
  res.forEach(([n, d], k) => {
    const y = 4.35 + k * 0.8;
    s.addText(n, { x: 5.75, y, w: 1.9, h: 0.7, fontSize: 24, bold: true, color: k == 0 ? C.accent2 : C.text2, valign: "middle", margin: 0, isTextBox: true, objectName: `resN${k}` });
    s.addText(d, { x: 7.7, y, w: 5.0, h: 0.7, fontSize: 13, color: C.text1, valign: "middle", margin: 0, isTextBox: true, objectName: `resD${k}` });
  });
  s.addNotes("韌性呈現在於每個坡面都有一份履歷。左邊是太魯閣一個坡面：橘線是官方圈出的地震崩塌，地震後紅邊指標從 0.53 掉到 0.20，之後沒有回來，全區只是正常季節起伏。太魯閣 330 個持續下降的坡面中，229 個出現在地震當月，大事件後幾週內就能列出要先看的坡面。履歷也回答崩了會碰到誰：高雄 411 個優先坡面中，200 個附近有道路，67 個位在土石流潛勢溪流集水區。（約 55 秒）");

  // 6 對公部門的啟示
  pres.addSection({ title: "啟示" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "啟示" });
  s.addText("對公部門的啟示：自動訊號只排「先看誰」，先量再用", { placeholder: "title" });
  const gv = [["75%・43%", "大事件後抓到崩塌坡面（花蓮・太魯閣）", "95% CI 66–83%・38–48%；誤報約 2%"], ["約 25%", "平時零星的紅邊下降經盲判屬實", "分層加權，95% CI 5–45%"], ["69%", "影像目視與官方圖層「有無裸露」一致", "人工盲測 n=29，95% CI 51–83%"]];
  gv.forEach(([n, l, c], k) => {
    const y = 1.4 + k * 1.75;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y, w: 4.9, h: 1.6, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.1, objectName: `gv${k}` });
    s.addText(n, { x: 0.85, y: y + 0.1, w: 4.5, h: 0.6, fontSize: 28, bold: true, color: k == 1 ? C.accent2 : C.text2, margin: 0, valign: "middle", isTextBox: true, objectName: `gvN${k}` });
    s.addText(l, { x: 0.85, y: y + 0.72, w: 4.5, h: 0.4, fontSize: 14, color: C.text1, margin: 0, isTextBox: true, objectName: `gvL${k}` });
    s.addText(c, { x: 0.85, y: y + 1.12, w: 4.5, h: 0.35, fontSize: 11, color: C.accent4, margin: 0, isTextBox: true, objectName: `gvC${k}` });
  });
  const gp = [[ic.ruler, "量過才用", "大事件後可靠、平時零星變化不可靠；系統把這些限制直接寫在每個坡面頁上，只用來排序，不用來判定。"],
              [ic.copy, "可複製、成本低", "全部開放資料與開源程式；換一個流域只要換範圍重跑，Sentinel-2 免費，一個 POC 約 1–2 小時計算。"],
              [ic.db, "資料治理的小改變，大效益", "UAV 照片上傳時保留拍攝座標與時間（EXIF）、事件目錄附判釋日期，就能省下大量事後對位與比對。"]];
  gp.forEach(([im, h, d], k) => {
    const y = 1.45 + k * 1.75;
    s.addShape(pres.shapes.OVAL, { x: 5.95, y: y + 0.1, w: 0.85, h: 0.85, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: `gpC${k}` });
    s.addImage({ data: im, x: 6.15, y: y + 0.3, w: 0.45, h: 0.45, objectName: `gpI${k}` });
    s.addText([{ text: h, options: { fontSize: 18, bold: true, color: C.text1, breakLine: true } }, { text: d, options: { fontSize: 14, color: C.accent4 } }],
      { x: 7.0, y, w: 5.7, h: 1.5, valign: "middle", margin: 0, isTextBox: true, objectName: `gpT${k}` });
  });
  s.addNotes("對公部門的啟示：我們量了準確度。大事件後，花蓮 75%、太魯閣 43% 的崩塌坡面被抓到；但平時零星的紅邊下降，經分層盲判只有約 25% 屬實；連人看影像和官方圖層的一致率也只有 69%。所以自動訊號只能排先看誰，不能拿來判定，導入前一定要先量。這套流程全用開放資料、可以搬到其他流域；而 UAV 照片上傳時保留座標、事件目錄附判釋日期，這種小改變就能省下大量事後工作。（約 50 秒）");

  // 7 對全民的啟示＋結語
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "啟示" });
  s.addText("對全民的啟示：看得懂、查得到、一起補", { placeholder: "title" });
  s.addImage({ path: "su3d_clean.png", x: 0.6, y: 1.35, w: 5.6, h: 4.62, sizing: { type: "cover", w: 5.6, h: 4.62 }, objectName: "su3d" });
  s.addText("坡面頁 3D 檢視：國土測繪中心航照貼在 20 m 地形上；黃線＝選取的坡面", { x: 0.6, y: 6.0, w: 5.6, h: 0.35, fontSize: 11, color: C.accent4, margin: 0, isTextBox: true, objectName: "su3dCap" });
  const pb = [[ic.eye, "透明，也誠實", "不給「風險分數」，避免把不確定的訊號說成定論；看不到就說看不到，每個數字附誤差。"],
              [ic.home, "查得到身邊的坡", "每個坡面頁列出附近道路、建物、聚落與土石流潛勢溪流，可切 3D 與逐年航照。"],
              [ic.hands, "一起補資料", "山區建物與產業道路在 OpenStreetMap 常缺漏，志工補上的每一筆都讓「會碰到誰」更完整；程式全部開源。"]];
  pb.forEach(([im, h, d], k) => {
    const y = 1.4 + k * 1.5;
    s.addShape(pres.shapes.OVAL, { x: 6.6, y: y + 0.05, w: 0.8, h: 0.8, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: `pbC${k}` });
    s.addImage({ data: im, x: 6.78, y: y + 0.23, w: 0.44, h: 0.44, objectName: `pbI${k}` });
    s.addText([{ text: h, options: { fontSize: 18, bold: true, color: C.text1, breakLine: true } }, { text: d, options: { fontSize: 13.5, color: C.accent4 } }],
      { x: 7.6, y: y - 0.05, w: 5.1, h: 1.35, valign: "middle", margin: 0, isTextBox: true, objectName: `pbT${k}` });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 6.35, w: 12.1, h: 0.55, fill: { color: C.accent1 }, line: { color: C.accent1 }, rectRadius: 0.1, objectName: "closing" });
  s.addText([{ text: "機器幫忙決定先看誰・證據鏈告訴你為什麼・最後由人決定要不要去　", options: { bold: true, color: C.background1 } }, { text: "rhynowu-ardswc-resilience-observatory.hf.space", options: { color: C.background2 } }],
    { x: 0.85, y: 6.35, w: 11.6, h: 0.55, fontSize: 15, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "closingT" });
  s.addNotes("最後，對全民的啟示：坡地資訊要看得懂、查得到、也能一起補。我們不給風險分數，看不到就說看不到；每個坡面頁列出附近的道路、建物和土石流潛勢溪流，還能用 3D 和逐年航照看；山區建物在 OpenStreetMap 常缺漏，志工補上的每一筆都有用，程式全部開源。我們不是重新發明崩塌判釋，而是把既有證據整理成可追蹤的坡面履歷：機器幫忙決定先看誰，證據鏈告訴你為什麼，最後由人決定要不要去。謝謝。（約 40 秒）");


  pres.addSection({ title: "備用" });
  // 8 備用：驗證細節（原主線的驗證頁）
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "備用" });
  s.addText("備用｜驗證細節：訊號只能「請人看」", { placeholder: "title" });
  s.addChart(pres.charts.BAR, [{ name: "被抓到的比例", labels: ["<5%", "5–10%", "10–20%", "20–40%", "≥40%"], values: [7, 15, 42, 75, 93] }],
    { x: 0.6, y: 1.4, w: 4.45, h: 4.4, barDir: "col", showTitle: true, title: "太魯閣：崩塌占坡面越大，越抓得到", titleFontSize: 13, titleColor: HEX.dk1, titleFontFace: "+mn-lt",
      chartColors: [HEX.accent1], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0"%"', dataLabelFontSize: 12, dataLabelColor: HEX.dk1, dataLabelFontFace: "+mn-lt",
      catAxisLabelFontSize: 11, valAxisHidden: true, catAxisLabelColor: HEX.accent4, catAxisLabelFontFace: "+mn-lt", valAxisMinVal: 0, valAxisMaxVal: 100,
      valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false, barGapWidthPct: 60, objectName: "detChart" });
  s.addText("橫軸：事件目錄崩塌占坡面有效面積；縱軸：NDRE 持續下降被抓到的比例（太魯閣 5,469 個坡面）", { x: 0.6, y: 5.9, w: 4.45, h: 0.7, fontSize: 11, color: C.accent4, margin: 0, valign: "top", isTextBox: true, objectName: "chartCap" });
  const val = [
    ["69%", "目視與官方圖層「有無裸露」一致", "人工盲測 n=29；95% CI 51–83%"],
    ["約 1/4", "舊版「近期有變化」訊號屬實", "分層驗證 S1 層 n=4；95% CI 5–70%"],
    ["約 25%", "高雄全區紅邊下降經盲判屬實", "分層加權（5/26）；95% CI 5–45%"],
    ["75%・43%", "大事件後崩塌坡面被抓到（花蓮・太魯閣），誤報約 2%", "95% CI 66–83%・38–48%；誤報 1.9%・2.0%"],
  ];
  val.forEach(([n, l, src_], k) => {
    const x = 5.3 + (k % 2) * 3.8, y = 1.45 + Math.floor(k / 2) * 2.0;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 3.6, h: 1.8, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.12, objectName: `val${k}` });
    s.addText(n, { x: x + 0.2, y: y + 0.12, w: 3.2, h: 0.65, fontSize: 28, bold: true, color: k == 2 ? C.accent2 : C.text2, margin: 0, valign: "middle", isTextBox: true, objectName: `valN${k}` });
    s.addText(l, { x: x + 0.2, y: y + 0.8, w: 3.2, h: 0.6, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: `valL${k}` });
    s.addText(src_, { x: x + 0.2, y: y + 1.4, w: 3.2, h: 0.3, fontSize: 11, color: C.accent4, margin: 0, isTextBox: true, objectName: `valS${k}` });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 5.3, y: 5.5, w: 7.4, h: 1.3, fill: { color: C.text2 }, line: { color: C.text2 }, rectRadius: 0.12, objectName: "concl" });
  s.addText([{ text: "所以：", options: { bold: true, color: C.accent5 } }, { text: "紅邊指標適合大事件後找出植被大片消失的坡面，平時零星變化不可靠；自動訊號只排「先看誰」，最後由人確認。", options: { color: C.background1 } }],
    { x: 5.55, y: 5.55, w: 6.95, h: 1.2, fontSize: 15, valign: "middle", margin: 0, isTextBox: true, objectName: "conclT" });
  s.addNotes("我們不只做系統，也量了它準不準。影像目視與官方圖層的有無裸露一致率只有 69%；舊版近期變化訊號約四分之一屬實。我們參考水保署電子報改用紅邊指標。在高雄全區，零星的下降經分層盲判只有約 25% 屬實，信賴區間 5% 到 45%；但在大事件後，花蓮有 75%、太魯閣有 43% 的崩塌坡面被抓到，沒有事件的坡面誤報約 2%。左邊這張圖是太魯閣：崩塌占坡面不到 5% 時只抓到 7%，超過四成時抓到 93%。所以紅邊指標適合大事件後快速找出植被大片消失的坡面，自動訊號只用來排出先看誰，最後一定由人確認。（約 55 秒）");

  // 9 備用：對照專家的崩塌智慧防災構想（不在 5 分鐘主線，問答時再翻）

  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "備用" });
  s.addText("備用｜對照崩塌智慧防災構想：現在在哪、下一步", { placeholder: "title" });
  const mods = [
    ["一、坡地風險底圖", 1, "20 m DEM、≥15° 有效坡面、事件目錄、四年崩塌圖層；新增 OSM 道路與建物暴露（坡面外推 100 m）", "五萬分之一斷層、位態、順向坡尚缺向量（小比例尺地層見下頁）"],
    ["二、降雨觸發與前期含水", 0, "—", "雨量 I-D 曲線、前期累積雨量、土壤含水"],
    ["三、多源監測與異常偵測", 1, "Sentinel-2 紅邊時間序列、Sentinel-1、現地照片、Wayback 人工查證", "InSAR、微地動、水位；UAV 依輔導建議暫停"],
    ["四、AI 輔助判釋", 1, "三級候選、坡面證據頁、每項訊號的準確度與 95% 信賴區間", "自動產生研判摘要與待確認事項"],
    ["五、下游影響與災害鏈", 1, "坡面所在的土石流潛勢溪流集水區與影響範圍（水保署）", "土砂量、河道、堰塞、下游的動態串接"],
    ["六、應變工作流", 1, "實務人員測試頁：判斷要不要現勘", "事件建立、任務派發、決策紀錄"],
  ];
  mods.forEach(([h, st, have, gap], k) => {
    const x = 0.6 + (k % 3) * 4.1, y = 1.35 + Math.floor(k / 3) * 2.5;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 3.9, h: 2.35, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.1, objectName: `mod${k}` });
    s.addText(h, { x: x + 0.2, y: y + 0.12, w: 2.7, h: 0.45, fontSize: 16, bold: true, color: C.text1, margin: 0, valign: "middle", isTextBox: true, objectName: `modH${k}` });
    const col = st ? C.accent1 : C.accent4;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 2.95, y: y + 0.17, w: 0.75, h: 0.36, fill: { color: col }, line: { color: col }, rectRadius: 0.18, objectName: `modP${k}` });
    s.addText(st ? "部分" : "未做", { x: x + 2.95, y: y + 0.17, w: 0.75, h: 0.36, fontSize: 12, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `modPT${k}` });
    s.addText([{ text: "已有　", options: { bold: true, color: C.text2 } }, { text: have, options: { color: C.text1, breakLine: true } },
               { text: "缺口　", options: { bold: true, color: C.accent4 } }, { text: gap, options: { color: C.accent4 } }],
      { x: x + 0.2, y: y + 0.65, w: 3.5, h: 1.6, fontSize: 14, valign: "top", margin: 0, paraSpaceAfter: 4, isTextBox: true, objectName: `modT${k}` });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 6.3, w: 12.1, h: 0.6, fill: { color: C.text2 }, line: { color: C.text2 }, rectRadius: 0.1, objectName: "posBar" });
  s.addText([{ text: "目前位置：", options: { bold: true, color: C.accent5 } }, { text: "平時篩查（構想第一階段尾、第二階段開頭）　", options: { color: C.background1 } },
             { text: "下一步：", options: { bold: true, color: C.accent5 } }, { text: "取得地質向量、串接降雨觸發與下游影響，走向事件反應", options: { color: C.background1 } }],
    { x: 0.85, y: 6.3, w: 11.6, h: 0.6, fontSize: 14, valign: "middle", margin: 0, isTextBox: true, objectName: "posBarT" });
  s.addNotes("（備用，評審問到系統藍圖或下一步時再翻。）專家提出的崩塌智慧防災平台有六個模組。我們目前做到的是平時篩查：坡地底圖、多源監測和 AI 輔助判釋都有一部分，其中道路與建物暴露是用 OpenStreetMap 疊合，回答坡面崩了會碰到什麼，但不改候選層級。地質目前只拿到影像圖磚，五萬分之一的斷層、位態、順向坡要取得向量後才能分析。下游影響目前只列出坡面所在的土石流潛勢溪流集水區與影響範圍；降雨觸發、土砂與河道的動態串接和應變工作流還沒做，這是從平時篩查走向事件反應的下一步。");

  // 10 備用：各地層崩塌密度（小比例尺地層面，geology_landslide_density.py）
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "備用" });
  s.addText("備用｜地質：哪些地層比較容易崩（粗尺度）", { placeholder: "title" });
  const gd = JSON.parse(require("fs").readFileSync("geology_landslide_density.json", "utf8"));
  const pct = (x, d = 1) => (x * 100).toFixed(d) + "%";
  const fmt = (x, d = 1) => x.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  const hd = ["地層", "年代", "有效坡面 km²", "崩塌 ha", "崩塌密度", "頻率比", "NDRE 下降坡面"].map(t => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 } } }));
  const body = gd.rows.filter(r => !r.small).map(r => {
    const hi = r.fr >= 1.2, lo = r.fr <= 0.6;
    const o = { color: HEX.dk1, fill: { color: hi ? "FBE3D6" : "FFFFFF" }, bold: hi };
    return [`${r.name}（${r.abbrev}）`, r.age, fmt(r.eff_km2), fmt(r.ls_ha, 0), pct(r.density), r.fr.toFixed(2), r.n_su ? `${pct(r.drop_rate)}（${r.n_drop}/${r.n_su.toLocaleString("en-US")}）` : "—"]
      .map((t, k) => ({ text: t, options: { ...o, color: k == 5 && lo ? HEX.accent4 : o.color } }));
  });
  const m = gd.meta;
  body.push(["全區", "", fmt(m.eff_km2), fmt(m.ls_ha, 0), pct(m.density), "1.00", `${pct(m.drop_rate)}（${m.n_drop}/${m.n_su.toLocaleString("en-US")}）`]
    .map(t => ({ text: t, options: { bold: true, color: HEX.dk1, fill: { color: HEX.lt2 } } })));
  s.addTable([hd, ...body], { x: 0.6, y: 1.3, w: 12.1, colW: [3.3, 1.75, 1.35, 1.1, 1.2, 1.0, 2.4], rowH: 0.36, fontSize: 12, fontFace: FONT, margin: [0.02, 0.08, 0.02, 0.08], valign: "middle",
    align: "left", border: { type: "solid", pt: 0.5, color: "DDE5DF" }, objectName: "geoTable" });
  const notes8 = [
    ["讀法", "崩塌密度＝2021–2024 年度崩塌圖層聯集 ÷ ≥15° 有效坡面；頻率比 >1 表示比全區平均容易崩。始新世西村層（板岩）1.53 最高，上新世錦水頁岩 0.32 最低。NDRE 下降比例在面積大的主要地層都約 2%，差異不大（小地層樣本太少）。"],
    ["限制", "公開的只有小比例尺地層面（POC 內一塊中位數約 160 ha），沒有斷層、位態、崩積區；這是描述性統計，不是敏感度模型。斷層距離與順向坡要五萬分之一向量。"],
  ];
  notes8.forEach(([h, t], k) => {
    const x = 0.6 + k * 6.15;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 5.5, w: 5.95, h: 1.4, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.1, objectName: `geoN${k}` });
    s.addText([{ text: h + "　", options: { bold: true, color: k ? C.accent4 : C.text2 } }, { text: t, options: { color: C.text1 } }],
      { x: x + 0.2, y: 5.55, w: 5.55, h: 1.3, fontSize: 12, valign: "middle", margin: 0, isTextBox: true, objectName: `geoNT${k}` });
  });
  s.addNotes("（備用，評審問到地質時再翻。）我們用 BigGIS 引用的公開地層圖，疊上 2021 到 2024 年的年度崩塌圖層，以坡度 15 度以上的有效坡面當分母，算每個地層的崩塌密度。全區約 5.3%；始新世的西村層，也就是板岩，8.1%，是平均的 1.5 倍；上新世的錦水頁岩只有 1.7%。紅邊下降的比例在幾個主要地層都約 2%，差異不大。限制是這份地層圖比例尺小，沒有斷層和位態，只能做描述性統計；斷層距離和順向坡要取得五萬分之一的向量才能分析。");

  await pres.writeFile({ fileName: "TASTI_SatDashboard-決賽簡報.pptx" });
  await applyTheme("TASTI_SatDashboard-決賽簡報.pptx", THEME);
  console.log("done");
})();
