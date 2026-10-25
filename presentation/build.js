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
  pres.addSection({ title: "問題與方法" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "問題與方法" });
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

  // 3 方法
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "問題與方法" });
  s.addText("以坡面為單位，四類證據分開，不給總分", { placeholder: "title" });
  s.addImage({ path: "v2_only.png", x: 0.6, y: 1.45, w: 4.6, h: 4.6, objectName: "suMap" });
  s.addText("集水區重疊法（正、反 DEM）切坡面，只保留坡度 ≥15° 的有效坡面；每個單元都是完整區塊（舊版 85% 不連續 → 0%）",
    { x: 0.6, y: 6.1, w: 4.6, h: 0.75, fontSize: 12, color: C.accent4, margin: 0, valign: "top", isTextBox: true, objectName: "suCap" });
  const ev = [["①", "事件目錄與年度圖層", "主骨架：歷次新增、擴大掛到同一坡面"], ["②", "現地照片", "補充證據，與①分開計、不相加"], ["③", "近期衛星變化", "Sentinel-2 時間序列，只回答「最近還有沒有變化」"], ["④", "影像品質提醒", "雲、陰影、有效觀測筆數、Wayback 偏移"]];
  ev.forEach(([n, h, d], k) => {
    const y = 1.5 + k * 0.95;
    s.addShape(pres.shapes.OVAL, { x: 5.7, y: y + 0.05, w: 0.6, h: 0.6, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: `evDot${k}` });
    s.addText(n, { x: 5.7, y: y + 0.05, w: 0.6, h: 0.6, fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `evN${k}` });
    s.addText([{ text: h, options: { fontSize: 18, bold: true, color: C.text1, breakLine: true } }, { text: d, options: { fontSize: 14, color: C.accent4 } }],
      { x: 6.5, y, w: 6.2, h: 0.8, valign: "middle", margin: 0, isTextBox: true, objectName: `evT${k}` });
  });
  const tiers = [["優先人工判讀", C.accent2], ["一般判讀", C.accent1], ["資料不足", C.accent4]];
  s.addText("只給三種候選層級，由人決定是否現勘：", { x: 5.7, y: 5.4, w: 7, h: 0.4, fontSize: 15, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: "tierLead" });
  tiers.forEach(([t, col], k) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 5.7 + k * 2.35, y: 5.95, w: 2.15, h: 0.6, fill: { color: col }, line: { color: col }, rectRadius: 0.3, objectName: `tier${k}` });
    s.addText(t, { x: 5.7 + k * 2.35, y: 5.95, w: 2.15, h: 0.6, fontSize: 15, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `tierT${k}` });
  });
  s.addNotes("方法上，我們以坡面為單位。依輔導委員建議，用集水區重疊法切坡面，只保留坡度 15 度以上的有效坡面，每個單元都是完整區塊。每個坡面有四類證據：事件目錄是主骨架，照片是補充，衛星只回答最近還有沒有變化，再加上影像品質提醒。四類分開呈現、不相加、不給總分，最後只給三種候選層級。（約 50 秒）");

  // 4 案例：太魯閣
  pres.addSection({ title: "成果與驗證" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "成果與驗證" });
  s.addText("案例：太魯閣 0403 地震，一個坡面的證據鏈", { placeholder: "title" });
  s.addImage({ path: "taroko_s2.gif", x: 0.6, y: 1.4, w: 4.75, h: 4.75, objectName: "tarokoGif" });
  s.addText("白線＝坡面；橘線＝事件目錄的地震崩塌（官方判釋）", { x: 0.6, y: 6.25, w: 4.75, h: 0.5, fontSize: 12, color: C.accent4, margin: 0, valign: "top", isTextBox: true, objectName: "gifCap" });
  const ts = JSON.parse(require("fs").readFileSync("taroko_ndre_series.json", "utf8"));
  const lab4 = ts.months.map(m => m.slice(2).replace("-", "/"));
  s.addChart(pres.charts.LINE, [
    { name: "此坡面", labels: lab4, values: ts.su.map(v => v == null ? null : v) },
    { name: "全區中位數", labels: lab4, values: ts.region },
  ], { x: 5.75, y: 1.35, w: 6.95, h: 3.05, showTitle: true, title: "NDRE 逐月中位數（地震 2024/04）", titleFontSize: 14, titleColor: HEX.dk1, titleFontFace: "+mn-lt",
       chartColors: [HEX.accent2, HEX.accent4], lineSize: 2.5, lineDataSymbol: "none", displayBlanksAs: "span",
       showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: "+mn-lt", catAxisLabelFontSize: 10, valAxisLabelFontSize: 10, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
       catAxisLabelColor: HEX.accent4, valAxisLabelColor: HEX.accent4, valAxisMinVal: 0, valAxisMaxVal: 0.8, valAxisMajorUnit: 0.2, valGridLine: { color: "DDE5DF", size: 0.5 }, catGridLine: { style: "none" },
       catAxisLabelFrequency: 3, objectName: "ndreChart" });
  const chain = [["①", "事件目錄", "0403 地震崩塌在此坡面 10.7 ha（有效坡面 21.4 ha 的一半）"], ["③", "近期衛星變化", "NDRE 自 2024-04-14 持續下降（0.53 → 0.20），全區沒有同步"], ["→", "結論", "列入人工判讀候選；是否現勘由人決定"]];
  chain.forEach(([n, h, d], k) => {
    const y = 4.6 + k * 0.72;
    s.addShape(pres.shapes.OVAL, { x: 5.75, y: y + 0.06, w: 0.48, h: 0.48, fill: { color: k == 2 ? C.accent2 : C.accent1 }, line: { color: k == 2 ? C.accent2 : C.accent1 }, objectName: `chD${k}` });
    s.addText(n, { x: 5.75, y: y + 0.06, w: 0.48, h: 0.48, fontSize: 14, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `chN${k}` });
    s.addText([{ text: h + "　", options: { bold: true, color: C.text1 } }, { text: d, options: { color: C.text1 } }], { x: 6.4, y, w: 6.3, h: 0.6, fontSize: 14, valign: "middle", margin: 0, isTextBox: true, objectName: `chT${k}` });
  });
  s.addNotes("看一個太魯閣的例子。白線是一個坡面，橘線是官方事件目錄圈出的地震崩塌，占了這個坡面的一半。地震前 3 月底，紅邊指標 0.53；地震後 11 天掉到 0.20，之後一直沒有回來，而全區中位數只是正常的季節起伏。事件目錄、衛星變化兩類證據指向同一件事，這個坡面就會被列入人工判讀候選，最後由人決定要不要去。線上系統的太魯閣頁可以看到全部 329 個下降坡面。（約 45 秒）");

  // 5 驗證
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "成果與驗證" });
  s.addText("我們量了它準不準：訊號只能「請人看」", { placeholder: "title" });
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

  // 6 落地
  pres.addSection({ title: "落地" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "落地" });
  s.addText("可複製、可檢驗，接下來交給第一線檢驗", { placeholder: "title" });
  const nx = [
    [ic.open, "全用開放資料與開源工具", "水保署 BigGIS、歷史影像平台、內政部 20 m DEM、Sentinel-2；程式公開在 GitHub"],
    [ic.map, "流程可搬到其他流域", "高雄完成完整證據鏈；花蓮、太魯閣加做坡面紅邊檢核；中部、嘉義已測試事件目錄與衛星比對"],
    [ic.people, "人工驗證持續進行", "專家盲判 2–3 位、實務人員操作測試 3–5 位；依結果調整候選規則"],
  ];
  nx.forEach(([im, h, d], k) => {
    const y = 1.5 + k * 1.45;
    s.addShape(pres.shapes.OVAL, { x: 0.6, y, w: 0.95, h: 0.95, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: `nxC${k}` });
    s.addImage({ data: im, x: 0.82, y: y + 0.22, w: 0.5, h: 0.5, objectName: `nxI${k}` });
    s.addText([{ text: h, options: { fontSize: 20, bold: true, color: C.text1, breakLine: true } }, { text: d, options: { fontSize: 15, color: C.accent4 } }],
      { x: 1.85, y: y - 0.05, w: 10.8, h: 1.1, valign: "middle", margin: 0, isTextBox: true, objectName: `nxT${k}` });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 5.95, w: 12.1, h: 0.8, fill: { color: C.accent1 }, line: { color: C.accent1 }, rectRadius: 0.12, objectName: "closing" });
  s.addText("機器幫忙決定先看誰・證據鏈告訴你為什麼・最後由人決定要不要去", { x: 0.9, y: 5.95, w: 11.5, h: 0.8, fontSize: 20, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "closingT" });
  s.addNotes("最後，這套流程全用開放資料與開源工具，程式公開，高雄完成完整流程，花蓮、太魯閣也加做了坡面紅邊檢核，另外兩區測試過衛星與事件比對。接下來請專家盲判與第一線同仁實際操作，依結果調整規則。我們不是重新發明崩塌判釋，而是把既有證據整理成可追蹤的坡面履歷：機器幫忙決定先看誰，證據鏈告訴你為什麼，最後由人決定要不要去。謝謝。（約 40 秒）");

  // 7 備用：對照專家的崩塌智慧防災構想（不在 5 分鐘主線，問答時再翻）
  pres.addSection({ title: "備用" });
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

  // 8 備用：各地層崩塌密度（小比例尺地層面，geology_landslide_density.py）
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
