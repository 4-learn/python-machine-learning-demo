// 〈關於這門課〉的兩張概念圖：開發流程、工具分工。風格與 story/diagrams/gen.js 一致（扁平、細線、灰＋單一強調色）。
const fs = require("fs");
const SANS = "'PingFang TC','Noto Sans TC','Noto Sans CJK TC','Microsoft JhengHei',sans-serif";
const MONO = "'SF Mono',Menlo,Consolas,'DejaVu Sans Mono',monospace";
const C = {
  ink: "#1f2328", sub: "#57606a", line: "#8a9099", box: "#f4f5f7", stroke: "#a3a8b0",
  sk: { fill: "#e8f1fb", stroke: "#3b7dd8", text: "#174a8b" },      // scikit-learn
  pt: { fill: "#fdeee6", stroke: "#e0672b", text: "#8a3510" },      // PyTorch
  ul: { fill: "#eef6ee", stroke: "#3f9a50", text: "#1d5a29" },      // ultralytics
  key: { fill: "#fff7e0", stroke: "#d9a21a", text: "#6b4d00" },     // 重點
};
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
function text(x, y, s, o = {}) {
  return `<text x="${x}" y="${y}" font-family="${o.mono ? MONO : SANS}" font-size="${o.size || 14}" font-weight="${o.bold ? 700 : 400}" fill="${o.fill || C.ink}" text-anchor="${o.anchor || "middle"}" dominant-baseline="middle">${esc(s)}</text>`;
}
function box(x, y, w, h, lines, o = {}) {
  const st = o.style || { fill: C.box, stroke: C.stroke, text: C.ink };
  let s = `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="${st.fill}" stroke="${st.stroke}" stroke-width="${o.bw || 1.4}"${o.dash ? ' stroke-dasharray="5 4"' : ""}/>`;
  const n = lines.length, lh = o.lh || 19, top = y + h / 2 - ((n - 1) * lh) / 2;
  lines.forEach((l, i) => {
    const first = i === 0;
    s += text(x + w / 2, top + i * lh, l, { size: first ? (o.size || 15) : 12.5, bold: first, fill: first ? st.text : C.sub, mono: !first && o.mono });
  });
  return s;
}
function arrow(pts, o = {}) {
  const d = pts.map((p, i) => (i ? "L" : "M") + p[0] + " " + p[1]).join(" ");
  let s = `<path d="${d}" fill="none" stroke="${o.color || C.line}" stroke-width="${o.w || 1.6}"${o.dash ? ' stroke-dasharray="5 4"' : ""} marker-end="url(#a${o.mk || ""})"/>`;
  if (o.label) s += text(o.label[0], o.label[1], o.label[2], { size: 12, fill: C.sub, anchor: o.anchor || "middle" });
  return s;
}
function svg(w, h, title, sub, body) {
  const marker = (id, col) => `<marker id="a${id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="${col}"/></marker>`;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
<defs>${marker("", C.line)}${marker("k", C.key.stroke)}</defs>
<rect width="${w}" height="${h}" fill="#ffffff"/>
${text(32, 34, title, { size: 22, bold: true, anchor: "start" })}
${text(32, 62, sub, { size: 13, fill: C.sub, anchor: "start" })}
${body}
</svg>`;
}

// ---------------------------------------------------------------- 圖 1：開發流程
function flow() {
  let b = "";
  const Y = 120, H = 64;
  b += box(32, Y, 150, H, ["① 看資料", "pandas"]);
  arrow; b += arrow([[182, Y + H / 2], [222, Y + H / 2]]);
  b += box(216, Y, 182, H, ["② 決定候選", "scikit-learn 官方的", "選模型地圖"], { lh: 17 });
  // 兩條候選
  b += box(450, 92, 210, 50, ["表格：樹模型、邏輯迴歸", "scikit-learn、XGBoost"], { style: C.sk, size: 13.5 });
  b += box(450, 162, 210, 50, ["影像、文字：預訓練模型", "PyTorch、ultralytics"], { style: C.pt, size: 13.5 });
  b += arrow([[392, Y + H / 2], [420, Y + H / 2], [420, 117], [450, 117]]);
  b += arrow([[420, Y + H / 2], [420, 187], [450, 187]]);
  // baseline
  b += box(450, 232, 210, 44, ["baseline：不用模型能拿幾分"], { size: 13.5, dash: true });
  b += arrow([[420, 187], [420, 254], [450, 254]]);
  // 同一把尺
  b += `<rect x="712" y="84" width="236" height="200" rx="10" fill="${C.key.fill}" stroke="${C.key.stroke}" stroke-width="1.8"/>`;
  b += text(732, 112, "③ 用同一把尺量", { size: 16, bold: true, fill: C.key.text, anchor: "start" });
  ["同一種切法（不偷看考卷）", "同一個指標（precision／recall）", "同一份 valid、test 只開一次"].forEach((s, i) => (b += text(732, 148 + i * 26, "· " + s, { size: 13, fill: C.ink, anchor: "start" })));
  b += text(732, 236, "多半用 scikit-learn 的工具", { size: 12.5, fill: C.sub, anchor: "start" });
  b += text(732, 258, "（偵測的 IoU、mAP 用 ultralytics）", { size: 12, fill: C.sub, anchor: "start" });
  [117, 187, 254].forEach((y) => (b += arrow([[660, y], [712, y]], { color: C.key.stroke, mk: "k" })));
  // 選 → 上線
  b += arrow([[948, Y + H / 2 + 32], [990, Y + H / 2 + 32]], { color: C.key.stroke, mk: "k" });
  b += box(990, Y + 6, 150, 52, ["④ 選一個", "比完才決定"], { size: 15 });
  b += arrow([[1065, Y + 58], [1065, 230]]);
  b += box(990, 230, 150, 52, ["⑤ 定門檻、上線", "持續監控"], { size: 15 });
  b += arrow([[1065, 282], [1065, 320], [107, 320], [107, Y + H]], { dash: true, label: [586, 336, "上線後資料變了（漂移）→ 回到 ① 重新評估"] });
  // 對應節次
  b += text(32, 384, "這門課：", { size: 12.5, bold: true, fill: C.sub, anchor: "start" });
  b += text(96, 384, "① 01–02　② 03、05–08、17–18、20　③ 04、09–12　④ 10、19　⑤ 12、15、22–24（進階：34 漂移、41 驗收）", { size: 12.5, fill: C.sub, anchor: "start" });
  return svg(1172, 410, "業界的機器學習開發流程", "", b);
}

// ---------------------------------------------------------------- 圖 2：工具分工
function roles() {
  let b = "";
  // 資料
  b += box(32, 110, 150, 70, ["原始資料", "影像、聲音、文字"]);
  b += box(32, 290, 150, 70, ["表格資料", "整理好的欄位"]);
  // PyTorch 層
  b += `<rect x="232" y="92" width="300" height="106" rx="10" fill="${C.pt.fill}" stroke="${C.pt.stroke}" stroke-width="1.6"/>`;
  b += text(382, 116, "PyTorch／torchvision", { size: 16, bold: true, fill: C.pt.text });
  b += text(382, 142, "把像素、聲音變成特徵", { size: 13 });
  b += text(382, 164, "（大多用別人預訓練好的模型）", { size: 12.5, fill: C.sub });
  b += text(382, 184, "大型模型、LLM 也在這一層", { size: 12, fill: C.sub });
  b += arrow([[182, 145], [232, 145]]);
  // ultralytics（建在 PyTorch 上）
  b += `<rect x="232" y="208" width="300" height="60" rx="10" fill="${C.ul.fill}" stroke="${C.ul.stroke}" stroke-width="1.6"/>`;
  b += text(382, 230, "ultralytics（建在 PyTorch 上）", { size: 15, bold: true, fill: C.ul.text });
  b += text(382, 252, "物件偵測、骨架：一行就能訓練、推論", { size: 12.5 });
  b += arrow([[182, 160], [207, 160], [207, 238], [232, 238]]);
  // sklearn 層
  b += `<rect x="600" y="92" width="300" height="268" rx="10" fill="${C.sk.fill}" stroke="${C.sk.stroke}" stroke-width="1.6"/>`;
  b += text(626, 116, "scikit-learn", { size: 16, bold: true, fill: C.sk.text, anchor: "start" });
  b += text(626, 146, "從特徵學判斷", { size: 13.5, bold: true, anchor: "start" });
  b += text(626, 168, "樹模型、邏輯迴歸、KNN…", { size: 12.5, fill: C.sub, anchor: "start" });
  b += text(626, 200, "偵測結果、骨架也可以再交給它判斷", { size: 12.5, fill: C.sub, anchor: "start" });
  b += `<line x1="626" y1="262" x2="874" y2="262" stroke="${C.sk.stroke}" stroke-opacity="0.35"/>`;
  b += text(626, 286, "評估：所有候選的同一把尺", { size: 13.5, bold: true, anchor: "start" });
  b += text(626, 308, "切割、交叉驗證、指標、門檻", { size: 12.5, fill: C.sub, anchor: "start" });
  b += text(626, 330, "（PyTorch 做的模型也用它量）", { size: 12.5, fill: C.sub, anchor: "start" });
  b += arrow([[532, 145], [600, 145]], { label: [566, 134, "特徵"] });
  b += arrow([[182, 325], [600, 325]], { label: [391, 313, "表格直接進來"] });
  b += arrow([[532, 238], [600, 238]], { dash: true, label: [566, 226, "偵測結果"] });
  // 決定
  b += box(960, 188, 180, 76, ["決定", "要不要叫人", "（判斷器、門檻）"], { lh: 20 });
  b += arrow([[900, 226], [960, 226]]);
  // 註
  b += text(32, 400, "不是互相取代：表格資料上神經網路通常沒有比較準（這門課實測：判斷器用 PyTorch 神經網路 AP 0.80–0.85，scikit-learn 梯度提升 0.87）", { size: 12.5, fill: C.sub, anchor: "start" });
  b += text(32, 422, "這門課：scikit-learn 03–15、19、35–40　PyTorch／torchvision 17–18、28　ultralytics 20–21、29", { size: 12.5, fill: C.sub, anchor: "start" });
  return svg(1172, 446, "工具分工：scikit-learn、PyTorch、ultralytics", "", b);
}
fs.writeFileSync("concept-flow.svg", flow());
fs.writeFileSync("concept-tools.svg", roles());
console.log("ok");
