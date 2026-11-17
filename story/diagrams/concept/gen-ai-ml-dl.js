// 00 導論的圖：人工智慧、機器學習、深度學習的包含關係，以及這門課用到的工具。風格同 concept/gen.js。
const fs = require("fs");
const SANS = "'PingFang TC','Noto Sans TC','Noto Sans CJK TC','Microsoft JhengHei',sans-serif";
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
const t = (x, y, s, o = {}) => `<text x="${x}" y="${y}" font-family="${SANS}" font-size="${o.size || 13}" font-weight="${o.bold ? 700 : 400}" fill="${o.fill || "#1f2328"}" text-anchor="${o.anchor || "start"}" dominant-baseline="middle">${esc(s)}</text>`;
const r = (x, y, w, h, fill, stroke, o = {}) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${o.rx || 12}" fill="${fill}" stroke="${stroke}" stroke-width="${o.bw || 1.6}"${o.dash ? ' stroke-dasharray="5 4"' : ""}/>`;
const SUB = "#57606a";
let b = "";
// 人工智慧
b += r(32, 80, 1108, 420, "#f4f5f7", "#a3a8b0");
b += t(56, 108, "人工智慧（AI）", { size: 17, bold: true });
b += t(56, 134, "讓機器做出看起來聰明的決定。寫死的規則也算。", { fill: SUB });
b += r(56, 160, 230, 70, "#ffffff", "#a3a8b0", { rx: 8 });
b += t(72, 184, "規則", { bold: true, size: 14 });
b += t(72, 208, "平台的 0.8／0.5（01 節）", { fill: SUB, size: 12.5 });
// 機器學習
b += r(320, 156, 796, 324, "#e8f1fb", "#3b7dd8");
b += t(344, 184, "機器學習（ML）", { size: 17, bold: true, fill: "#174a8b" });
b += t(344, 210, "不寫規則，從資料學出判斷。", { fill: SUB });
b += r(344, 236, 300, 104, "#ffffff", "#3b7dd8", { rx: 8 });
b += t(360, 260, "傳統機器學習", { bold: true, size: 14 });
b += t(360, 284, "樹模型、邏輯迴歸、KNN…", { fill: SUB, size: 12.5 });
b += t(360, 306, "scikit-learn：判斷器（03–15、19、35–40）", { fill: SUB, size: 12.5 });
b += t(360, 326, "擅長整理好的表格", { fill: SUB, size: 12.5 });
// 深度學習
b += r(672, 236, 420, 220, "#fdeee6", "#e0672b");
b += t(696, 262, "深度學習（DL）", { size: 17, bold: true, fill: "#8a3510" });
b += t(696, 288, "用很多層的神經網路學習；擅長影像、聲音、文字。", { fill: SUB, size: 12.5 });
b += r(696, 308, 180, 128, "#ffffff", "#e0672b", { rx: 8 });
b += t(710, 330, "預訓練模型", { bold: true, size: 14 });
b += t(710, 354, "PyTorch／torchvision", { fill: SUB, size: 12.5 });
b += t(710, 374, "（17–18、28）", { fill: SUB, size: 12.5 });
b += t(710, 396, "ultralytics YOLO", { fill: SUB, size: 12.5 });
b += t(710, 416, "（20–21、29）", { fill: SUB, size: 12.5 });
b += r(890, 308, 180, 128, "#ffffff", "#e0672b", { rx: 8, dash: true });
b += t(904, 330, "LLM", { bold: true, size: 14 });
b += t(904, 354, "大型語言模型", { fill: SUB, size: 12.5 });
b += t(904, 374, "下一門課", { fill: SUB, size: 12.5 });
// 神經網路註
b += t(344, 372, "神經網路＝一種模型（一層一層的數字運算）", { size: 12.5, bold: true });
b += t(344, 394, "層數少：也算傳統機器學習（sklearn 的 MLPClassifier）", { fill: SUB, size: 12.5 });
b += t(344, 414, "層數多、用大量資料訓練：就是深度學習", { fill: SUB, size: 12.5 });
const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1172" height="530" viewBox="0 0 1172 530"><rect width="1172" height="530" fill="#ffffff"/>${t(32, 40, "人工智慧、機器學習、深度學習：一層包一層", { size: 22, bold: true })}${b}</svg>`;
fs.writeFileSync("ai-ml-dl.svg", svg);
console.log("ok");
