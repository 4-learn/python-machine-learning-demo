// 架構圖產生器（Python 機器學習 2026 v2）：一份定義，產出每一節的 SVG。
// 繪圖引擎沿用 fastapi-workshop/story/diagrams/gen.js，只換掉元件定義。
//
// 每個元件記著 since（第幾節出現）、until（最後出現在第幾節，可省略）、
// changed（哪幾節改過）。since 是 "00" 的是一開始就有的告警平台，永遠是灰色。
//
// 用法：node gen.js 12 > v12.svg
//       node gen.js all          （在這個資料夾產出 v01–v42）

const VERS = ["00", ...Array.from({ length: 42 }, (_, i) => String(i + 1).padStart(2, "0"))];

const TITLES = {
  "01": ["平台規則準不準", "pandas crosstab"],
  "02": ["挑選特徵：哪些欄位模型可以看", "pandas"],
  "03": ["切出考卷", "train_test_split、GroupShuffleSplit"],
  "04": ["不用模型能拿幾分", "DummyClassifier"],
  "05": ["第一個模型", "KNeighborsClassifier"],
  "06": ["前處理接成一條", "Pipeline"],
  "07": ["輸出機率與係數", "LogisticRegression"],
  "08": ["判斷器學會寫規則", "決策樹與集成"],
  "09": ["分數要考五次", "cross_validate、StratifiedGroupKFold"],
  "10": ["調參，考卷只開一次", "GridSearchCV"],
  "11": ["錯誤地圖", "confusion_matrix、classification_report"],
  "12": ["0.8 和 0.5 從哪裡來", "precision_recall_curve"],
  "13": ["模型看什麼、錯在哪", "permutation_importance"],
  "14": ["沒有標籤的時候", "KMeans、IsolationForest"],
  "15": ["判斷器 MVP", "Workshop 1"],
  "16": ["影像就是陣列", "Pillow、NumPy"],
  "17": ["tensor 與預訓練模型", "PyTorch"],
  "18": ["比較預訓練骨幹", "torchvision get_model＋KNN"],
  "19": ["linear probe", "預訓練特徵＋LogisticRegression"],
  "20": ["物件偵測推論", "ultralytics YOLO"],
  "21": ["偵測框對不對", "IoU 與偵測門檻"],
  "22": ["偵測結果送進平台", "requests"],
  "23": ["從攝影機到告警", "Workshop 2"],
  "24": ["交接：凍結評測集", "hashlib"],
  "25": ["讀圖與 ROI", "OpenCV"],
  "26": ["ROI 有沒有被占用", "OpenCV threshold、findContours"],
  "27": ["從攝影機到陣列", "EZCam"],
  "28": ["預訓練第一層學到什麼", "torchvision resnet18.conv1"],
  "29": ["17 個關鍵點", "ultralytics YOLO-pose"],
  "30": ["姿勢特徵與時間窗口", "NumPy"],
  "31": ["偵測串流整理成每分鐘一格", "pandas resample"],
  "32": ["只看過去的特徵", "pandas rolling"],
  "33": ["對齊兩個時間來源", "pandas merge_asof"],
  "34": ["分布有沒有變", "pandas quantile、NumPy histogram"],
  "35": ["預測下一班的告警量", "LinearRegression"],
  "36": ["時間序列的切割與殘差", "TimeSeriesSplit"],
  "37": ["資料還要收多少", "learning_curve"],
  "38": ["調參預算", "RandomizedSearchCV"],
  "39": ["少數類", "class_weight"],
  "40": ["降維", "PCA"],
  "41": ["判斷器的自動驗收", "pytest"],
  "42": ["OpenCV 與骨架事件也送進平台", "Workshop 3"],
  full: ["全貌：做完這門課會長這樣", "01–24 兩班必修；小標籤 v25 以後是 AI 新秀"],
};

let FULL = false;

const vi = (v) => VERS.indexOf(v);

const W = 1340, H = 924;
const SANS = "'PingFang TC','Noto Sans TC','Noto Sans CJK TC','Microsoft JhengHei',sans-serif";
const MONO = "'SF Mono',Menlo,Consolas,'DejaVu Sans Mono',monospace";

const PALETTE = {
  old: { fill: "#f4f5f7", stroke: "#a3a8b0", text: "#3a3f47", line: "#8a9099" },
  new: { fill: "#e3f7e8", stroke: "#1f9d4c", text: "#0f5132", line: "#1f9d4c" },
  changed: { fill: "#fff4dc", stroke: "#d9901a", text: "#7a4b00", line: "#d9901a" },
  error: { fill: "#fdecec", stroke: "#d33a3a", text: "#9b1c1c", line: "#d33a3a" },
};

const E = [];
function el(id, kind, x, y, w, h, lines, since, opts = {}) {
  if (typeof lines[0] === "string") lines = [[since, lines]];
  E.push({ id, kind, x, y, w, h, lines, since, until: null, changed: [], badge: since !== "00", lock: null, ...opts });
}
const ED = [];
function edge(pts, since, opts = {}) {
  let label = opts.label || null;
  if (typeof label === "string") label = [[since, label]];
  ED.push({ pts, since, until: null, changed: [], dashed: false, error: false, arrow: true,
    lpos: null, lanchor: "middle", ...opts, label });
}

// ---------- 外框 ----------
el("pytest", "frame", 186, 470, 846, 446, ["pytest：判斷器的自動驗收"], "41", { dashed: true });
el("f_platform", "frame", 670, 112, 340, 336, ["告警平台"], "00", { badge: false });
el("f_det", "frame", 196, 112, 404, 336, ["偵測器"], "16", { badge: false });
el("f_judge", "frame", 196, 480, 824, 400, ["告警判斷器"], "01", { badge: false });

// ---------- 左邊：資料從哪裡來 ----------
el("a_dev", "actor", 24, 86, 150, 140, ["設備", "攝影機／手環"], "00", { badge: false });
el("a_ezcam", "actor", 24, 270, 150, 64, ["EZCam", "被叫才拍"], "27");

// ---------- 偵測器（16–30） ----------
el("d_array", "service", 212, 152, 118, 46, ["影像陣列", "Pillow／NumPy"], "16");
el("d_torch", "service", 340, 152, 118, 46, ["預訓練模型", "PyTorch"], "17");
el("d_loop", "service", 468, 152, 118, 46, ["選骨幹", "get_model＋KNN"], "18");
el("d_cls", "service", 212, 210, 118, 46, ["影像分類器", "linear probe"], "19");
el("d_yolo", "service", 340, 210, 118, 46, ["YOLO 偵測", "ultralytics"], "20");
el("d_iou", "service", 468, 210, 118, 46, ["偵測門檻", "IoU"], "21");
el("d_conv", "service", 212, 268, 118, 46, ["第一層", "學出來的濾波器"], "28");
el("d_cv", "service", 340, 268, 118, 46, [["25", ["OpenCV", "讀圖、ROI"]], ["26", ["OpenCV", "ROI 被占用"]]], "25", { changed: ["26"] });
el("d_pose", "service", 212, 326, 118, 46, ["YOLO-pose", "17 個關鍵點"], "29");
el("d_posef", "service", 340, 326, 118, 46, ["姿勢特徵", "時間窗口"], "30");
el("d_send", "service", 468, 326, 118, 46,
  [["22", ["送事件", "requests.post"]], ["42", ["送事件", "＋ROI／骨架事件"]]], "22", { changed: ["23", "42"] });

// ---------- 告警平台（灰色，已完成） ----------
el("p_events", "port", 686, 158, 150, 30, ["POST /events"], "00", { badge: false });
el("p_guard", "service", 846, 154, 148, 40,
  [["00", ["防呆分流", "信心值 0.8／0.5"]], ["23", ["防呆分流", "問判斷器"]]], "00", { badge: false, changed: ["23"] });
el("p_sm", "service", 686, 210, 308, 60, ["告警狀態機", "open → acknowledged → resolved／dismissed"], "00", { badge: false });
el("p_prof", "chip", 686, 286, 308, 26,
  [["00", ["profiles/<客戶>.json"]], ["10", ["profiles ＋ 模型參數"]], ["12", ["profiles ＋ 模型參數 ＋ 兩個門檻"]]],
  "00", { badge: false, changed: ["10", "12"] });
el("p_alerts", "port", 686, 326, 308, 30, ["GET /alerts、ack、resolve、dismiss"], "00", { badge: false });
el("n_platform", "note", 690, 392, 0, 0, ["每一筆告警最後都會被處理成 resolved 或 dismissed"], "00",
  { badge: false, anchor: "start", size: 10.5 });

// ---------- 右邊：用結果的人 ----------
el("a_handler", "actor", 1060, 112, 250, 200, ["值班人員", "主管／護理師"], "00", { badge: false });
el("a_manager", "actor", 1060, 330, 250, 64, ["主管", "排下一班的人力"], "35");
el("a_llm", "external", 1060, 816, 250, 60, ["LLM 課", "用同一份考卷比較"], "24");

// ---------- 判斷器（01–15、31–41） ----------
el("j_stream", "chip", 376, 494, 330, 24,
  [["31", ["串流 → 每分鐘一格（resample）"]], ["32", ["串流 → resample ＋ 只看過去（rolling）"]],
   ["33", ["串流 → resample ＋ rolling ＋ 對齊（merge_asof）"]]], "31", { changed: ["32", "33"] });
el("j_alerts", "store", 212, 540, 150, 50,
  [["01", ["告警歷史", "alerts.csv"]], ["31", ["告警歷史", "自己從串流整理"]]], "01", { changed: ["31", "32", "33"] });
el("j_rule", "service", 376, 540, 150, 50, ["平台規則", "pd.cut＋crosstab"], "01");
el("j_xy", "service", 540, 540, 130, 50, ["X 和 y", "去掉事後欄位"], "02");
el("j_split", "service", 684, 540, 150, 50, ["切出考卷", "fit／valid／test"], "03");
el("j_cv", "service", 848, 540, 156, 50, ["交叉驗證", "5 折、依事件分組"], "09");

el("j_pre", "service", 212, 610, 150, 50, ["前處理", "Pipeline"], "06");
el("j_model", "service", 376, 610, 180, 50,
  [["05", ["模型", "KNN"]], ["07", ["模型", "邏輯迴歸"]], ["08", ["模型", "梯度提升"]], ["39", ["模型", "梯度提升＋class_weight"]]],
  "05", { changed: ["07", "08", "39"] });
el("j_grid", "service", 570, 610, 150, 50,
  [["10", ["調參", "GridSearchCV"]], ["38", ["調參", "＋RandomizedSearchCV"]]], "10", { changed: ["38"] });
el("j_lc", "service", 734, 610, 130, 50, ["學習曲線", "資料夠不夠"], "37");
el("j_pca", "service", 878, 610, 126, 50, ["降維", "PCA"], "40");

el("j_board", "service", 212, 680, 150, 50,
  [["04", ["成績表", "baseline"]], ["05", ["成績表", "＋模型"]], ["11", ["成績表", "precision／recall"]]],
  "04", { changed: ["05", "07", "08", "10", "11"] });
el("j_thr", "service", 376, 680, 180, 50, ["兩個門檻", "HIGH／LOW"], "12");
el("j_err", "service", 570, 680, 150, 50, ["錯誤分析", "permutation"], "13");
el("j_drift", "service", 734, 680, 130, 50, ["漂移監控", "分位數＋PSI"], "34");
el("j_rare", "service", 878, 680, 126, 50, ["罕見告警", "IsolationForest"], "14");

el("j_eval", "store", 212, 750, 150, 54, ["凍結評測集", "sha256 指紋"], "24");
el("j_fore", "service", 376, 750, 324, 54, [["35", ["告警量預測", "LinearRegression"]], ["36", ["告警量預測", "LinearRegression＋TimeSeriesSplit"]]],
  "35", { changed: ["36"] });
el("j_save", "store", 714, 750, 150, 54, ["judge.joblib", "＋judge.json"], "15");
el("j_decide", "service", 878, 750, 126, 54, ["decide(event)", "三種決定"], "15", { changed: ["23"] });

// --------------------------------------------------
// 箭頭（箭頭 = 資料往哪裡走）
// --------------------------------------------------
edge([[174, 98], [650, 98], [650, 170], [686, 170]], "00", { label: "送事件", lpos: [420, 92] });
edge([[174, 175], [212, 175]], "16", { label: "照片", lpos: [193, 189] });
edge([[174, 302], [190, 302], [190, 320], [399, 320], [399, 314]], "27");
edge([[330, 175], [340, 175]], "17");
edge([[458, 175], [468, 175]], "18");
edge([[399, 198], [399, 204], [271, 204], [271, 210]], "19");
edge([[458, 233], [468, 233]], "21");
edge([[330, 349], [340, 349]], "30");
edge([[271, 256], [271, 262], [527, 262]], "22", { arrow: false });
edge([[527, 256], [527, 326]], "22");
edge([[458, 291], [480, 291], [480, 326]], "26");
edge([[458, 349], [468, 349]], "42");
edge([[586, 349], [630, 349], [630, 180], [686, 180]], "22", { label: "requests.post", lpos: [640, 340], lanchor: "start", changed: ["23"] });

edge([[994, 236], [1060, 236]], "00", { label: "通知", lpos: [1027, 230] });
edge([[740, 448], [740, 466], [340, 466], [340, 540]], "01", { label: "10 週的處理紀錄", lpos: [560, 461] });
edge([[362, 565], [376, 565]], "01");
edge([[526, 565], [540, 565]], "02");
edge([[670, 565], [684, 565]], "03");
edge([[834, 565], [848, 565]], "09");
edge([[362, 635], [376, 635]], "06");
edge([[556, 635], [570, 635]], "10");
edge([[287, 610], [287, 590]], "06", { arrow: false });
edge([[466, 660], [466, 680]], "12");
edge([[556, 705], [570, 705]], "13");
edge([[466, 730], [466, 740], [789, 740], [789, 750]], "15", { label: "模型＋門檻存檔", lpos: [628, 735] });
edge([[864, 777], [878, 777]], "15");
edge([[1004, 777], [1024, 777], [1024, 174], [994, 174]], "23", { label: "decide(event)", lpos: [1030, 700], lanchor: "start" });
edge([[1004, 705], [1044, 705], [1044, 290], [1060, 290]], "14", { label: "交給人看", lpos: [1050, 660], lanchor: "start" });
edge([[538, 804], [538, 838], [1052, 838], [1052, 362], [1060, 362]], "35", { label: "下一班的告警量", lpos: [800, 832] });
edge([[287, 804], [287, 852], [1060, 852]], "24", { label: "考卷＋成績表", lpos: [700, 866] });

// --------------------------------------------------
// 繪製
// --------------------------------------------------
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const isAscii = (s) => [...s].every((c) => c.charCodeAt(0) < 128);

function stateOf(item, t) {
  if (FULL) return "old";
  if (item.since === t) return "new";
  if (item.changed.includes(t)) return "changed";
  return "old";
}

function visible(item, t) {
  if (vi(item.since) > vi(t)) return false;
  if (item.until && vi(t) > vi(item.until)) return false;
  return true;
}

function atVersion(variants, t) {
  let cur = variants[0][1];
  for (const [since, value] of variants) if (vi(since) <= vi(t)) cur = value;
  return cur;
}

function text(x, y, s, o = {}) {
  const size = o.size || 13, fill = o.fill || "#333", weight = o.weight || "normal", anchor = o.anchor || "middle";
  const mono = o.mono === undefined ? isAscii(s) : o.mono;
  return `<text x="${x}" y="${y}" font-family="${mono ? MONO : SANS}" font-size="${size}" fill="${fill}" ` +
    `font-weight="${weight}" text-anchor="${anchor}">${esc(s)}</text>`;
}

function lockIcon(x, y, color) {
  return `<g transform="translate(${x},${y})" fill="none" stroke="${color}" stroke-width="1.4">` +
    `<rect x="-5" y="-1" width="10" height="8" rx="1.5" fill="${color}" stroke="none"/>` +
    `<path d="M-3,-1 v-3 a3,3 0 0 1 6,0 v3"/></g>`;
}

function badge(x2, y, v, st) {
  const label = `v${v}`;
  const w = label.length * 6.4 + 10;
  const [bg, fg] = st === "new" ? ["#1f9d4c", "#fff"] : ["#e9ebee", "#5b616b"];
  return `<rect x="${x2 - w}" y="${y - 8}" width="${w}" height="15" rx="7.5" fill="${bg}"/>` +
    text(x2 - w / 2, y + 3.5, label, { size: 10, fill: fg, mono: true });
}

function render(t) {
  FULL = t === "full";
  const titleKey = t;
  if (FULL) t = "42";
  if (!VERS.includes(t)) throw new Error(`沒有這個版本：${t}`);
  const out = [`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">`, "<defs>"];
  for (const [name, p] of Object.entries(PALETTE)) {
    out.push(`<marker id="arr-${name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" ` +
      `orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${p.line}"/></marker>`);
  }
  out.push("</defs>");
  out.push(`<rect width="${W}" height="${H}" fill="#ffffff"/>`);

  // 標題和圖例
  const [title, chapter] = TITLES[titleKey];
  out.push(text(24, 40, FULL ? title : `v${t}　${title}`, { size: 22, fill: "#1f2328", weight: "bold", anchor: "start", mono: false }));
  out.push(text(24, 64, FULL ? `${chapter}　｜　箭頭 = 資料往哪裡走` : `這一節：${chapter}　｜　箭頭 = 資料往哪裡走　｜　小標籤 = 第幾節出現`,
    { size: 13, fill: "#6b7280", anchor: "start", mono: false }));
  let lx = 760;
  if (!FULL) {
  for (const [st, label] of [["old", "之前就有"], ["new", "這一節新增"], ["changed", "這一節改了"]]) {
    const p = PALETTE[st];
    out.push(`<rect x="${lx}" y="27" width="16" height="16" rx="3" fill="${p.fill}" stroke="${p.stroke}" stroke-width="1.5"/>`);
    out.push(text(lx + 22, 40, label, { size: 12, fill: "#444", anchor: "start", mono: false }));
    lx += 112;
  }
  }

  const badges = [];
  const order = { frame: 0, group: 1 };
  const shown = E.filter((e) => visible(e, t)).sort((a, b) => (order[a.kind] ?? 2) - (order[b.kind] ?? 2));
  for (const e of shown) {
    const st = stateOf(e, t);
    const p = PALETTE[e.kind === "error" ? "error" : st];
    const ls = atVersion(e.lines, t);
    const { x, y, w, h, kind: k } = e;
    const sw = st !== "old" ? 2 : 1.3;

    if (k === "frame") {
      const dash = e.dashed ? ' stroke-dasharray="7 5"' : "";
      const fill = e.dashed ? "none" : "#fbfbfc";
      const stroke = st !== "old" ? p.stroke : "#9aa0a8";
      out.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="16" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash}/>`);
      if (e.dashed) {
        out.push(text(x + 14, y + h - 7, ls[0], { size: 12, fill: p.text, anchor: "start", mono: false }));
        badges.push([x + 14 + ls[0].length * 12.5 + 46, y + h - 11, e.since, st]);
      } else {
        out.push(text(x + 16, y + 28, ls[0], { size: 16, fill: "#1f2328", weight: "bold", anchor: "start", mono: false }));
      }
      continue;
    }

    if (k === "group") {
      if (e.solid) {
        out.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10" fill="${p.fill}" stroke="${p.stroke}" stroke-width="${sw}"/>`);
        out.push(text(x + 12, y + 22, ls[0], { size: 13, fill: p.text, weight: "bold", anchor: "start" }));
      } else {
        out.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10" fill="none" stroke="${p.stroke}" stroke-width="${sw}" stroke-dasharray="5 4"/>`);
        out.push(text(x + 10, y + 14, ls[0], { size: 10.5, fill: p.text, anchor: "start" }));
      }
      if (e.badge) badges.push([x + w - 4, y, e.since, st]);
      continue;
    }

    if (k === "note") {
      const fill = st !== "old" ? PALETTE[st].text : "#6b7280";
      out.push(text(x, y, ls[0], { size: e.size || 12, fill, anchor: e.anchor || "middle", mono: false }));
      continue;
    }

    const rx = { port: h / 2, chip: h / 2, error: h / 2, actor: 12, external: 12 }[k] ?? 8;
    const fill = (["port", "actor", "external"].includes(k) && st === "old") ? "#ffffff" : p.fill;
    const dash = k === "external" ? ' stroke-dasharray="6 4"' : "";
    if (k === "store" && !e.labelTop) {
      out.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6" fill="${fill}" stroke="${p.stroke}" stroke-width="${sw}"/>` +
        `<path d="M${x},${y + 8} h${w}" stroke="${p.stroke}" stroke-width="1"/>`);
    } else {
      out.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${p.stroke}" stroke-width="${sw}"${dash}/>`);
    }

    let cx = x + w / 2;
    if (e.lock && visible({ since: e.lock }, t)) {
      const lk = e.lock === t ? "new" : "old";
      out.push(lockIcon(x + 15, y + h / 2 - 1, PALETTE[lk].line));
      cx += 7;
    }

    if (e.labelTop) {
      out.push(text(x + 10, y + 20, ls[0], { size: 13, fill: p.text, weight: "bold", anchor: "start" }));
    } else {
      const big = { actor: 15, external: 15, service: 13.5 }[k] ?? 10.5;
      const small = 10.5;
      if (ls.length === 1) {
        const size = k !== "chip" ? big : (e.id.startsWith("st_") ? 9.5 : 10.5);
        const bold = ["actor", "external", "service"].includes(k);
        out.push(text(cx, y + h / 2 + big * 0.36, ls[0], { size, fill: p.text, weight: bold ? "bold" : "normal" }));
      } else {
        const gap = ["actor", "external"].includes(k) ? 16 : 14;
        const top = y + h / 2 - (ls.length - 1) * gap / 2 + 4;
        ls.forEach((s, i) => {
          const first = i === 0 && ["actor", "external", "service", "store"].includes(k);
          out.push(text(cx, top + i * gap, s, { size: first ? big : small, fill: p.text, weight: first ? "bold" : "normal" }));
        });
      }
    }
    if (e.badge) badges.push([x + w + 4, y, e.since, st]);
  }

  for (const d of ED) {
    if (!visible(d, t)) continue;
    const st = d.error ? "error" : stateOf(d, t);
    const p = PALETTE[st];
    const pts = d.pts.map(([a, b]) => `${a},${b}`).join(" ");
    const dash = (d.dashed || d.error) ? ' stroke-dasharray="5 4"' : "";
    const marker = d.arrow ? ` marker-end="url(#arr-${st})"` : "";
    const sw = (st === "new" || st === "changed") ? 1.8 : 1.4;
    out.push(`<polyline points="${pts}" fill="none" stroke="${p.line}" stroke-width="${sw}"${dash}${marker}/>`);
    if (d.label) {
      const [lx2, ly2] = d.lpos;
      out.push(`<text x="${lx2}" y="${ly2}" font-family="${SANS}" font-size="10.5" fill="${p.text}" ` +
        `text-anchor="${d.lanchor}" stroke="#fff" stroke-width="4" paint-order="stroke">${esc(atVersion(d.label, t))}</text>`);
    }
  }

  for (const [bx, by, v, st] of badges) out.push(badge(bx, by, v, st));
  out.push("</svg>");
  return out.join("\n") + "\n";
}

if (typeof process !== "undefined" && process.argv && process.argv[1] && process.argv[1].endsWith("gen.js")) {
  const arg = process.argv[2] || "42";
  if (arg === "all") {
    const fs = require("fs");
    const path = require("path");
    for (const v of VERS.slice(1)) fs.writeFileSync(path.join(__dirname, `v${v}.svg`), render(v));
    fs.writeFileSync(path.join(__dirname, "full.svg"), render("full"));
  } else {
    process.stdout.write(render(arg));
  }
}
