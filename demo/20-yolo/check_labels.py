"""10｜對答案：Label Studio 匯出的 YOLO 標註，和標準答案比，並整理成訓練用的資料夾。

執行（在 repo 根目錄，課程環境 .venv）：
    python demo/20-yolo/check_labels.py story/my/labels-export

做三件事：
  1. 讀 classes.txt，把類別編號換回課程的順序（0 helmet、1 head）
  2. 每張照片的框和標準答案配對，算 IoU（11 學的），列出漏畫、多畫、畫歪的框
  3. 寫出 story/my/labels-mine/：images、labels、data.yaml，給 train_mine.py 訓練
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANSWER = ROOT / "story" / "data" / "factory" / "scenes" / "labels" / "train"
OUT = ROOT / "story" / "my" / "labels-mine"
NAMES = ["helmet", "head"]          # 課程的順序：0 沒事、1 要注意


def read_boxes(path):
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    return [(int(r[0]), *map(float, r[1:5])) for r in rows]


def iou(a, b):
    """a、b：(cx, cy, w, h)，都是 0–1 的比例。"""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    w, h = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = w * h
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def main(export):
    export = Path(export)
    ls_names = (export / "classes.txt").read_text().split()
    print(f"== Label Studio 的類別順序：{dict(enumerate(ls_names))}")
    remap = {i: NAMES.index(n) for i, n in enumerate(ls_names)}
    print(f"   換回課程的順序：{dict(enumerate(NAMES))}")

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)
    (OUT / "labels").mkdir(parents=True)

    total = ok = 0
    print("\n== 和標準答案比（IoU ≥ 0.5 而且類別對，才算畫對）==")
    for lab in sorted((export / "labels").glob("*.txt"), key=lambda p: p.stem.split("-", 1)[-1]):
        stem = lab.stem.split("-", 1)[-1]          # 去掉 Label Studio 加的亂碼前綴
        mine = [(remap[c], cx, cy, w, h) for c, cx, cy, w, h in read_boxes(lab)]
        answer = read_boxes(ANSWER / f"{stem}.txt")
        notes, used = [], set()
        for c, *box in answer:
            best = max(range(len(mine)), key=lambda j: iou(box, mine[j][1:]) if j not in used else -1, default=None)
            score = iou(box, mine[best][1:]) if best is not None and best not in used else 0
            if score >= 0.5 and mine[best][0] == c:
                ok += 1
                used.add(best)
                if score < 0.8:
                    notes.append(f"{NAMES[c]} 畫歪了（IoU {score:.2f}）")
            elif score >= 0.5:
                used.add(best)
                notes.append(f"類別選錯：應該是 {NAMES[c]}，畫成 {NAMES[mine[best][0]]}")
            else:
                notes.append(f"漏畫一個 {NAMES[c]}")
        notes += [f"多畫一個 {NAMES[mine[j][0]]}" for j in range(len(mine)) if j not in used]
        total += len(answer)
        print(f"{stem}：答案 {len(answer)} 個、你畫 {len(mine)} 個  " + ("全對" if not notes else "；".join(notes)))

        image = next((export / "images").glob(f"{lab.stem}.*"))
        shutil.copy(image, OUT / "images" / f"{stem}{image.suffix}")
        (OUT / "labels" / f"{stem}.txt").write_text(
            "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in mine))

    print(f"\n畫對 {ok}／{total} 個框")
    (OUT / "data.yaml").write_text(f"path: {OUT}\ntrain: images\nval: images\nnames:\n"
                                   + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)), encoding="utf-8")
    print(f"已整理到 {OUT.relative_to(ROOT)}（images、labels、data.yaml）")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("用法：python demo/20-yolo/check_labels.py 匯出解壓的資料夾")
    main(sys.argv[1])
