"""把 master_detector.py 切成 v16–v24 各版的 detector.py；v23、v24 另外附上 v15 的 judge.py。"""
import re, shutil, sys
from pathlib import Path
src = Path(sys.argv[1]).read_text().splitlines()
root = Path(sys.argv[2])
names = {16: "v16-image-array", 17: "v17-pytorch", 18: "v18-backbone-features", 19: "v19-linear-probe", 20: "v20-yolo",
         21: "v21-iou", 22: "v22-requests", 23: "v23-workshop2", 24: "v24-handoff"}
for v, name in names.items():
    keep, mode = [], ("all", None)
    for line in src:
        m = re.match(r"#@(all|v(\d\d)|main((?: v\d\d)+))$", line.strip())
        if m:
            if m.group(1) == "all": mode = ("all", None)
            elif m.group(2): mode = ("ver", int(m.group(2)))
            else: mode = ("main", [int(x[1:]) for x in m.group(3).split()])
            continue
        if mode[0] == "all" or (mode[0] == "ver" and v >= mode[1]) or (mode[0] == "main" and v in mode[1]):
            keep.append(line)
    text = "\n".join(keep).replace("{VERSION}", name.split("-")[0]) + "\n"
    text = re.sub(r"\n{3,}(?=def |# )", "\n\n\n", text)
    d = root / name; d.mkdir(parents=True, exist_ok=True); (d / "detector.py").write_text(text)
    if v >= 23:
        shutil.copy(root / "v15-workshop1" / "judge.py", d / "judge.py")
print("ok")
