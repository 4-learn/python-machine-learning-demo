"""把 master_extra.py 切成 v25–v41 各版的 extra.py。"""
import re, sys
from pathlib import Path
src = Path(sys.argv[1]).read_text().splitlines()
root = Path(sys.argv[2])
names = {25: "v25-opencv", 26: "v26-roi", 27: "v27-ezcam", 28: "v28-conv2d", 29: "v29-yolo-pose", 30: "v30-pose-features",
         31: "v31-resample", 32: "v32-rolling", 33: "v33-merge-asof", 34: "v34-drift", 35: "v35-regression",
         36: "v36-timeseries", 37: "v37-learning-curve", 38: "v38-random-search", 39: "v39-class-weight", 40: "v40-pca"}
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
    text = re.sub(r"\n{3,}(?=def |class |# |[A-Z_]+ = )", "\n\n\n", text)
    d = root / name; d.mkdir(parents=True, exist_ok=True); (d / "extra.py").write_text(text)
print("ok")
