"""把 master.py 依標記切成 v01–v14 各版的 judge.py。"""
import re, sys
from pathlib import Path
src = Path(sys.argv[1]).read_text().splitlines()
names = {1:"v01-platform-rule",2:"v02-features",3:"v03-split",4:"v04-baseline",5:"v05-knn",6:"v06-pipeline",7:"v07-logistic",
         8:"v08-trees",9:"v09-cv",10:"v10-tuning",11:"v11-metrics",12:"v12-thresholds",13:"v13-errors",14:"v14-unsupervised",15:"v15-workshop1"}
out_root = Path(sys.argv[2])
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
    text = re.sub(r"\n{3,}(?=def |# |CV|GRID|LINE|NUMERIC|VALID)", "\n\n\n", text)
    d = out_root / name; d.mkdir(parents=True, exist_ok=True); (d / "judge.py").write_text(text)
print("ok")
