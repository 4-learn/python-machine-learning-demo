"""06 加練：多考幾次。每次往後滑一週，只用前面的資料學、考接下來那一週，一共考 5 次。

執行（在這個資料夾裡）：
    python rolling.py
"""
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

from judge import candidate_models, load_alerts, make_preprocessor, score_row, xy

# 5 次考試，每次考卷的第一天；考卷是那一天起的 7 天
STARTS = ["2025-11-24", "2025-12-01", "2025-12-08", "2025-12-15", "2025-12-22"]


def three_models():
    trees = candidate_models()
    return {"KNN": make_pipeline(make_preprocessor(), KNeighborsClassifier(n_neighbors=15)),
            "隨機森林": trees["隨機森林"],
            "梯度提升": trees["梯度提升"]}


def main():
    df = load_alerts()
    rows = []
    for start in STARTS:
        start = pd.Timestamp(start)
        end = start + pd.Timedelta(days=7)
        learn = df[df["occurred_at"] < start]
        exam = df[(df["occurred_at"] >= start) & (df["occurred_at"] < end)]
        print(f"第 {len(rows) // 3 + 1} 次：學 {len(learn)} 筆（到 {start:%m-%d} 前），考 {len(exam)} 筆（{start:%m-%d}～{end - pd.Timedelta(days=1):%m-%d}）")
        for name, model in three_models().items():
            model.fit(*xy(learn))
            row = score_row(name, xy(exam)[1], model.predict(xy(exam)[0]))
            row["考試"] = start.strftime("%m-%d")
            rows.append(row)

    table = pd.DataFrame(rows)
    print("\n== 每一次的分數 ==")
    print(table.pivot(index="考試", columns="model", values=["precision", "recall"]).to_string())
    print("\n== 5 次平均 ==")
    print(table.groupby("model")[["precision", "recall"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
