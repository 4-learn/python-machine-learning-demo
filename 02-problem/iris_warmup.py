"""02｜暖身：sklearn 內建的 iris 資料集長什麼樣子——X、y、一列是一朵花。"""
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target

print("== 0. iris：一列＝一朵花 ==")
print(f"X 的形狀 {X.shape}；y 的形狀 {y.shape}")
print(X.head(3).to_string())
counts = y.value_counts().sort_index()
print("y 的類別：", ", ".join(f"{name} {n} 朵" for name, n in zip(iris.target_names, counts)))
