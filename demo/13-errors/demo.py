"""13｜scikit-learn 的 permutation_importance：模型看什麼，以及把錯誤一筆筆拉出來看。"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

# 合成 200 件事、每件 1–3 筆告警。規則：停留夠久、而且不在休息區，才是真的。
rng = np.random.default_rng(0)
rows = []
for i in range(200):
    location = rng.choice(["aisle_01", "aisle_02", "rest_01"], p=[0.4, 0.4, 0.2])
    duration = rng.uniform(2, 60)
    label = int(duration > 25 and location != "rest_01")
    if rng.random() < 0.06:          # 少數事件，值班的人處理得和規則不一樣
        label = 1 - label
    for k in range(rng.integers(1, 4)):
        rows.append({"incident_id": f"I{i:03d}", "location": location,
                     "duration_seconds": round(duration + rng.normal(0, 1), 1),
                     "confidence": round(float(np.clip(0.4 + 0.3 * label + rng.normal(0, 0.15), 0.05, 0.99)), 3),
                     "light_lux": int(rng.uniform(50, 900)), "label": label})
alerts = pd.DataFrame(rows)
FEATURES = ["duration_seconds", "confidence", "light_lux", "location"]
first = alerts["incident_id"] < "I140"
train, valid = alerts[first], alerts[~first]
model = make_pipeline(
    ColumnTransformer([("num", "passthrough", FEATURES[:3]),
                       ("cat", OneHotEncoder(handle_unknown="ignore"), ["location"])]),
    RandomForestClassifier(n_estimators=200, random_state=0)).fit(train[FEATURES], train["label"])
X_valid, y_valid = valid[FEATURES], valid["label"]

print("== 1. 打亂一欄，分數掉多少（手動做一次）==")
base = average_precision_score(y_valid, model.predict_proba(X_valid)[:, 1])
print(f"原本的 average precision：{base:.3f}")
for col in ["duration_seconds", "light_lux"]:
    shuffled = X_valid.copy()
    shuffled[col] = rng.permutation(shuffled[col].to_numpy())
    after = average_precision_score(y_valid, model.predict_proba(shuffled)[:, 1])
    print(f"打亂 {col:<17}→ {after:.3f}（掉 {base - after:.3f}）")

print("\n== 2. permutation_importance：每欄打亂 10 次取平均 ==")
imp = permutation_importance(model, X_valid, y_valid, scoring="average_precision",
                             n_repeats=10, random_state=0)
table = pd.DataFrame({"平均掉多少": imp.importances_mean, "標準差": imp.importances_std}, index=FEATURES)
print(table.sort_values("平均掉多少", ascending=False).round(3).to_string())

print("\n== 3. 錯誤清單：漏報和誤報一筆筆列出來 ==")
valid = valid.assign(pred=model.predict(X_valid))
valid["錯誤"] = np.select([(valid["label"] == 1) & (valid["pred"] == 0),
                          (valid["label"] == 0) & (valid["pred"] == 1)], ["漏報", "誤報"], "")
errors = valid[valid["錯誤"] != ""]
print(errors[["錯誤", "incident_id", "location", "duration_seconds", "confidence", "label"]].to_string(index=False))

print("\n== 4. 找規律：錯誤是不是集中在同幾件事、同一個地點 ==")
print(f"{len(errors)} 筆錯誤，來自 {errors['incident_id'].nunique()} 件事")
print(errors.groupby("incident_id").size().sort_values(ascending=False).head(3).to_string())
print(pd.crosstab(valid["location"], valid["錯誤"].replace("", "對")).to_string())
