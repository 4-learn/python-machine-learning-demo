"""03｜scikit-learn 切資料：train_test_split、GroupShuffleSplit，以及為什麼最後用時間切。"""
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("story/data/factory/alerts.csv")
df["occurred_at"] = pd.to_datetime(df["occurred_at"])
NUMERIC = ["confidence", "duration_seconds", "light_lux", "days_on_site", "prior_alerts_30d", "hour"]
X, y = df[NUMERIC], df["label"]


def caught(train, test):
    model = KNeighborsClassifier(n_neighbors=1).fit(train[NUMERIC], train["label"])
    pred = model.predict(test[NUMERIC])
    return pred[test["label"] == 1].mean()


print("== 1. train_test_split：隨機抽 20% 當考卷 ==")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(f"train {len(X_train)} 筆、test {len(X_test)} 筆")
print(f"要叫人的比例：全部 {y.mean():.3f}、train {y_train.mean():.3f}、test {y_test.mean():.3f}")

print("\n== 2. stratify=y：兩邊的比例一樣 ==")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
print(f"要叫人的比例：train {y_train.mean():.3f}、test {y_test.mean():.3f}")

print("\n== 3. 同一件事被切到兩邊 ==")
train_ids = set(df.loc[X_train.index, "incident_id"])
test_ids = set(df.loc[X_test.index, "incident_id"])
print(f"test 有 {len(test_ids)} 件事，其中 {len(train_ids & test_ids)} 件在 train 也看得到")
cols = ["alert_id", "incident_id", "confidence", "duration_seconds", "light_lux", "label", "side"]
df["side"] = "train"
df.loc[X_test.index, "side"] = "test"
print(df.loc[df["incident_id"] == "I00142", cols].to_string(index=False))

print("\n== 4. GroupShuffleSplit：同一個 incident_id 只會在一邊 ==")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_idx, test_idx = next(gss.split(X, y, groups=df["incident_id"]))
g_train, g_test = df.iloc[train_idx], df.iloc[test_idx]
shared = set(g_train["incident_id"]) & set(g_test["incident_id"])
print(f"train {len(g_train)} 筆、test {len(g_test)} 筆；兩邊都有的事件：{len(shared)} 件")

print("\n== 5. 同一個 1-NN，換一種切法，分數差多少 ==")
fit = df[df["occurred_at"] < "2025-12-15"]
valid = df[(df["occurred_at"] >= "2025-12-15") & (df["occurred_at"] < "2025-12-29")]
rows = [("隨機切（stratify）", caught(df.loc[X_train.index], df.loc[X_test.index])),
        ("依 incident 分組切", caught(g_train, g_test)),
        ("依時間切（前 6 週→第 7–8 週）", caught(fit, valid))]
print(pd.DataFrame(rows, columns=["切法", "真事件抓到幾成"]).round(3).to_string(index=False))
