"""03｜三種「分數好得可疑」：洩漏欄、同一 incident 跨兩邊、用到未來的資料。

模型先當黑盒子：邏輯迴歸 07 節才講、KNN 05 節才講。這裡只看「切法」怎麼改變分數。
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (CATEGORICAL, FEATURES, NUMERIC, TARGET, fit_valid_split, load_events,
                    make_preprocessor, score_row, time_split)


def logistic(extra=()):
    """數值欄縮放＋類別欄 one-hot＋邏輯迴歸（06、07 節會拆開講）。extra＝額外加進來的數值欄。"""
    pre = ColumnTransformer([("num", StandardScaler(), NUMERIC + list(extra)),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    return make_pipeline(pre, LogisticRegression(max_iter=1000))


def nearest():
    """1-NN：找訓練資料裡最像的一筆，照抄它的答案。"""
    return make_pipeline(make_preprocessor(), KNeighborsClassifier(n_neighbors=1))


def evaluate(name, model, train, test, cols=FEATURES):
    model.fit(train[cols], train[TARGET])
    return score_row(name, test[TARGET], model.predict(test[cols]))


events = load_events()
events["has_ticket"] = events.ticket_id.notna().astype(int)

print("== 1. 洩漏欄：把「有沒有工單」當特徵（隨機切割 75%／25%）==")
train, test = train_test_split(events, test_size=0.25, random_state=0, stratify=events[TARGET])
rows = [evaluate("邏輯迴歸 10 個特徵", logistic(), train, test),
        evaluate("＋has_ticket", logistic(["has_ticket"]), train, test, FEATURES + ["has_ticket"])]
leaky = logistic(["has_ticket"]).fit(train[FEATURES + ["has_ticket"]], train[TARGET])
live = test.assign(has_ticket=0)                    # 上線時：事件剛發生，每一筆都還沒有工單
rows.append(score_row("＋has_ticket 上線時", test[TARGET], leaky.predict(live[FEATURES + ["has_ticket"]])))
print(pd.DataFrame(rows).to_string(index=False, float_format="%.3f"))

print("\n== 2. 同一起 incident 跨到兩邊（1-NN）==")
knn = nearest()
rows = [evaluate("隨機切割", knn, train, test)]
share = len(set(train.incident_id) & set(test.incident_id))
print(f"隨機切割：test 裡的 incident 有 {share} 起也出現在 train")
pre, nn = knn[:-1], knn[-1]
neighbor = nn.kneighbors(pre.transform(test[FEATURES]), return_distance=False)[:, 0]
twin = (train.incident_id.values[neighbor] == test.incident_id.values).mean()
print(f"test 每一筆找到的「最像的一筆」，{twin:.3f} 來自同一起 incident（等於抄自己的分身）")
splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
tr_idx, te_idx = next(splitter.split(events, events[TARGET], groups=events.incident_id))
g_train, g_test = events.iloc[tr_idx], events.iloc[te_idx]
share = len(set(g_train.incident_id) & set(g_test.incident_id))
print(f"依 incident 分組切割：test {len(g_test)} 筆，跨兩邊的 incident {share} 起")
rows.append(evaluate("依 incident 分組", nearest(), g_train, g_test))
print(pd.DataFrame(rows).to_string(index=False, float_format="%.3f"))

print("\n== 3. 用到未來：「當日違規數」是下班後才結算的 ==")
# 從這裡開始，最後 2 週（test）封存；在前 8 週裡再用時間切：前 6 週訓練、第 7–8 週驗證
events = events.sort_values("event_time", kind="stable")
day = events.event_time.dt.date
by_day = events.groupby(["worker_id", day])[TARGET]
events["day_violations"] = by_day.transform("sum")                  # 日報：整天結算（含之後的事件）
events["day_so_far"] = by_day.cumsum() - events[TARGET]            # 到這筆事件之前為止
fit, valid = fit_valid_split(time_split(events)[0])
rows = [evaluate("邏輯迴歸 10 個特徵", logistic(), fit, valid),
        evaluate("＋當日違規數（日報）", logistic(["day_violations"]), fit, valid, FEATURES + ["day_violations"]),
        evaluate("＋當下為止的違規數", logistic(["day_so_far"]), fit, valid, FEATURES + ["day_so_far"])]
print(pd.DataFrame(rows).to_string(index=False, float_format="%.3f"))
