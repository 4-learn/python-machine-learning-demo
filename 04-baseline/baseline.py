"""04｜不用模型能拿幾分：多數類、現行規則。成績表的前兩列。"""
import pandas as pd
from sklearn.dummy import DummyClassifier

from common import TARGET, current_rule, fit_valid_split, load_events, score_row, time_split, xy

train, test = time_split(load_events())
fit, valid = fit_valid_split(train)

print("== 1. 三份資料：fit 學、valid 比、test 封存 ==")
for name, part in [("fit", fit), ("valid", valid)]:
    print(f"{name:5s} {len(part):4d} 筆，{part.event_time.min():%m-%d}～{part.event_time.max():%m-%d}，"
          f"該通知 {part[TARGET].sum():3d} 筆（{part[TARGET].mean():.3f}）")
print(f"test  {len(test):4d} 筆，{test.event_time.min():%m-%d}～{test.event_time.max():%m-%d}，封存到 10 節")

X_fit, y_fit = xy(fit)
X_valid, y_valid = xy(valid)

print("\n== 2. 多數類：永遠猜最常見的答案 ==")
dummy = DummyClassifier(strategy="most_frequent").fit(X_fit, y_fit)
pred = dummy.predict(X_valid)
print(f"fit 裡最常見的答案：{dummy.classes_[dummy.class_prior_.argmax()]}（比例 {dummy.class_prior_.max():.3f}）")
print(f"對 valid {len(pred)} 筆的預測：", pd.Series(pred).value_counts().to_dict())
print(f"答對 {(pred == y_valid).sum()} 筆，accuracy {(pred == y_valid).mean():.3f}")
print(f"該通知的 {y_valid.sum()} 筆，抓到 {((pred == 1) & (y_valid == 1)).sum()} 筆")

print("\n== 3. 現行規則：confidence >= 0.7 ==")
rule = current_rule(valid)
print(pd.crosstab(y_valid.map({1: "該通知", 0: "不必通知"}),
                  rule.map({1: "規則通知", 0: "規則不通知"}).rename("rule")).to_string())

print("\n== 4. 成績表（valid，第 7–8 週）==")
board = pd.DataFrame([score_row("多數類", y_valid, pred),
                      score_row("現行規則 conf>=0.7", y_valid, rule)])
print(board.to_string(index=False, float_format="%.3f"))
days = valid.event_time.dt.date.nunique()
print(f"valid 共 {days} 天：規則平均每天通知 {rule.sum() / days:.1f} 次，其中該通知的 {y_valid.sum() / days:.1f} 次")

print("\n== 5. 反過來：全部通知 ==")
everything = pd.Series(1, index=valid.index)
print(pd.DataFrame([score_row("全部通知", y_valid, everything)]).to_string(index=False, float_format="%.3f"))
