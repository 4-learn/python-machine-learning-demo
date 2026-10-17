"""10｜整門課第一次打開 test（後 2 週）。參數已經在 grid_search.py 用 train 決定好，這裡不再改。"""
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from common import current_rule, fit_valid_split, load_events, make_preprocessor, score_row, time_split, xy

threadpool_limits(2)

BEST = {"learning_rate": 0.05, "max_depth": 3, "max_leaf_nodes": 15}  # 抄自 grid_search.py 的輸出，不在這裡改

train, test = time_split(load_events())
fit, valid = fit_valid_split(train)
X_train, y_train = xy(train)


def best_model():
    return make_pipeline(make_preprocessor(scale=False), HistGradientBoostingClassifier(random_state=0, **BEST))


print("== 1. 打開之前先寫下預期 ==")
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
cvres = cross_validate(best_model(), X_train, y_train, groups=train["incident_id"], cv=cv,
                       scoring=["accuracy", "precision", "recall"])
for m in ["accuracy", "precision", "recall"]:
    print(f"CV {m}：{cvres[f'test_{m}'].mean():.3f} ± {cvres[f'test_{m}'].std():.3f}")

# 候選模型：前 6 週訓練。第 7–8 週（valid）要留給 11–13 節做錯誤分析、選門檻，模型不能先看過它。
model = best_model().fit(*xy(fit))
X_valid, y_valid = xy(valid)
print("前 6 週訓練的候選模型，在 valid 上：")
print(pd.DataFrame([score_row("梯度提升（調參後）", y_valid, model.predict(X_valid))]).to_string(index=False, float_format="%.3f"))

print("\n== 2. test 成績（只算這一次）==")
X_test, y_test = xy(test)
rows = [score_row("多數類", y_test, DummyClassifier().fit(*xy(fit)).predict(X_test)),
        score_row("現行規則 conf>=0.7", y_test, current_rule(test)),
        score_row("梯度提升（調參後）", y_test, model.predict(X_test))]
print(f"test {len(test)} 筆（2025-12-29～2026-01-11），該通知 {y_test.mean():.3f}")
print(pd.DataFrame(rows).to_string(index=False, float_format="%.3f"))
