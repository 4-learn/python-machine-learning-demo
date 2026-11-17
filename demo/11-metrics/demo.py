"""11｜scikit-learn 的 confusion_matrix 與 classification_report：判斷器錯在哪一格。"""
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score

# 12 筆告警：label 是事後的結果（1＝真的要叫人），pred 是判斷器的決定（1＝叫人）
alerts = pd.DataFrame({
    "alert_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "location": ["aisle_01", "aisle_01", "dock_01", "rest_01", "rest_01", "aisle_02",
                 "aisle_02", "dock_01", "aisle_01", "rest_01", "aisle_02", "dock_01"],
    "label": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "pred":  [1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
})
y_true, y_pred = alerts["label"], alerts["pred"]

print("== 1. confusion_matrix(y_true, y_pred) ==")
cm = confusion_matrix(y_true, y_pred)
print(cm)
print(pd.DataFrame(cm, index=["事後：不用叫", "事後：要叫"], columns=["判斷：沒叫", "判斷：叫了"]).to_string())

print("\n== 2. ravel()：拆成 tn fp fn tp ==")
tn, fp, fn, tp = cm.ravel()
tn2, fp2, fn2, tp2 = confusion_matrix(y_pred, y_true).ravel()   # 參數放反
print(f"tn={tn}  fp={fp}（誤報）  fn={fn}（漏報）  tp={tp}")

print("\n== 3. precision 和 recall 從哪兩格來 ==")
print(f"precision = tp / (tp + fp) = {tp}/{tp + fp} = {tp / (tp + fp):.3f}   叫的裡面幾成是真的")
print(f"recall    = tp / (tp + fn) = {tp}/{tp + fn} = {tp / (tp + fn):.3f}   真的裡面叫到幾成")
print(f"precision_score {precision_score(y_true, y_pred):.3f}、recall_score {recall_score(y_true, y_pred):.3f}")

print("\n== 4. 參數放反：fp 和 fn 對調 ==")
print(f"confusion_matrix(y_pred, y_true) 拆出來：fp={fp2}  fn={fn2}")

print("\n== 5. classification_report：兩個類別一次看 ==")
print(classification_report(y_true, y_pred, target_names=["不用叫", "要叫人"], digits=3))

print("== 6. 錯誤地圖：錯的是哪幾筆、在哪裡 ==")
alerts["結果"] = "對"
alerts.loc[(y_true == 0) & (y_pred == 1), "結果"] = "誤報"
alerts.loc[(y_true == 1) & (y_pred == 0), "結果"] = "漏報"
print(alerts[alerts["結果"] != "對"].to_string(index=False))
