"""三種事件送進平台：ROI（OpenCV）、骨架（YOLO-pose 特徵）、偵測（YOLO）（ML 課進度存檔 v42，Workshop 3）。

執行（在這個資料夾裡）：
    LINE=factory python workshop3.py     # aisle_blocked、彎腰搬重物、no_helmet
    LINE=care python workshop3.py        # bed_exit、fall_detected（骨架）、fall_detected（偵測）
"""
import json

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import detector
import extra

LINE = extra.LINE
POSE_EVENT = {"factory": "unsafe_lifting", "care": "fall_detected"}[LINE]


def roi_events():
    """一天的固定攝影機畫面 → ROI 事件（aisle_blocked／bed_exit）。
    回傳 (事件, 事後答案)：答案只留在本機對答案用，不能放進送出去的事件。"""
    index, roi = extra.load_roi()
    ref = extra.make_reference(index, roi)
    events, answers = [], []
    for _, row in index.iterrows():
        score = extra.roi_score(extra.read_frame(row["file"]), roi, ref)
        if extra.decide(score, roi):
            events.append({"event_type": roi["event_type"], "location": roi["name"], "source": "ezcam_01",
                           "confidence": round(min(1.0, 0.5 + abs(score - extra.PROFILE["roi_score_min"]) * 5), 3),
                           "occurred_at": row["taken_at"].replace(" ", "T"),
                           "detail": {"roi_score": round(score, 3), "frame": row["file"]}})
            answers.append(int(row["label"]))
    return events, answers


def pose_events():
    """骨架段落：用沒看過的人當考卷，模型判斷「要注意」的段落變成事件。只送分數與特徵，不送畫面。
    回傳 (事件, 事後答案)。"""
    table = extra.sequence_table(extra.load_poses())
    train, test = extra.split_by_subject(table)
    model = HistGradientBoostingClassifier(class_weight={0: 1, 1: 10}, random_state=0)
    model.fit(train[extra.POSE_FEATURES], train["alert"])
    proba = model.predict_proba(test[extra.POSE_FEATURES])[:, 1]
    events, answers = [], []
    for (seq_id, row), p in zip(test.iterrows(), proba):
        if p >= 0.5:
            events.append({"event_type": POSE_EVENT, "location": extra.PROFILE["locations"][0], "source": "pose_01",
                           "confidence": round(float(p), 3), "occurred_at": "2026-01-19T10:00:00",
                           "detail": {"seq_id": seq_id, "torso_angle": float(row["torso_angle"]),
                                      "hip_drop": float(row["hip_drop"])}})
            answers.append(int(row["alert"]))
    return events, answers


def main():
    profile = detector.PROFILE
    if POSE_EVENT not in profile["event_types"]:
        # 平台只收設定檔裡有的事件類型（平台的規則）。新事件要先加進設定檔，平台才收。
        profile["event_types"][POSE_EVENT] = {"severity": "high", "escalate_after_minutes": 10}
        print(f"設定檔加上新的事件類型：{POSE_EVENT}")
    url = detector.platform_url()
    sources = {"ROI（OpenCV）": roi_events(), "骨架（YOLO-pose 特徵）": pose_events()}
    print(f"== {LINE}：三種偵測器，同一個平台 ==")
    rows = []
    for name, (events, answers) in sources.items():
        codes = []
        for event in events:
            code, body = detector.send(url, event)
            codes.append(code)
        rows.append({"來源": name, "事件": len(events), "HTTP 200": codes.count(200), "HTTP 400": codes.count(400),
                     "事後確認是真的": sum(answers)})
    images = sorted((detector.SCENES / "images" / "test").glob("*.jpg"))
    det = detector.to_events(detector.detect(detector.load_detector(), images), profile["detector_conf"])
    codes = [detector.send(url, e)[0] for e in det]
    rows.append({"來源": "物件偵測（YOLO）", "事件": len(det), "HTTP 200": codes.count(200), "HTTP 400": codes.count(400),
                 "事後確認是真的": None})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n== 平台上的告警（依事件類型）==")
    import requests
    alerts = pd.DataFrame(requests.get(f"{url}/alerts", timeout=5).json())
    print(alerts.groupby(["event_type", "status"]).size().to_string())


if __name__ == "__main__":
    main()
