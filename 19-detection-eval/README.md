# 19｜偵測結果的評估與事件化

自己寫 IoU，數 head（沒戴）類的 TP／FP／FN，在 val 上用 12 節的成本法（10×FN＋1×FP）選偵測門檻，test 只開一次；對照 `model.val()` 的 mAP；最後把偵測框轉成全課共用的 Event JSON（`events.json`，21 節 Mock 模式讀它）。

    python 19-detection-eval/iou.py
    python 19-detection-eval/evaluate.py        # CPU 約 1 分鐘
    python 19-detection-eval/official_val.py
    python 19-detection-eval/to_events.py

`detlib.py` 是本節的小工具（IoU、讀標註、配對、夜間模擬），可被 import。
