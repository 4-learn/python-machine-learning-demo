# 18｜YOLO：物件偵測推論

用 ultralytics 載入教師訓練的 `models/helmet-yolo11n.pt` 與 COCO 預訓練的 `models/yolo11n.pt`，對合成場景與 3 張擬真照片推論，拆解 `Results` 物件（`boxes.xyxy`／`conf`／`cls`）。

    YOLO_OFFLINE=1 python 18-yolo/predict.py

ultralytics 與 YOLO11 權重為 AGPL-3.0，見講義第 2 小節。
