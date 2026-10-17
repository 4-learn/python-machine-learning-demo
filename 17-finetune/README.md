# 17｜微調預訓練模型：安全帽分類

凍結 MobileNetV3-Small 骨幹、只訓練最後一層，分辨 `data/crops` 的頭部小圖有沒有戴安全帽；依攝影機分組切割（cam5 夜間留作測試）。`crops.py` 是本節共用的小工具。

    python 17-finetune/finetune.py
