# 10｜過擬合與調參

決策樹深度的驗證曲線、用 GridSearchCV 在 train 上調梯度提升，最後只打開 test 一次。

    python 10-tuning/depth_curve.py   # 約 10 秒
    python 10-tuning/grid_search.py   # 90 次訓練，單執行緒，記憶體約 200 MB
    python 10-tuning/test_once.py     # 整門課第一次看 test
