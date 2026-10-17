# 12｜門檻怎麼選：0.7 從哪裡來？

用有標籤的驗證集與「漏報＝10 倍誤報」的成本選門檻；PR 曲線、`TunedThresholdClassifierCV`（含 metadata routing），以及 KMeans 切點的反例。設定檔只印出範例，不覆寫 `config/`。

    python 12-threshold/threshold.py
    python 12-threshold/tuned.py
    python 12-threshold/kmeans_cut.py
