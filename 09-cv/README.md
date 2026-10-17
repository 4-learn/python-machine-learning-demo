# 09｜交叉驗證：一次切割的分數不可信

只換切割的 random_state 分數就會變；改用 StratifiedGroupKFold（依 incident 分組）做 5 折，成績表寫成平均 ± 標準差。

    python 09-cv/seed_spread.py   # 換 10 次 random_state
    python 09-cv/cv_table.py      # 5 折交叉驗證版成績表
