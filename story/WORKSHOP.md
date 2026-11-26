# Workshop 參考答案

每一份 Workshop 分兩段：

- **Workshop-1（必交）**：進到該份的資料夾，直接執行解答，輸出截圖上傳。跑得出來就給分。
- **Workshop-2（加分題）**：把解答的某一個小地方改掉，再跑一次，用自己的話解釋結果怎麼變、為什麼。

改之前先複製一份到 `story/my/`，原本的檔案不要動：

```bash
mkdir -p story/my
cp story/v03-split/judge.py story/my/judge.py   # 換成該份的資料夾
cd story/my
python judge.py
```

| 份 | Workshop-1 執行 |
|---|---|
| 02 | `story/v02-features/judge.py` |
| 03 | `story/v03-split/judge.py` |
| 04 | `story/v04-baseline/judge.py` |
| 05 | `story/v05-knn/judge.py` |
| 06 | `story/v08-trees/judge.py` |
| 07 | `story/v11-metrics/judge.py` |
| 08 | `story/v12-thresholds/judge.py` |
| 09 | `story/v15-workshop1/judge.py` |

以下是 Workshop-2 每一題改哪裡、實際跑出來的結果，以及可以怎麼解釋。

---

## 02｜接手時間也會偷看答案嗎？

改：最後一段的 `resolved_at` 改成 `acked_at`（有沒有人接手）。

```python
print(pd.crosstab(df["acked_at"].notna().map({True: "有接手時間", False: "沒有"}), y).to_string())
```

結果：

```
label        0    1
acked_at
有接手時間     1154  545
沒有        1841    0
```

解釋：沒有接手的 1841 筆全是誤報，所以 `acked_at` 也洩漏答案，不能當特徵。但有接手的裡面，1154 筆還是誤報（看一眼就駁回），不像結案時間那樣 100% 對上。事後才知道的欄位，都不能用。

## 03｜考卷切早一週

改：`VALID_START` 從 `2025-12-15` 改成 `2025-12-08`。

結果：

```
fit    1840 筆  11-03～12-07  要叫人 0.154
valid   990 筆  12-08～12-28  要叫人 0.140
test    710 筆  12-29～01-11  要叫人 0.172
同一件事被切到兩邊：0 件
```

解釋：fit 少了一週（2163 → 1840），valid 多了一週（667 → 990）。test 沒變，因為 `TEST_START` 沒動。照時間切，怎麼挪都不會把同一件事拆到兩邊。

## 04｜平台規則的 0.8 改掉

改：`platform_rule` 的 `bins=[0, 0.5, 0.8, 1.01]`，把 0.8 改成 0.7 或 0.9。

結果：

| 門檻 | precision | recall | 叫了幾次 |
|---|---:|---:|---:|
| 0.7 | 0.394 | 0.656 | 160 |
| 0.8（原本） | 0.629 | 0.458 | 70 |
| 0.9 | 0.842 | 0.167 | 19 |

解釋：門檻調低，叫得多，抓到的真事件變多（recall 升），但白跑也多（precision 降）；調高剛好相反。兩個分數像蹺蹺板。

## 05｜問幾個鄰居

改：`model_rows` 裡 `KNeighborsClassifier(n_neighbors=15)` 改成 3 或 30。

結果（KNN 未縮放那一列）：

| n_neighbors | precision | recall | 叫了幾次 |
|---|---:|---:|---:|
| 3 | 0.549 | 0.406 | 71 |
| 15（原本） | 0.792 | 0.198 | 24 |
| 30 | 1.000 | 0.052 | 5 |

解釋：問的鄰居越多，要「多數是真的」越難，所以叫得越少、越保守。30 個鄰居時只叫 5 次，全對，但 96 件真事件只抓到 5 件。

## 06｜決策樹切幾層

改：`candidate_models` 裡 `DecisionTreeClassifier(random_state=0)` 改成 `DecisionTreeClassifier(max_depth=1, random_state=0)`，再試 `max_depth=3`。

結果（決策樹那一列）：

| max_depth | precision | recall | 叫了幾次 |
|---|---:|---:|---:|
| 1 | 0.680 | 0.729 | 103 |
| 3 | 0.908 | 0.719 | 76 |
| 不限（原本） | 0.660 | 0.708 | 103 |

解釋：只切 1 刀太粗；不限深度會背答案；切 3 層在這份考卷上最好。越深不一定越好。

## 07｜換成 KNN，看錯誤地圖

改：`main` 裡 `model = final_model().fit(*xy(fit))` 改成

```python
model = make_pipeline(make_preprocessor(), KNeighborsClassifier(n_neighbors=15)).fit(*xy(fit))
```

結果：

| | 誤報 | 漏報 |
|---|---:|---:|
| 原本（梯度提升） | 14 | 27 |
| KNN | 5 | 29 |

解釋：KNN 比較謹慎，白跑少了 9 次，但漏掉的真事件多了 2 件。和 06「多考幾次」看到的一樣：KNN 的 precision 高、recall 低。

## 08｜客戶要求放寬

改：`story/profiles/factory.json` 的 `"auto_alert_min_precision": 0.9` 改成 `0.8`，跑完記得改回 0.9。
（設定檔是共用的，判斷器都從這裡讀客戶要求。）

結果：

| | HIGH | 自動叫人 | 其中真的 |
|---|---:|---:|---:|
| 原本（9 成） | 0.807 | 74 | 66 |
| 改成 8 成 | 0.369 | 91 | 73 |

解釋：客戶能接受多一點白跑，HIGH 就能往下降，自動叫人變多，多抓到 7 件真事件。LOW 沒變（0.019），因為「漏報不超過 1 成」那個要求沒改。

## 09｜模型檔被動過

改：先跑一次 `python judge.py` 存出模型，再在模型檔最後加一點東西，然後重新讀取：

```bash
python judge.py
echo x >> out/factory/judge.joblib
python -c "from judge import Judge; Judge()"
```

結果：

```
RuntimeError: 模型檔和設定檔對不上，拒絕啟動
```

解釋：模型檔只多了 2 個 byte，指紋就完全不同。平台讀檔前先比指紋，對不上就停，不會用到被改過或壞掉的模型。重跑一次 `python judge.py` 就會恢復。
