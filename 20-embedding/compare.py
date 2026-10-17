"""20｜巡檢回報要不要通知：關鍵字規則 vs TF-IDF＋LR vs bge embedding＋LR。

    HF_HUB_OFFLINE=1 python 20-embedding/compare.py
"""
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import score_row
from textlib import embed, sentence_pattern, keyword_rule, load_encoder, load_new_reports, load_reports, make_embedding_lr, make_tfidf

reports = load_reports()
train, test = train_test_split(reports, test_size=0.3, stratify=reports.needs_notify, random_state=0)

print("== 1. reports.csv：分層切割 7:3，seed 0 ==")
print(f"全部 {len(reports)} 筆（正例 {reports.needs_notify.mean():.3f}）；train {len(train)}（{train.needs_notify.mean():.3f}）、"
      f"test {len(test)}（{test.needs_notify.mean():.3f}）")
print(reports.sample(4, random_state=3)[["text", "needs_notify"]].to_string(index=False))

encoder = load_encoder()
t0 = time.perf_counter()
vectors = embed(encoder, reports.text)
encode_ms = (time.perf_counter() - t0) / len(reports) * 1000
print("\n== 2. embedding：每句一個向量 ==")
print("形狀：", vectors.shape, "| 長度：", np.round(np.linalg.norm(vectors[:3], axis=1), 3))
again = embed(encoder, reports.text[:20])
print("同樣 20 句再算一次，最大差異 < 1e-6：", bool(np.abs(again - vectors[:20]).max() < 1e-6))
pos = reports.index[reports.text.str.contains("沒戴安全帽")][0]
for label, q in [("同義（裸頭）", "裸頭"), ("反義（已戴好）", "皆已戴好"), ("無關（照明）", "照明不足")]:
    j = reports.index[reports.text.str.contains(q)][0]
    print(f"「{reports.text[pos]}」vs {label}「{reports.text[j]}」：cosine {vectors[pos] @ vectors[j]:.3f}")

tfidf = make_tfidf().fit(train.text, train.needs_notify)
lr = make_embedding_lr().fit(vectors[train.index], train.needs_notify)
preds = {"關鍵字規則": keyword_rule(test.text),
         "TF-IDF+LR": tfidf.predict(test.text),
         "bge+LR": lr.predict(vectors[test.index])}
print("\n== 3. 成績表（test 144 筆）==")
print("（模型大小）TF-IDF 詞表", len(tfidf[0].vocabulary_), "個字元片段；bge 參數",
      f"{sum(p.numel() for p in encoder.parameters()) / 1e6:.1f} 百萬個")
print(pd.DataFrame([score_row(k, test.needs_notify, v) for k, v in preds.items()]).to_string(index=False))

print("\n== 4. 為什麼兩個模型都滿分？==")
seen = test.text.isin(set(train.text))
print(f"test 裡有 {seen.sum()} 句（{seen.mean():.3f}）一字不差出現在 train")
patterns = sentence_pattern(reports.text)
print("不同的句子只有", reports.text.nunique(), "種；拿掉地點與結尾語後只有", patterns.nunique(), "種句型")
print("test 的句型在 train 出現過的比例：", f"{patterns[test.index].isin(set(patterns[train.index])).mean():.3f}")

new = load_new_reports()
new_vec = embed(encoder, new.text)
print("\n== 5. 另一位巡檢員用自己的話寫的 20 句（模型沒看過的說法）==")
out = new.assign(關鍵字=keyword_rule(new.text), TFIDF=tfidf.predict(new.text), bge=lr.predict(new_vec),
                 P_tfidf=tfidf.predict_proba(new.text)[:, 1].round(2), P_bge=lr.predict_proba(new_vec)[:, 1].round(2))
print(pd.DataFrame([score_row(n, new.needs_notify, out[c]) for n, c in
                    [("關鍵字規則", "關鍵字"), ("TF-IDF+LR", "TFIDF"), ("bge+LR", "bge")]]).to_string(index=False))
wrong = out[(out.關鍵字 != out.needs_notify) | (out.TFIDF != out.needs_notify) | (out.bge != out.needs_notify)]
print("\n至少一種方法答錯的句子：")
print(wrong[wrong.TFIDF.ne(wrong.needs_notify) | wrong.bge.ne(wrong.needs_notify)].to_string(index=False))
print(f"關鍵字規則漏掉 {((out.關鍵字 == 0) & (out.needs_notify == 1)).sum()} 句（只抓到「沒戴」那一句）")

print("\n== 6. TF-IDF+LR 看得到理由：權重最大的字元片段 ==")
coef = pd.Series(tfidf[-1].coef_[0], index=tfidf[0].get_feature_names_out()).sort_values()
print("推向通知：", "、".join(f"{k}({v:.2f})" for k, v in coef.tail(8)[::-1].items()))
print("推向不通知：", "、".join(f"{k}({v:.2f})" for k, v in coef.head(8).items()))

t0 = time.perf_counter()
for _ in range(20):
    tfidf.predict(new.text)
tfidf_ms = (time.perf_counter() - t0) / (20 * len(new)) * 1000
print(f"\n（本機計時，每次不同，不貼進講義）encode 每句 {encode_ms:.2f} ms；TF-IDF+LR 每句 {tfidf_ms:.3f} ms")
