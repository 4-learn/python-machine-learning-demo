"""20｜三種文字分類方法共用的小工具（21 節以後要用也可以 import）。"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 只用本機快取，不連網

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from common import DATA, ROOT

MODEL_ID = "BAAI/bge-small-zh-v1.5"
REVISION = "7999e1d3359715c523056ef9478215996d62a620"  # 與 LLM 課 16 節相同
KEYWORDS = ["沒戴", "未配戴", "未戴", "裸頭", "沒有任何防護", "沒有戴"]


def load_reports() -> pd.DataFrame:
    return pd.read_csv(DATA / "reports.csv")


def load_new_reports() -> pd.DataFrame:
    return pd.read_csv(ROOT / "20-embedding" / "new_reports.csv")


PLACES = r"^(三樓鋼筋區|B 棟高架平台|物料堆置區|北側出入口|五樓外牆鷹架|地下室模板區|吊掛作業區|休息區)"
ENDINGS = r"(，請處理。|，已拍照。|（下午巡檢）|，麻煩主管看一下。|。)$"


def sentence_pattern(texts: pd.Series) -> pd.Series:
    """拿掉開頭的地點與結尾語，剩下的就是「句型」。同一句型的回報要放在同一邊（像 03 節的 incident）。"""
    return texts.str.replace(PLACES, "", regex=True).str.replace(ENDINGS, "", regex=True)


def keyword_rule(texts: pd.Series) -> pd.Series:
    """方法 1：出現任何一個關鍵字就通知。"""
    return texts.str.contains("|".join(KEYWORDS)).astype(int)


def make_tfidf():
    """方法 2：字元 n-gram（1–2 個字）TF-IDF＋邏輯迴歸。中文不用斷詞。"""
    return make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(1, 2)), LogisticRegression(max_iter=1000))


def load_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_ID, revision=REVISION, device="cpu")


def embed(encoder, texts) -> "np.ndarray":
    """方法 3 的前半：文字 → 512 維向量（正規化成長度 1）。文件不加查詢前綴，與 LLM 課 16 節一致。"""
    return encoder.encode(list(texts), batch_size=32, normalize_embeddings=True, show_progress_bar=False)


def make_embedding_lr():
    """方法 3 的後半：向量 → 邏輯迴歸（07 節的同一個模型）。"""
    return LogisticRegression(max_iter=1000)
