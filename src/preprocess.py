"""
preprocess.py — sklearn Pipeline 기반 전처리
재현 가능하고 train/test leakage 없는 구조
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from src.utils import get_logger

logger = get_logger(__name__)


# ── Custom Transformer ─────────────────────────────────────────────────────────
class AmountLogTransformer(BaseEstimator, TransformerMixin):
    """Amount 컬럼 log1p 변환 후 정규화"""
    def __init__(self, amount_col_idx: int = 0):
        self.amount_col_idx = amount_col_idx

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[:, self.amount_col_idx] = np.log1p(X[:, self.amount_col_idx])
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    금융 도메인 피처 엔지니어링
    - Amount_log: 로그 변환 금액
    - Hour: 거래 시간대 (0-23)
    - IsNight: 야간 거래 여부 (22시-06시)
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Amount 로그 변환
        X["Amount_log"] = np.log1p(X["Amount"])

        # 시간 피처 (Time은 초 단위, 48시간 주기)
        if "Time" not in (self.cfg["data"].get("drop_cols") or []):
            X["Hour"] = (X["Time"] % 86400 // 3600).astype(int)
            X["IsNight"] = X["Hour"].apply(lambda h: 1 if (h >= 22 or h <= 6) else 0)

        return X


# ── Split ─────────────────────────────────────────────────────────────────────
def split_data(df: pd.DataFrame, cfg: dict):
    """
    Train / Validation / Test 분리
    Stratify로 사기 비율 유지
    """
    target = cfg["data"]["target_col"]
    drop_cols = cfg["data"].get("drop_cols", [])
    feature_cols = [c for c in df.columns if c != target and c not in drop_cols]

    X = df[feature_cols]
    y = df[target]

    seed = cfg["project"]["seed"]
    test_size = cfg["data"]["test_size"]
    val_size = cfg["data"]["val_size"]

    # 1차 분리: train+val / test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # 2차 분리: train / val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio, random_state=seed, stratify=y_trainval
    )

    logger.info(f"Train:      {len(X_train):,} (사기: {y_train.sum():,})")
    logger.info(f"Validation: {len(X_val):,}   (사기: {y_val.sum():,})")
    logger.info(f"Test:       {len(X_test):,}  (사기: {y_test.sum():,})")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


# ── Scaler ────────────────────────────────────────────────────────────────────
def get_scaler(cfg: dict):
    strategy = cfg["preprocessing"].get("scaling", "standard")
    return {
        "standard": StandardScaler(),
        "robust": RobustScaler(),       # 이상치에 강건
        "minmax": MinMaxScaler(),
    }[strategy]


# ── Imbalance Handler ─────────────────────────────────────────────────────────
def get_sampler(cfg: dict):
    strategy = cfg["imbalance"]["strategy"]
    seed = cfg["project"]["seed"]
    ratio = cfg["imbalance"]["sampling_ratio"]

    if strategy == "smote":
        return SMOTE(
            sampling_strategy=ratio,
            k_neighbors=cfg["imbalance"]["smote_k_neighbors"],
            random_state=seed,
        )
    elif strategy == "adasyn":
        return ADASYN(sampling_strategy=ratio, random_state=seed)
    elif strategy == "none":
        return None
    else:
        raise ValueError(f"알 수 없는 imbalance strategy: {strategy}")


# ── Pipeline Builder ──────────────────────────────────────────────────────────
def build_pipeline(model, cfg: dict, with_sampler: bool = True):
    """
    sklearn Pipeline 구성
    scaler → (sampler) → model
    Logistic Regression처럼 스케일 의존 모델에 사용
    """
    scaler = get_scaler(cfg)
    steps = [("scaler", scaler)]

    if with_sampler and cfg["imbalance"]["strategy"] != "none":
        sampler = get_sampler(cfg)
        steps.append(("sampler", sampler))
        pipeline_cls = ImbPipeline   # imblearn pipeline (sampler 지원)
    else:
        pipeline_cls = Pipeline

    steps.append(("model", model))
    return pipeline_cls(steps)


# ── Feature Engineering Pipeline ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """피처 엔지니어링 적용 후 불필요 컬럼 제거"""
    fe = FeatureEngineer(cfg)
    df = fe.transform(df)

    drop_cols = cfg["data"].get("drop_cols", [])
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    logger.info(f"전처리 완료: {df.shape} | 피처: {df.shape[1] - 1}개")
    return df
