"""
data_loader.py — 데이터 로드 & 검증
실제 Kaggle creditcard.csv 또는 데모용 합성 데이터 생성
"""
import os
import numpy as np
import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)

EXPECTED_COLS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "Class"]


def load_data(cfg: dict) -> pd.DataFrame:
    """
    실제 데이터(Kaggle CSV)가 있으면 로드,
    없으면 동일 구조의 데모 데이터 생성 후 경고 출력.
    """
    path = cfg["data"]["raw_path"]

    if os.path.exists(path):
        logger.info(f"실제 데이터 로드: {path}")
        df = pd.read_csv(path)
        _validate(df)
        logger.info(f"로드 완료: {df.shape} | 사기율: {df['Class'].mean():.4%}")
        return df
    else:
        logger.warning("=" * 60)
        logger.warning("실제 데이터 없음 → 데모 데이터로 실행")
        logger.warning("실제 실행: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        logger.warning("다운로드 후 data/creditcard.csv 에 저장하세요")
        logger.warning("=" * 60)
        return _make_demo_data()


def _validate(df: pd.DataFrame):
    missing = set(EXPECTED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")
    if df["Class"].nunique() != 2:
        raise ValueError("Class 컬럼은 0/1 이진값이어야 합니다")
    null_rate = df.isnull().mean()
    high_null = null_rate[null_rate > 0.3]
    if not high_null.empty:
        logger.warning(f"결측치 30% 초과 컬럼: {high_null.index.tolist()}")


def _make_demo_data(n: int = 284_807, fraud_ratio: float = 0.001727) -> pd.DataFrame:
    """
    Kaggle creditcard.csv 와 동일한 통계 구조를 가진 데모 데이터.
    V1-V28: PCA 변환 피처 (정규분포)
    Amount: 지수분포 (실제 분포 근사)
    Time: 48시간 균등분포
    Class: 불균형 (0.17% 사기)
    """
    logger.info(f"데모 데이터 생성 중: {n:,}건")
    rng = np.random.default_rng(42)

    n_fraud = int(n * fraud_ratio)
    n_normal = n - n_fraud

    # ── 정상 거래 ──────────────────────────────────────────
    normal_v = rng.standard_normal((n_normal, 28))
    normal_amount = rng.exponential(scale=88.35, size=n_normal).clip(0, 25691)
    normal_time = rng.uniform(0, 172792, n_normal)

    # ── 사기 거래 (정상과 다른 분포) ───────────────────────
    # 실제 데이터 분석 결과: V1, V3, V4, V9, V10, V11, V14, V16이 큰 차이
    fraud_v = rng.standard_normal((n_fraud, 28))
    shift = np.zeros(28)
    shift[[0, 2, 3, 8, 9, 10, 13, 15]] = [-4.8, -4.5, 4.2, -3.6, -4.0, 4.5, -4.2, -3.8]
    fraud_v += shift
    fraud_amount = rng.exponential(scale=122.21, size=n_fraud).clip(0, 2125)
    fraud_time = rng.uniform(0, 172792, n_fraud)

    # ── 결합 ───────────────────────────────────────────────
    V_all = np.vstack([normal_v, fraud_v])
    amount_all = np.concatenate([normal_amount, fraud_amount])
    time_all = np.concatenate([normal_time, fraud_time])
    class_all = np.array([0] * n_normal + [1] * n_fraud)

    df = pd.DataFrame(V_all, columns=[f"V{i}" for i in range(1, 29)])
    df["Amount"] = amount_all.round(2)
    df["Time"] = time_all.round(0)
    df["Class"] = class_all

    # 섞기
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info(f"데모 데이터 생성 완료: {df.shape} | 사기율: {df['Class'].mean():.4%}")
    return df


def describe_data(df: pd.DataFrame) -> dict:
    """EDA용 기초 통계 반환"""
    fraud = df[df["Class"] == 1]
    normal = df[df["Class"] == 0]

    return {
        "total": len(df),
        "fraud_count": len(fraud),
        "normal_count": len(normal),
        "fraud_rate": df["Class"].mean(),
        "amount_stats": {
            "fraud_mean": fraud["Amount"].mean(),
            "normal_mean": normal["Amount"].mean(),
            "fraud_median": fraud["Amount"].median(),
            "normal_median": normal["Amount"].median(),
        },
        "null_counts": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
    }
