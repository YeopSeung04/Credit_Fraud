"""
explain.py — SHAP 기반 모델 해석
"왜 이 거래가 사기로 판단됐는가?" 를 설명
MLE 포트폴리오의 핵심 차별화 포인트
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
from src.utils import get_logger

logger = get_logger(__name__)

# 전역 스타일
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#0d1117",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#c9d1d9",
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "font.family": "monospace",
})


def get_explainer(model, X_sample, model_name: str = ""):
    """
    모델 타입에 맞는 SHAP Explainer 선택
    Tree 모델 → TreeExplainer (빠름)
    선형 모델 → LinearExplainer
    기타 → KernelExplainer (느림, 샘플 필요)
    """
    model_name_lower = model_name.lower()

    if any(k in model_name_lower for k in ["xgboost", "lightgbm", "random forest", "gradient"]):
        logger.info("TreeExplainer 사용")
        return shap.TreeExplainer(model)
    elif "logistic" in model_name_lower:
        logger.info("LinearExplainer 사용")
        return shap.LinearExplainer(model, X_sample)
    else:
        logger.info("KernelExplainer 사용 (느릴 수 있음)")
        bg = shap.sample(X_sample, 100)
        return shap.KernelExplainer(model.predict_proba, bg)


def compute_shap(pipeline, X: pd.DataFrame, model_name: str, n_samples: int = 500):
    """
    Pipeline에서 전처리 분리 후 SHAP 계산
    Returns: shap_values (array), X_transformed (DataFrame)
    """
    # Pipeline의 마지막 스텝(모델)과 나머지 전처리 분리
    steps = list(pipeline.named_steps.items())
    model_step_name, model = steps[-1]

    # 전처리만 적용
    X_proc = X.copy()
    for name, transformer in steps[:-1]:
        if hasattr(transformer, 'transform') and name != 'sampler':
            try:
                X_proc = transformer.transform(X_proc)
            except Exception:
                pass

    # 샘플링 (SHAP은 계산 비용이 높음)
    n = min(n_samples, len(X_proc))
    if hasattr(X_proc, 'iloc'):
        X_sample = X_proc.iloc[:n]
    else:
        X_sample = X_proc[:n]
        X_sample = pd.DataFrame(X_sample, columns=X.columns[:X_sample.shape[1]])

    explainer = get_explainer(model, X_sample, model_name)

    shap_values = explainer.shap_values(X_sample)

    # 이진 분류: [class0, class1] → class1만 사용
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values, X_sample


def plot_shap_summary(shap_values, X: pd.DataFrame, title: str = "", save_path: str = None):
    """SHAP Summary Plot — 피처 중요도 + 방향성"""
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0d1117")

    feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f"V{i}" for i in range(shap_values.shape[1])]
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:15]  # Top 15

    # 컬러맵: SHAP값 부호로 색상 구분
    colors_pos = "#f97316"   # 오렌지: 사기 방향
    colors_neg = "#3b82f6"   # 블루: 정상 방향

    for rank, feat_idx in enumerate(reversed(order)):
        y_pos = rank
        vals = shap_values[:, feat_idx]
        feat_vals = X.iloc[:, feat_idx] if hasattr(X, 'iloc') else X[:, feat_idx]

        # 정규화
        feat_norm = (feat_vals - feat_vals.min()) / (feat_vals.max() - feat_vals.min() + 1e-9)

        scatter_colors = [colors_pos if v > 0 else colors_neg for v in vals]
        ax.scatter(vals, [y_pos] * len(vals), c=scatter_colors, alpha=0.4, s=8, linewidths=0)
        ax.scatter(vals.mean(), y_pos, c="#f59e0b", s=60, zorder=5, marker="D")

    feat_labels = [feature_names[i] for i in reversed(order)]
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(feat_labels, fontsize=10)
    ax.axvline(x=0, color="#8b949e", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP Value (← 정상  |  사기 →)", fontsize=11)
    ax.set_title(title or "SHAP Feature Importance", fontsize=13, pad=15, color="#e6edf3")
    ax.grid(axis="x", alpha=0.3)

    pos_patch = mpatches.Patch(color=colors_pos, label="사기 방향 기여")
    neg_patch = mpatches.Patch(color=colors_neg, label="정상 방향 기여")
    ax.legend(handles=[pos_patch, neg_patch], loc="lower right", fontsize=9,
              facecolor="#161b22", edgecolor="#30363d")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        logger.info(f"SHAP summary 저장: {save_path}")
    return fig


def plot_shap_waterfall_single(shap_values, X: pd.DataFrame,
                                sample_idx: int = 0,
                                title: str = "", save_path: str = None):
    """
    단일 거래 SHAP Waterfall — "왜 이 거래가 사기인가?"
    면접에서 '설명 가능한 AI' 질문에 대응하는 핵심 차트
    """
    feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f"V{i}" for i in range(shap_values.shape[1])]
    sv = shap_values[sample_idx]
    order = np.argsort(np.abs(sv))[::-1][:12]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0d1117")

    labels = [feature_names[i] for i in reversed(order)]
    values = [sv[i] for i in reversed(order)]
    colors = ["#f97316" if v > 0 else "#3b82f6" for v in values]

    bars = ax.barh(labels, values, color=colors, edgecolor="none", height=0.6)

    for bar, val in zip(bars, values):
        ax.text(
            val + (0.002 if val >= 0 else -0.002),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}",
            va="center", ha="left" if val >= 0 else "right",
            fontsize=9, color="#e6edf3"
        )

    ax.axvline(x=0, color="#8b949e", linewidth=0.8)
    ax.set_xlabel("SHAP Value", fontsize=11)
    ax.set_title(
        title or f"개별 거래 설명 (Sample #{sample_idx})\n← 정상 기여  |  사기 기여 →",
        fontsize=12, pad=12, color="#e6edf3"
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        logger.info(f"SHAP waterfall 저장: {save_path}")
    return fig


def get_feature_importance_df(shap_values, feature_names: list) -> pd.DataFrame:
    """SHAP 기반 피처 중요도 DataFrame"""
    return pd.DataFrame({
        "feature": feature_names,
        "shap_mean_abs": np.abs(shap_values).mean(axis=0),
        "shap_mean": shap_values.mean(axis=0),
    }).sort_values("shap_mean_abs", ascending=False).reset_index(drop=True)
