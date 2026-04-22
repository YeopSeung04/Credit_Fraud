"""
visualize.py — 포트폴리오급 시각화
다크 테마 / GitHub README에 바로 올릴 수 있는 품질
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from src.utils import get_logger

logger = get_logger(__name__)

# ── 전역 다크 테마 ────────────────────────────────────────────────────────────
BG = "#0d1117"
SURFACE = "#161b22"
BORDER = "#30363d"
TEXT = "#c9d1d9"
TEXT_BRIGHT = "#e6edf3"
MUTED = "#8b949e"

PALETTE = {
    "Logistic Regression": "#60a5fa",
    "Random Forest":       "#34d399",
    "XGBoost":             "#fb923c",
    "LightGBM":            "#a78bfa",
    "baseline":            "#4b5563",
}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": SURFACE,
    "axes.edgecolor": BORDER, "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "grid.color": BORDER,
    "grid.alpha": 0.5, "font.family": "monospace",
    "axes.titlecolor": TEXT_BRIGHT,
})


def _ax_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    if title:  ax.set_title(title, fontsize=11, pad=10, color=TEXT_BRIGHT)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(True, alpha=0.3, color=BORDER)
    ax.tick_params(labelsize=8)


# ── EDA 차트 ─────────────────────────────────────────────────────────────────
def plot_class_distribution(y, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("클래스 불균형 분석", fontsize=13, color=TEXT_BRIGHT, y=1.02)

    counts = y.value_counts().sort_index()
    labels = ["정상 (0)", "사기 (1)"]
    colors = ["#3b82f6", "#f97316"]

    # 좌: 원형 차트
    ax = axes[0]
    ax.set_facecolor(SURFACE)
    wedges, texts, autotexts = ax.pie(
        counts, labels=labels, colors=colors,
        autopct="%1.3f%%", startangle=90,
        textprops={"color": TEXT, "fontsize": 9},
        wedgeprops={"edgecolor": BG, "linewidth": 2},
    )
    for at in autotexts:
        at.set_color(TEXT_BRIGHT)
        at.set_fontsize(10)
    ax.set_title("클래스 비율", fontsize=11, color=TEXT_BRIGHT)

    # 우: 카운트 바 차트 (log scale)
    ax2 = axes[1]
    ax2.set_facecolor(SURFACE)
    bars = ax2.bar(labels, counts.values, color=colors, edgecolor=BG, linewidth=1.5, width=0.5)
    ax2.set_yscale("log")
    for bar, cnt in zip(bars, counts.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                 f"{cnt:,}", ha="center", va="bottom", fontsize=10, color=TEXT_BRIGHT)
    _ax_style(ax2, title="거래 건수 (log scale)", ylabel="건수")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    return fig


def plot_amount_distribution(df, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("거래 금액 분포 (정상 vs 사기)", fontsize=13, color=TEXT_BRIGHT)

    fraud  = df[df["Class"] == 1]["Amount"]
    normal = df[df["Class"] == 0]["Amount"]

    for ax, data, label, color, title in [
        (axes[0], normal, "정상", "#3b82f6", "정상 거래 금액"),
        (axes[1], fraud,  "사기", "#f97316", "사기 거래 금액"),
    ]:
        ax.hist(data, bins=60, color=color, alpha=0.8, edgecolor="none")
        ax.axvline(data.mean(), color="#fbbf24", linestyle="--", linewidth=1.5,
                   label=f"평균: ${data.mean():.0f}")
        ax.axvline(data.median(), color="#a78bfa", linestyle=":", linewidth=1.5,
                   label=f"중앙값: ${data.median():.0f}")
        _ax_style(ax, title=title, xlabel="금액 ($)", ylabel="건수")
        ax.legend(fontsize=8, facecolor=SURFACE, edgecolor=BORDER)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    return fig


# ── 모델 비교 차트 ────────────────────────────────────────────────────────────
def plot_roc_pr_curves(results: dict, y_test, save_path=None):
    """ROC + PR 커브 — 불균형 데이터에서 PR이 더 중요함을 함께 보여줌"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("ROC Curve  &  Precision-Recall Curve", fontsize=13, color=TEXT_BRIGHT)

    ax_roc, ax_pr = axes

    # baseline
    ax_roc.plot([0, 1], [0, 1], "--", color=PALETTE["baseline"], linewidth=1, alpha=0.5, label="Random")
    ax_pr.axhline(y_test.mean(), color=PALETTE["baseline"], linestyle="--",
                  linewidth=1, alpha=0.5, label=f"Random ({y_test.mean():.4f})")

    for name, res in results.items():
        color = PALETTE.get(name, "#94a3b8")
        y_prob = res["model"].predict_proba(res.get("X_test"))[:, 1] if "X_test" in res else None
        if y_prob is None:
            continue

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)

        ax_roc.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{name} (AUC={res['roc_auc']:.4f})")
        ax_pr.plot(rec, prec, color=color, linewidth=2,
                   label=f"{name} (AP={res['average_precision']:.4f})")

    _ax_style(ax_roc, title="ROC Curve", xlabel="FPR (False Positive Rate)", ylabel="TPR (True Positive Rate)")
    _ax_style(ax_pr,  title="PR Curve ★ 주 평가 지표", xlabel="Recall", ylabel="Precision")
    ax_roc.set_xlim(-0.01, 1.01); ax_roc.set_ylim(-0.01, 1.05)
    ax_pr.set_xlim(-0.01, 1.01);  ax_pr.set_ylim(-0.01, 1.05)

    for ax in axes:
        ax.legend(fontsize=8.5, facecolor=SURFACE, edgecolor=BORDER, loc="lower left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    return fig


def plot_model_comparison(compare_df: pd.DataFrame, save_path=None):
    """모델별 지표 바 차트"""
    metrics = ["ROC-AUC", "Avg Precision", "F1", "Precision", "Recall"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("모델 성능 비교", fontsize=13, color=TEXT_BRIGHT)

    for ax, metric in zip(axes, metrics):
        colors = [PALETTE.get(m, "#94a3b8") for m in compare_df["Model"]]
        bars = ax.barh(compare_df["Model"], compare_df[metric], color=colors,
                       edgecolor="none", height=0.5)
        for bar, val in zip(bars, compare_df[metric]):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", fontsize=8, color=TEXT_BRIGHT)
        _ax_style(ax, title=metric)
        ax.set_xlim(0, 1.1)
        ax.invert_yaxis()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    return fig


def plot_confusion_matrix(y_true, y_pred, model_name="", save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(BG)

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_facecolor(SURFACE)

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            label = ["TN", "FP", "FN", "TP"][i*2+j]
            ax.text(j, i, f"{cm[i,j]:,}\n({label})",
                    ha="center", va="center", fontsize=12,
                    color="white" if cm[i,j] > thresh else TEXT)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["정상 예측", "사기 예측"], fontsize=9)
    ax.set_yticklabels(["실제 정상", "실제 사기"], fontsize=9)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=11, color=TEXT_BRIGHT, pad=12)

    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    return fig


def plot_threshold_analysis(y_true, y_prob, save_path=None):
    """임계값별 Precision / Recall / F1 변화"""
    thresholds = np.linspace(0.01, 0.99, 200)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f = 2*p*r/(p+r+1e-9)
        precisions.append(p); recalls.append(r); f1s.append(f)

    best_idx = np.argmax(f1s)
    best_t = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)

    ax.plot(thresholds, precisions, color="#3b82f6", linewidth=2, label="Precision")
    ax.plot(thresholds, recalls,   color="#f97316", linewidth=2, label="Recall")
    ax.plot(thresholds, f1s,       color="#34d399", linewidth=2.5, label="F1")
    ax.axvline(best_t, color="#fbbf24", linestyle="--", linewidth=1.5,
               label=f"최적 임계값: {best_t:.3f} (F1={f1s[best_idx]:.4f})")

    _ax_style(ax, title="임계값 분석 — Precision / Recall / F1 트레이드오프",
              xlabel="Threshold", ylabel="Score")
    ax.legend(fontsize=9, facecolor=SURFACE, edgecolor=BORDER)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    return fig
