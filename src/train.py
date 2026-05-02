"""
train.py — MLflow 실험 추적 포함 학습 모듈
각 모델 실험을 자동으로 기록 → 재현 가능한 실험 관리
"""
import os
import pickle
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
except (ImportError, OSError) as exc:
    LGBMClassifier = None
    LIGHTGBM_IMPORT_ERROR = exc
else:
    LIGHTGBM_IMPORT_ERROR = None

from src.preprocess import build_pipeline
from src.evaluate import evaluate, get_curve_data
from src.utils import get_logger

logger = get_logger(__name__)


def get_models(cfg: dict, scale_pos: float = None) -> dict:
    """
    config에서 모델 인스턴스 생성
    scale_pos: XGBoost의 pos_weight (자동 계산값 사용)
    """
    seed = cfg["project"]["seed"]
    lrc = cfg["models"]["logistic_regression"]
    rfc = cfg["models"]["random_forest"]
    xgbc = cfg["models"]["xgboost"]
    lgbc = cfg["models"]["lightgbm"]

    if scale_pos is not None:
        xgbc = {**xgbc, "scale_pos_weight": scale_pos}

    models = {
        "Logistic Regression": LogisticRegression(
            C=lrc["C"],
            max_iter=lrc["max_iter"],
            class_weight=lrc["class_weight"],
            solver=lrc["solver"],
            random_state=seed,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=rfc["n_estimators"],
            max_depth=rfc["max_depth"],
            min_samples_leaf=rfc["min_samples_leaf"],
            class_weight=rfc["class_weight"],
            n_jobs=rfc["n_jobs"],
            random_state=seed,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=xgbc["n_estimators"],
            max_depth=xgbc["max_depth"],
            learning_rate=xgbc["learning_rate"],
            subsample=xgbc["subsample"],
            colsample_bytree=xgbc["colsample_bytree"],
            scale_pos_weight=xgbc["scale_pos_weight"],
            eval_metric=xgbc["eval_metric"],
            tree_method=xgbc["tree_method"],
            random_state=seed,
            verbosity=0,
        ),
    }

    if LGBMClassifier is None:
        logger.warning(f"LightGBM unavailable; skipping LightGBM. Reason: {LIGHTGBM_IMPORT_ERROR}")
    else:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=lgbc["n_estimators"],
            num_leaves=lgbc["num_leaves"],
            learning_rate=lgbc["learning_rate"],
            min_child_samples=lgbc["min_child_samples"],
            class_weight=lgbc["class_weight"],
            verbose=lgbc["verbose"],
            random_state=seed,
        )

    return models


def train_one(
    name: str,
    model,
    X_train, y_train,
    X_val, y_val,
    cfg: dict,
    use_pipeline: bool = True,
) -> dict:
    """
    단일 모델 학습 + MLflow 기록
    Returns: 검증 세트 평가 결과 dict
    """
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=name):
        # 파이프라인 구성 (LR은 스케일링 필요, 트리 모델은 불필요)
        needs_scale = "Logistic" in name
        if use_pipeline:
            pipe = build_pipeline(model, cfg, with_sampler=True)
        else:
            pipe = model

        logger.info(f"\n{'='*50}")
        logger.info(f"  학습 시작: {name}")
        logger.info(f"  Train: {len(X_train):,}  Val: {len(X_val):,}")

        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_val)[:, 1]

        # 평가
        threshold_strategy = cfg["evaluation"]["threshold_strategy"]
        metrics = evaluate(y_prob=y_prob, y_true=y_val,
                           strategy=threshold_strategy, verbose=True)

        # MLflow 기록
        mlflow.log_params({
            "model": name,
            "imbalance_strategy": cfg["imbalance"]["strategy"],
            "scaling": cfg["preprocessing"]["scaling"],
            "threshold_strategy": threshold_strategy,
        })
        mlflow.log_metrics(metrics)

        # 모델 저장
        model_dir = cfg["output"]["model_dir"]
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{name.replace(' ', '_')}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(pipe, f)
        mlflow.sklearn.log_model(pipe, artifact_path=name.replace(" ", "_"))

        logger.info(f"  모델 저장: {model_path}")

    return {**metrics, "model": pipe, "name": name}


def train_all(
    X_train, y_train,
    X_val, y_val,
    cfg: dict,
) -> dict:
    """
    전체 모델 학습 및 비교
    Returns: 모델명 → 결과 dict
    """
    # XGBoost scale_pos_weight 자동 계산
    scale_pos = float((y_train == 0).sum() / (y_train == 1).sum())
    logger.info(f"XGBoost scale_pos_weight: {scale_pos:.1f}")

    models = get_models(cfg, scale_pos=scale_pos)
    results = {}

    for name, model in models.items():
        result = train_one(
            name=name,
            model=model,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            cfg=cfg,
        )
        results[name] = result

    return results


def load_model(name: str, cfg: dict):
    """저장된 모델 로드"""
    path = os.path.join(cfg["output"]["model_dir"],
                        f"{name.replace(' ', '_')}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)
