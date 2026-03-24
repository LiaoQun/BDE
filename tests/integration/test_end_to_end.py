"""
End-to-end integration test covering the full pipeline:
    train -> save model -> predict
Uses a minimal subset of data and epochs to keep runtime short.
"""
import os
import shutil
import pytest
import pandas as pd
from datetime import datetime

from src.config.schema import MainConfig, DataConfig, ModelConfig, TrainConfig
from src.config.loader import save_flattened_config


@pytest.fixture
def minimal_config(tmp_path):
    """
    Builds a minimal MainConfig that runs fast and writes all outputs
    into a pytest-managed temporary directory.
    """
    cfg = MainConfig(
        data=DataConfig(
            data_paths=["examples/test_data.csv.gz"],
            dataset_dir=str(tmp_path / "dataset"),
            test_size=0.1,
            val_size=0.1,
            random_seed=42,
            sample_percentage=0.02,   # 只用 2% 的資料，跑得快
            featurizer_type="TokenFeaturizer",
            target_columns=["bde"],
        ),
        model=ModelConfig(
            atom_features=32,          # 縮小模型，跑得快
            num_messages=2,
            num_tasks=1,
        ),
        train=TrainConfig(
            epochs=3,
            lr=0.001,
            batch_size=16,
            model_save_path="bde_model.pt",
            output_dir=str(tmp_path / "training_runs"),
            early_stopping_patience=10,
        ),
    )
    return cfg


def test_training_produces_expected_artifacts(minimal_config, tmp_path):
    """
    Runs a full training loop and checks that all expected output files
    are created in the run directory.
    """
    run_dir = _run_training(minimal_config, tmp_path)

    # 確認所有預期產出物都存在
    expected_files = [
        "bde_model.pt",
        "config.yaml",
        "training.log",
        "training_log.csv",
        "training_curve.png",
        "vocab.json",
        "parity_plot_all.png",
        "predictions_train.csv",
        "predictions_test.csv",
    ]
    for filename in expected_files:
        filepath = os.path.join(run_dir, filename)
        assert os.path.exists(filepath), f"Expected artifact missing: {filename}"


def test_training_log_has_correct_columns(minimal_config, tmp_path):
    """
    Checks that the training log CSV contains the expected columns
    and has at least one epoch recorded.
    """
    run_dir = _run_training(minimal_config, tmp_path)

    log_path = os.path.join(run_dir, "training_log.csv")
    df = pd.read_csv(log_path)

    assert "epoch" in df.columns
    assert "train_loss" in df.columns
    assert "val_loss" in df.columns
    assert len(df) >= 1
    assert (df["train_loss"] >= 0).all()
    assert (df["val_loss"] >= 0).all()


def test_flattened_config_is_complete(minimal_config, tmp_path):
    """
    Checks that the saved config.yaml is a complete standalone file
    (no _base_ reference) and contains the correct values.
    """
    import yaml

    run_dir = _run_training(minimal_config, tmp_path)

    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, "r") as f:
        saved = yaml.safe_load(f)

    assert "_base_" not in saved
    assert saved["model"]["atom_features"] == 32
    assert saved["model"]["num_messages"] == 2
    assert saved["train"]["epochs"] == 3
    assert saved["data"]["featurizer_type"] == "TokenFeaturizer"


def test_prediction_runs_after_training(minimal_config, tmp_path):
    from src.config.loader import load_config
    from src.inference.predictor import get_bde_predictions

    run_dir = _run_training(minimal_config, tmp_path)

    cfg = load_config(os.path.join(run_dir, "config.yaml"))

    results_df = get_bde_predictions(
        smiles=["CCO", "CCC"],
        model_path=os.path.join(run_dir, cfg.train.model_save_path),
        vocab_path=os.path.join(run_dir, "vocab.json"),
        featurizer_type=cfg.data.featurizer_type,
        atom_features=cfg.model.atom_features,
        num_messages=cfg.model.num_messages,
        num_tasks=cfg.model.num_tasks,
        target_columns=cfg.data.target_columns,
        device="cpu",
    )

    assert not results_df.empty
    assert "molecule" in results_df.columns
    assert "bond_index" in results_df.columns
    assert "bde_pred" in results_df.columns
    assert "fragment1" in results_df.columns
    assert "fragment2" in results_df.columns
    assert results_df["bde_pred"].notna().all()
    assert results_df["bde_pred"].apply(lambda x: abs(x) < 1000).all()


def test_predictions_csv_has_expected_columns(minimal_config, tmp_path):
    """
    Checks that the predictions_test.csv produced by evaluate()
    contains the expected ground-truth and prediction columns.
    """
    run_dir = _run_training(minimal_config, tmp_path)

    pred_path = os.path.join(run_dir, "predictions_test.csv")
    df = pd.read_csv(pred_path)

    assert "molecule" in df.columns
    assert "bond_index" in df.columns
    assert "bde_pred" in df.columns
    assert "bde" in df.columns          # ground-truth 欄位
    assert not df.empty


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_run_dir_cache: dict = {}


def _run_training(cfg: MainConfig, tmp_path) -> str:
    """
    Runs training once per tmp_path and caches the run_dir,
    so multiple tests sharing the same fixture don't re-train.

    Mirrors the logic of ``train.main()``:
    1. Creates a timestamped run_dir under cfg.train.output_dir.
    2. Saves the flattened config.yaml into run_dir.
    3. Calls ``run_training(cfg, run_dir)`` directly.
    4. Cleans up the temporary dataset directory afterward.
    """
    cache_key = str(tmp_path)
    if cache_key in _run_dir_cache:
        return _run_dir_cache[cache_key]

    from train import run_training

    # 建立與 main() 相同結構的 run_dir
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(cfg.train.output_dir, run_timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # 在 run_dir 內儲存 config.yaml（與 main() 行為一致）
    save_flattened_config(cfg, run_dir)

    # 執行訓練；run_training 不再負責建立 run_dir
    run_training(cfg, run_dir)

    _run_dir_cache[cache_key] = run_dir

    # 清理暫存 dataset
    if os.path.exists(cfg.data.dataset_dir):
        shutil.rmtree(cfg.data.dataset_dir)

    return run_dir