# Implementation Plan: Enhancements for BDE Prediction Project
規則強調：
✓ 模組化、多檔案
✓ 禁止單一巨型檔案
✓ 一步一驗收
✓ 不准自己擴大 scope
✓ **程式碼需包含 Tensor 維度註解**
✓ 必須在完成一個步驟後更新 progress.md 文件
✓ 請將重構的檔案更新在獨立的資料夾中，並且全部重寫不要import舊的alfabet或是bde_model_methods檔案

## Overview
This plan outlines a series of enhancements to improve the project's code quality, maintainability, and data handling for large-scale applications. The modifications focus on logging, featurizer data preparation, and (deferred) flexible featurizer specification.

## Phase 1: Immediate Enhancements (Current Task)

### 1. Replace `print` Statements with `logging` Module

**Rationale:** The standard Python `logging` module provides a more robust and flexible way to handle application messages compared to `print()`. It allows for message categorization (DEBUG, INFO, WARNING, ERROR), redirection to various outputs (console, file), and structured message formatting (timestamps, file/line information), which is crucial for debugging, monitoring, and maintaining production-grade applications.

**Action Plan:**
*   **Global Configuration:** Set up a basic `logging` configuration in `main.py` to ensure all modules inherit a consistent logging behavior (e.g., output format, minimum level, console output).
*   **Module-level Loggers:** In each affected Python module (`main.py`, `src/inference/predictor.py`, `scripts/predict.py`, `scripts/visualize_embeddings.py`, `src/data/dataset.py`, `src/curation/template_generator.py`), instantiate a module-specific logger using `logging.getLogger(__name__)`.
*   **Replace Calls:** Systematically replace all `print()` calls with appropriate `logger` methods (`logger.info()`, `logger.warning()`, `logger.error()`, etc.) based on the severity and purpose of the original `print` message.
*   **Error Handling:** Ensure `try...except` blocks use `logger.error()` and capture traceback information (`exc_info=True`) for better debugging.
*   **`tqdm` Integration:** Review `tqdm` usage to ensure it doesn't interfere with logging output or is properly configured to output to a specific stream if desired.

**Affected Files:**
*   `main.py`
*   `src/inference/predictor.py`
*   `scripts/predict.py`
*   `scripts/visualize_embeddings.py`
*   `src/data/dataset.py` (specifically `SMILESAugmenter` and `BDEDataset` `process` method)
*   `src/curation/template_generator.py`

### 2. Ensure `featurizer.prepare_data` Uses Correct Training Data

**Rationale:** The `featurizer.prepare_data()` method (e.g., in `TokenFeaturizer`) is responsible for learning a vocabulary or specific configurations from a dataset. It is critical that this method learns *only* from the *original, unaugmented* training data to prevent data leakage from augmented samples or validation/test sets, which could lead to an overly optimistic evaluation of model performance.

**Action Plan:**
*   **Identify Call Site:** Locate the call to `featurizer.prepare_data()` within `main.py`.
*   **Adjust Data Source:** Modify the list of SMILES passed to `featurizer.prepare_data()` to ensure it exclusively comprises the SMILES strings from the *original, unaugmented training set*. This means using the list of SMILES *before* any augmentation has been applied to `train_smiles_data`.

**Affected Files:**
*   `main.py`

## Phase 2: Deferred Enhancements

### 3. Support Separate Atom/Bond Featurizer Specification in `BDEDataset`

**Rationale:** To provide greater flexibility and modularity in defining how atom and bond features are generated, it is desirable to allow separate specification of featurizers or featurizer configurations for atoms and bonds. This enables experimentation with heterogeneous feature engineering strategies.

**Action Plan (Conceptual - to be detailed later):**
*   **`DataConfig` Update:** Add dedicated configuration parameters (e.g., `atom_featurizer_config`, `bond_featurizer_config`) to `src/config.py`.
*   **`get_featurizer` Refinement:** Enhance the `src/features/__init__.py` factory function (`get_featurizer`) to interpret these granular configurations and construct a composite featurizer (or a single featurizer that internally uses separate atom/bond logic) based on the specified types and parameters.
*   **`BaseFeaturizer` / `Featurizer` Implementations:** Potentially refactor existing featurizer classes or introduce new ones (`BaseAtomFeaturizer`, `BaseBondFeaturizer`) if a deep separation is desired.

**Affected Files:**
*   `src/config.py`
*   `src/features/__init__.py`
*   `src/features/base.py` (potential)
*   Existing featurizer implementations (e.g., `src/features/featurizer.py`, `src/features/chemprop.py`)

## Conclusion
These phased enhancements aim to incrementally improve the project's robustness, flexibility, and adherence to best practices in machine learning and software engineering. The current task focuses on immediate, high-impact changes related to logging and data integrity.
