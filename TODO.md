# TODO List - 專案改進計畫

## 數據處理與擴充 (Data Processing & Extensions)
- [x] **數據預處理**: 在進行任何特徵提取前，應先將 SMILES 字符串標準化 (Canonicalization)。
- [ ] **支援超大型數據集與高效加載**:
    - 轉換 `BDEDataset` 為標準 PyG `Dataset`，以支持處理超出記憶體容量的數據集。
    - 在實現的同時，**必須** 配合 `DataLoader` 的 `num_workers` 參數進行優化，以隱藏磁碟 I/O 延遲，確保高效數據加載。
- [ ] **數據增強 (Data Augmentation)**: 導入如 SMILES 隨機化等策略，以提升模型的泛化能力。
- [X] generate_template optimize: frags_smiles_list = sorted(fragmented_smiles.split("."), key=len,reverse=True)

## 實驗管理與 ML Ops (Experiment Management & ML Ops)
- [x] **儲存完整的預測結果**: 在評估階段，應將訓練、驗證、測試集的完整預測結果（包含原始 SMILES 和碎片資訊）儲存下來。
- [x] **儲存運行配置**: 為確保可重現性，在每次訓練運行時，應將當次使用的 `config.json` 複製並保存到該次運行的目錄中。
- [ ] **集成超參數優化 (HPO)**: 引入如 Optuna 或 Ray Tune 等框架，以自動化探索最佳超參數。
- [ ] **集成實驗追蹤平台**: 引入如 MLflow, TensorBoard, 或 W&B 等工具，以系統化地管理、可視化和比較所有實驗結果。
- [ ] **實施模型版本管理**: 建立更完善的模型版本控制機制，確保訓練過程的可追溯性。

## 模型與特徵工程 (Model & Featurization)
- [X] **模塊化 Featurizer**: 將特徵提取器抽象化，使其可通過配置文件輕鬆替換或組合。
    - [X] **Ref:** https://github.com/chemprop/chemprop/blob/main/chemprop/featurizers/atom.py
- [ ] **支持多種 GNN 骨幹**: 透過配置，允許動態選擇除 MPNN 之外的其他 GNN 模型架構。

## 性能優化 (Performance Optimization)
- [ ] **整合大規模計算優化**: 整合更多針對大規模計算的性能優化策略，例如更高效的批次處理（Batching）。
- [ ] **集成 `torch.compile`**: 利用 PyTorch 2.x 的優化功能來加速模型訓練和推斷。
- [ ] **實現多 GPU 訓練**: 考慮加入多 GPU 訓練功能，以加速在大型模型或數據集上的訓練過程。

## 推論與腳本 (Inference & Scripts)
- [x] **完善 BDE 預測器**: 實現或修復 BDE 預測器的功能，確保能正確處理 SMILES 列表輸入、去重、以及潛在的 `fragment_bond_indices` 問題。(已增加 `is_valid` 欄位功能)
- [x] **優化 `create_training_template.py`**: 引入命令行參數，使其能從命令行接收 SMILES 輸入並指定輸出路徑。
- [x] **優化 `predict.py`**:
    - [x] 簡化模型加載邏輯，允許通過 `--run_dir` 參數直接指向訓練運行目錄。

## 程式碼重構與清理 (Code Refactoring & Cleanup)
- [x] **移除重複的 `evaluate` 函數**: 在代碼中存在兩個 `evaluate` 函數，需要移除其中一個以保持代碼整潔。
- [x] **重構 `mol_to_graph` 轉換邏輯**: 將 `src/data/dataset.py` 和 `src/inference/predictor.py` 中重複的分子到圖轉換邏輯，提取到 `src/features/featurizer.py` 中的共用函數 `mol_to_graph`，並讓兩邊的模塊都調用此函數，以遵循 DRY 原則。
## 專案架構重構 (Project Structure Refactoring) - 已完成
- [x] 刪除廢棄的 `src/features/chemprop.py`（舊版）
- [x] `main.py` 改名為 `train.py`
- [x] `src/data_preparation/` 改名為 `src/curation/`
- [x] 合併 `src/features/` 模組（4個檔案 → 3個）
- [x] 修復所有測試至新 API
- [x] 新增 `configs/` 目錄（YAML + `_base_` 繼承 + 訓練時自動展開儲存）
- [x] `src/config.py` 拆分為 `src/config/schema.py` + `src/config/loader.py`

## 測試與品質保證 (Testing & Quality Assurance)
- [X] **擴充單元測試覆蓋率**: 為核心的數據處理、特徵提取和模型組件編寫更全面的單元測試。
- [ ] **建立整合測試**: 建立一個小型的端到端測試，自動運行一個完整的訓練和預測流程，以確保所有模塊能正確協同工作。
- [ ] **導入靜態分析與 Linter**: 引入如 `ruff`, `mypy` 等工具，並考慮整合到 pre-commit hook 中，以自動化程式碼風格檢查和型別檢查，確保代碼品質。
