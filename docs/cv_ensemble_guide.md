# Nested CV + Ensemble 訓練管線配置指南

本指南詳細說明了專案中最新的「嵌套交叉驗證 (Nested CV) 與整合預測 (Ensemble)」的設計架構、資料流向以及如何在 `config.yaml` 中正確設置相關參數。

---

## 1. 架構概述 (High-level Concept)

我們的訓練管線現已升級，支援針對**不同的資料層級**進行交叉驗證與模型整合。這套管線具備以下三大特點：

1. **分層數據設計 (`base_data` vs `extra_data`)**
   - `base_data`（基礎資料）：模型每次訓練**必定包含**的龐大資料集（如 DB1），不參與 CV 切割。
   - `extra_data`（附加資料）：較小但精確的資料集（如 DB2、特定實驗數據），這是 CV **切割與驗證的對象**。

2. **Nested CV (嵌套交叉驗證) 架構**
   - **外迴圈 (Outer Loop)**：扮演「裁判」。將 `extra_data` 切成 K 份，每次留 1 份不參與該次折 (Fold) 訓練，作為純淨的測試使用。
   - **內迴圈 (Inner Loop)**：扮演「教練」。將剩餘的 `broad_train` (包含了全量 `base_data` + K-1 的 `extra_data`) 以比例留出驗證集 (Validation Set)，用來監控與觸發 Early Stopping (提前終止)。

3. **整合預測 (Ensemble Inference)**
   - 當 K 份 Fold 皆訓練完畢後，系統會拿這 K 個獨立的模型，對**完整的 `extra_data`** 進行預測。
   - 所有模型的預測結果會進行平均 (Mean) 與計算不確定性 (Standard Deviation)，以提供更穩健的預測結果。

---

## 2. Config 設置指南

所有的資料分配策略都集中在 YAML 配置檔的 `data` 區塊之下。以下是 `configs/experiments/cv_ensemble.yaml` 的標準範例：

```yaml
data:
  # 基礎資料集：永遠參與訓練，不會被當作 Outer Test
  base_data_paths:
    - examples/test_data_db2_1000.csv

  # 附加資料集：進行 K-Fold 切割的對象
  extra_data_paths:      
    - examples/test_db2_bdfe.csv

  # Multi-task 設定：如果不同資料檔缺失特定的目標值，系統會自動補 NaN 並參與訓練
  target_columns:
    - bde
    - bdfe

  # ── 外迴圈 (Outer CV) ──
  # 支援三種模式：
  #   - 'none'           : 不啟動 CV。全部資料混在一起做單次訓練 (預設值)。
  #   - 'leave_one_out'  : 留一法 (對 extra_data 操作，較耗時)。
  #   - 數字 (例如 3 或 5) : 執行 K-Fold CV (如 3 折代表 K=3)。
  cross_validation: 3         

  # ── 內迴圈 (Inner Validation) ──
  # 從「廣義訓練集」中切出多少比例用於監控 Early Stopping。
  #   - 0.0 : Method-A，不會有 Early Stopping，每個 Fold 會硬跑完所有 Epoches。
  #   - 0.1 : Method-B，留出 10% 作為 Val Set，動態監控 Loss。
  val_size: 0.1               
  
train:
  epochs: 10
  early_stopping_patience: 5  # 當 val_size > 0.0 時生效
```

> **注意：** 先前的 `sample_percentage` 參數已全面廢棄，請保持資料原始大小，或自行準備子集資料集。

---

## 3. 輸出結構 (Directory Structure)

訓練結束後，所有的產出物會井然有序地存放在 timestamp 資料夾中，每個 Fold 會有自己的子資料夾，以維持乾淨的檔案結構。

```text
training_runs/20260407_123456/
├── config.yaml                    # 本次執行的完整打平訓練設定
├── vocab.json                     # Featurizer 的詞彙表 (SMILES 聯集產生)
├── fold_metrics.csv               # 記錄每個 Fold 分別在各自的 Outer_test 上的 MAE / RMSE / R2
├── ensemble_predictions.csv       # 最終 Ensemble 產出的預測結果 (包含標註 _mean 與 _std)
├── parity_ensemble.png            # 帶有預測誤差棒 (Error bars) 的 Parity Plot 圖表
├── training.log                   # 訓練過程全域日誌
│
├── fold_0/
│   ├── bde_model.pt               # 第一折的 PyTorch 模型權重
│   ├── training_log.csv           # 第一折的逐 Epoch Loss 紀錄
│   └── training_curve.png         # 第一折的 Loss 曲線圖
├── fold_1/
│   ├── bde_model.pt               
│   ...
└── fold_2/
    ├── bde_model.pt               
    ...
```

### 多任務 (Multi-Task) 的特別行為說明
在 Ensemble 的 Parity Plot 與 `fold_metrics.csv` 中，系統會自動迭代所有的 `target_columns`：
- 單一視窗內會並排繪製多張 `bde`、`bdfe` 的驗證圖表。
- 你可以直接檢視 `bde_pred_mean` 與 `bdfe_pred_mean` 兩個欄位的預測，系統會自動略過缺失地面真實標籤 (Ground-truth NaN) 的資料，正確計算各 Task 的統計指標。
