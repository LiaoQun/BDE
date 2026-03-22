# Project Requirement Document (PRD)

## 1. 專案目標 (Problem Statement)
將既有的化學鍵解離能 (BDE) 預測模型（原基於 Keras/TensorFlow 1.x 的 `pstjohn/bde_model_methods`）重構並移植到現代主流的 **PyTorch + PyTorch Geometric (PyG)** 架構。旨在解決舊版環境架設困難、依賴過時、維護不易的問題，並為後續研究提供一個乾淨、現代化的 Codebase。

## 2. 目標受眾 (Target Audience)
* 化學資訊學 (Cheminformatics) 研究人員
* 機器學習工程師
* 需要預測分子 BDE 的開發者（包括未來的維護者）

## 3. 成功標準 (Success Criteria)
* **架構還原度**：新模型需能接受 SMILES 字串，並產出與原版邏輯一致的圖結構特徵（特別是雙向邊的特徵處理）。
* **可訓練性**：建立完整的 PyTorch 訓練迴圈，Loss 能正常收斂。
* **再現性 (Reproducibility)**：在相同測試數據集上，新模型的 BDE 預測結果應與原論文基準誤差在可接受範圍內（MAE 接近原版水準）。
* **易用性**：代碼結構清晰，分為 `src/` (核心邏輯) 與 `run/` (執行腳本)，不依賴過時的 `nfp` 庫。

## 4. 範疇排除 (Out of Scope)
* **不進行模型架構改良**：本專案僅做「移植」，不引入 Transformer、Attention 機制或更換 GNN 骨幹。
* **不進行超參數優化**：沿用原作者設定的 Hidden Dimension (128)、層數等參數，不進行額外的 Grid Search。
* **不開發圖形介面 (GUI/Web)**：產出物僅包含 Python 訓練與推論腳本。

---
---

# 附錄：系統架構與使用說明

## A1. 總覽 (Overview)

本專案旨在建立一個圖神經網絡 (Graph Neural Network, GNN) 模型，用於預測分子的特定化學鍵的鍵解離能 (Bond Dissociation Energy, BDE)。

整套流程已經重構為一個以 `main.py` 為核心的單一入口框架。使用者僅需修改 `config.json` 檔案即可輕鬆調整參數並啟動訓練。

其核心工作流程包含三個主要階段：
1.  **參數設定**: 透過 `config.json` 檔案集中管理所有參數。
2.  **資料準備與訓練**: `main.py` 讀取設定檔，處理原始數據，並執行模型的訓練、驗證與儲存。
3.  **模型架構**: 模型本身定義在 `src/models/mpnn.py` 中，採用訊息傳遞機制來學習原子與化學鍵的特徵表示。

---

## A2. 參數設定 (`config.json` & `src/config.py`)

### 模塊功能概述

參數管理系統由兩個檔案協作完成：
-   `config.json`: 這是使用者面向的設定檔。所有實驗相關的超參數，如學習率、批次大小、資料路徑等，都應在此修改。
-   `src/config.py`: 這是程式碼層級的設定檔定義。它使用 Python 的 `dataclasses` 來定義一個強型別的設定結構 (`MainConfig`)，確保程式載入的參數有固定的結構與預設值。

### 細項說明

`config.json` 檔案主要分為三個區塊：

1.  **`data`**: 資料處理相關參數。
    -   `data_path`: 原始 `csv.gz` 資料檔案的路徑。
    -   `vocab_path`: 預處理詞彙表 (`preprocessor.json`) 的路徑。
    -   `sample_percentage`: 用於訓練的資料採樣比例 (例如 `0.01` 代表使用 1% 的資料)，方便快速測試。
    -   `test_size` / `val_size`: 測試集與驗證集的分割比例。

2.  **`model`**: 模型架構相關參數。
    -   `atom_features`: 模型中原子/化學鍵特徵向量的維度。
    -   `num_messages`: 訊息傳遞層 (`BDEInteractionLayer`) 的堆疊數量。

3.  **`train`**: 訓練過程相關參數。
    -   `epochs`: 訓練的總輪數。
    -   `lr`: 學習率。
    -   `batch_size`: 批次大小。
    -   `model_save_path`: 最佳模型的儲存路徑。

---

## A3. 主執行流程 (`main.py`)

### 模塊功能概述

`main.py` 是整個訓練、評估流程的唯一入口。它整合了所有步驟，從讀取設定、處理資料到執行訓練迴圈和最終評估，提供了一個清晰、可重複的執行環境。

### 細項說明

腳本的執行流程如下：

1.  **載入設定**:
    -   程式啟動時，會自動讀取根目錄下的 `config.json` 檔案。
    -   讀取到的 JSON 內容會被用來填充 `MainConfig` dataclass，若 `config.json` 中有缺漏的參數，則會自動使用 `src/config.py` 中定義的預設值。

2.  **資料準備與採樣**:
    -   使用 `pandas` 讀取 `data.data_path` 指定的原始 CSV 檔案。
    -   根據 `sample_percentage` 參數對資料庫中的**獨特分子**進行採樣，以產生一個較小的、用於快速測試的子集。
    -   呼叫 `prepare_data` 函式，將 DataFrame 整理成 `(SMILES, bde_labels_dict)` 的元組列表。

3.  **資料集建立**:
    -   初始化 `src.features.token.Tokenizer`，它負責將原子和鍵的類別映射為整數 ID。
    -   將前一步產生的資料傳遞給 `src.data.dataset.BDEDataset`。此類別會將每個 SMILES 字串轉換為 RDKit 分子物件，然後再轉換為 PyTorch Geometric 的 `Data` 圖物件，最終儲存為快取檔案以便快速讀取。

4.  **訓練迴圈**:
    -   模型 (`BDEModel`)、優化器 (`Adam`) 和資料載入器 (`DataLoader`) 被初始化。
    -   進入主迴圈，對每個 `epoch` 執行以下操作：
        -   **訓練 (Training)**:
            -   模型設定為 `train()` 模式。
            -   從 `train_loader` 中取出一個批次的圖資料。
            -   執行模型前向傳播 (`model(batch)`) 得到預測值。
            -   使用 L1 損失函數 (`F.l1_loss`) 計算預測值與真實標籤 (`batch.y`) 之間的平均絕對誤差 (MAE)。注意，損失只在 `batch.mask` 為 `True` 的化學鍵上計算。
            -   執行反向傳播與優化器步驟。
        -   **驗證 (Validation)**:
            -   模型設定為 `eval()` 模式。
            -   在驗證集 (`val_loader`) 上計算 MAE，此過程不計算梯度。

5.  **模型儲存**:
    -   在每個 `epoch` 結束後，如果當前的驗證損失比先前所有 `epoch` 的都低，則將當前模型權重儲存到 `train.model_save_path` 指定的檔案中。

6.  **最終測試**:
    -   訓練迴圈結束後，載入儲存下來的最佳模型。
    -   在測試集 (`test_loader`) 上執行最後一次評估，並輸出最終的測試 MAE 作為模型性能的最終指標。

---

## A4. 模型架構 (`src/models/mpnn.py`)

### 模塊功能概述

模型的核心實作位於 `src/models/mpnn.py`，主要由 `BDEModel` 和 `BDEInteractionLayer` 兩個類別組成。

-   **`BDEModel` (主模型)**: 這是完整的 GNN 模型容器。它負責初始化原子/鍵的嵌入向量，堆疊多個互動層，並在最後從化學鍵的最終特徵中預測 BDE 值。
-   **`BDEInteractionLayer` (核心互動層)**: 這是模型的核心組件，負責執行一輪訊息傳遞。它的主要特點是**同時更新原子和化學鍵的狀態**，模擬分子內化學環境的互動。

### 細項說明

#### `BDEInteractionLayer`

每一層的內部數據流如下：

1.  **輸入**: 來自上一層的原子狀態 `x` 和化學鍵狀態 `edge_attr`。
2.  **批次正規化**: 為了穩定訓練，在計算前會先對原子和鍵的狀態進行批次正規化 (`BatchNorm1d`)。
3.  **化學鍵狀態更新 (Edge Update)**:
    -   對於每一條化學鍵，模型會收集其相連的兩個原子（來源 `source_nodes`、目標 `target_nodes`）的狀態，以及這條鍵本身的狀態 `edge_attr`。
    -   將這三者的向量拼接後，輸入一個小型 MLP (`edge_mlp`)，生成一個「更新量」。
    -   使用殘差連接 (Residual Connection)，將這個更新量加回到原始的化學鍵狀態上，得到新的鍵狀態。
4.  **原子狀態更新 (Node Update)**:
    -   **訊息生成 (`message`)**: 對於每條邊，模型將其**更新後的鍵狀態**與其**來源原子的狀態**進行結合（在此為逐元素相乘），產生一條「訊息」。
    -   **訊息聚合 (`aggregate`)**: PyG 框架會將所有傳遞給同一個目標原子的訊息加總 (`aggr='sum'`)。
    -   **狀態更新 (`update`)**: 聚合後的訊息會通過另一個 MLP (`node_mlp`) 進行轉換，生成「原子狀態更新量」，並同樣透過殘差連接更新原子狀態。

#### `BDEModel`

整體的預測流程如下：

1.  **嵌入 (Embedding)**:
    -   輸入的原子 ID (`data.x`) 和化學鍵 ID (`data.edge_attr`) 被送入各自的 `nn.Embedding` 層，轉換為高維度的初始狀態向量 `atom_state` 和 `bond_state`。
    -   同時，模型會為每一種化學鍵類型學習一個特定的偏差 `bond_mean`。
2.  **訊息傳遞**:
    -   `atom_state` 和 `bond_state` 在 `interaction_layers` 中進行多輪 (`num_messages`) 迭代更新。
3.  **預測**:
    -   經過多輪更新後，富含上下文資訊的最終 `bond_state` 被送入一個線性層 (`output_mlp`)，預測出 BDE 純量值。
    -   將此預測值與該鍵類型對應的偏差 `bond_mean` 相加，得到最終的 BDE 預測。

---

## A5. 如何使用 (How to Use)

1.  **編輯設定檔**:
    -   打開根目錄下的 `config.json` 檔案。
    -   根據您的需求修改資料路徑、模型超參數或訓練設定。
2.  **執行訓練**:
    -   打開終端機，確認已啟動正確的 Conda 環境。
    -   執行以下命令：
    ```bash
    python main.py
    ```
    -   程式將自動載入 `config.json` 並開始訓練。若要指定不同的設定檔，可使用 `python main.py --config_path your_config.json`。
