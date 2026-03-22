# Technology Stack

本專案採用目前 Graph Learning 領域最穩定且生態系最完整的技術組合。

## Core Framework
* **Python 3.8+**: 確保最佳的套件相容性與長期支援。
* **PyTorch (Latest Stable)**: 深度學習核心框架。動態圖計算特性方便除錯，且擁有強大的社群資源。

## Graph Neural Networks
* **PyTorch Geometric (PyG)**: 
    * 目前 PyTorch 生態中最強大的 GNN 庫。
    * 封裝了高效的稀疏矩陣運算 (`SparseTensor`)。
    * 內建 `MessagePassing` 基類，能輕鬆實作客製化的 Edge/Node 更新邏輯。
    * 自動處理圖數據的 Batching (Mini-batching)。

## Cheminformatics
* **RDKit**: 
    * 化學資訊處理的工業標準。
    * 用於解析 SMILES、生成分子物件、提取原子與化學鍵特徵 (Atom/Bond Features)。

## Data Processing & Utilities
* **Pandas**: 處理 CSV 格式的 BDE 數據集。
* **Numpy**: 數值計算與矩陣操作。
* **Tqdm**: 提供訓練與數據處理的進度條顯示。
