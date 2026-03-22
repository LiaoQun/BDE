# Featurizer 重構指南

## 目標

將 `src/features/` 模組的 Featurizer 初始化方式，從「兩步驟 instance method」改為「單步驟 classmethod」模式，統一訓練與推論端的初始化路徑。

## 執行順序

> **重要**：請嚴格按照步驟順序執行，每個步驟完成後再進行下一步。

1. `src/features/base.py`
2. `src/features/featurizer.py`
3. `src/features/chem_prop_featurizer.py`
4. `src/features/chemprop.py`
5. `src/features/__init__.py`
6. `main.py`
7. `src/inference/predictor.py`

---

## Step 1：重構 `src/features/base.py`

### 目標
重新定義 `BaseFeaturizer` 的 abstract interface。

### 變更內容

**移除以下 abstract methods：**
- `prepare_data()`
- `save()`
- `load()`

**新增以下 abstract classmethods：**
- `from_smiles(cls, smiles_list, save_path)` — 從訓練資料建構，並直接序列化
- `from_vocab(cls, vocab_path)` — 從序列化檔案重建

**不要動：**
- `featurize()` abstract method
- `atom_dim`、`bond_dim`、`is_discrete` abstract properties

### After

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from rdkit import Chem
from torch_geometric.data import Data


class BaseFeaturizer(ABC):

    @property
    @abstractmethod
    def atom_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def bond_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def is_discrete(self) -> bool:
        pass

    @classmethod
    @abstractmethod
    def from_smiles(cls, smiles_list: List[str], save_path: str) -> "BaseFeaturizer":
        """
        從訓練資料建構 Featurizer，並將狀態序列化到 save_path。
        訓練端使用此方法。
        """
        pass

    @classmethod
    @abstractmethod
    def from_vocab(cls, vocab_path: str) -> "BaseFeaturizer":
        """
        從序列化檔案重建 Featurizer。
        推論端使用此方法。
        """
        pass

    @abstractmethod
    def featurize(self, mol: Chem.Mol,
                  labels: Optional[Dict[Tuple[int, int], float]] = None,
                  smiles: str = "") -> Optional[Data]:
        pass
```

### 驗收條件
- `base.py` 不再有 `prepare_data`、`save`、`load` 的定義
- `from_smiles` 和 `from_vocab` 為 abstract classmethod

---

## Step 2：重構 `src/features/featurizer.py`

### 目標
`TokenFeaturizer` 改用 classmethod 模式，移除對 `prepare_data()` 和 `save()` / `load()` 的依賴。

### 變更內容

**`Tokenizer` class：不需要修改。**

**`TokenFeaturizer` class：**

移除：
- `__init__(self, vocab_filepath)` 中載入 vocab 的邏輯
- `prepare_data()` method
- `save()` method
- `load()` method

新增：
- `__init__(self, tokenizer: Tokenizer)` — 只接收已建構好的 tokenizer
- `from_smiles(cls, smiles_list, save_path)` classmethod
- `from_vocab(cls, vocab_path)` classmethod

**不要動：**
- `Tokenizer` class 的所有邏輯
- `featurize()` method
- `atom_dim`、`bond_dim`、`is_discrete` properties
- 檔案底部的 `mol_to_graph()` wrapper function

### After（只列出 `TokenFeaturizer` class，`Tokenizer` 維持原樣）

```python
class TokenFeaturizer(BaseFeaturizer):

    def __init__(self, tokenizer: Tokenizer):
        # __init__ 只負責接收已建構好的 tokenizer，不做任何 I/O
        self.tokenizer = tokenizer

    @classmethod
    def from_smiles(cls, smiles_list: List[str], save_path: str) -> "TokenFeaturizer":
        """從訓練資料建構詞彙表，並序列化到 save_path。"""
        tokenizer = Tokenizer()
        tokenizer.build_from_smiles(smiles_list)
        tokenizer.save(save_path)
        return cls(tokenizer)

    @classmethod
    def from_vocab(cls, vocab_path: str) -> "TokenFeaturizer":
        """從已存在的詞彙表檔案載入。"""
        tokenizer = Tokenizer(vocab_filepath=vocab_path)
        return cls(tokenizer)

    @property
    def atom_dim(self) -> int:
        return self.tokenizer.atom_num_classes + 1

    @property
    def bond_dim(self) -> int:
        return self.tokenizer.bond_num_classes + 1

    @property
    def is_discrete(self) -> bool:
        return True

    def featurize(self, mol, labels=None, smiles=""):
        # 維持原有邏輯，不需要修改
        ...
```

### 驗收條件
- `TokenFeaturizer.__init__` 只接受一個 `Tokenizer` 參數
- `TokenFeaturizer.from_smiles(smiles_list, save_path)` 可正確建構並儲存
- `TokenFeaturizer.from_vocab(vocab_path)` 可正確載入

---

## Step 3：重構 `src/features/chem_prop_featurizer.py`

### 目標
`MultiHotAtomFeaturizer` 新增從資料動態建構的 classmethod，所有參數皆從訓練資料掃描決定。

### 變更內容

**`MultiHotAtomFeaturizer` class：**

新增：
- `from_smiles(cls, smiles_list)` classmethod — 掃描 smiles_list，動態收集所有出現的 atomic_nums、degrees、hybridizations、formal_charges、chiral_tags、num_Hs，然後回傳一個新的 `MultiHotAtomFeaturizer` instance

**不要動：**
- `__init__`、`__len__`、`__call__` 的現有邏輯
- `v1()`、`v2()`、`organic()` classmethods（保留向後相容）
- `MultiHotBondFeaturizer` class（完整保留，不需修改）

### After（只列出新增的 classmethod）

```python
@classmethod
def from_smiles(cls, smiles_list: List[str]) -> "MultiHotAtomFeaturizer":
    """
    掃描 smiles_list，動態收集所有出現的原子特性，建構 Featurizer。
    OOV 行為：特徵向量為全零（不加入 unknown token）。
    """
    from rdkit import Chem
    from rdkit.Chem.rdchem import HybridizationType

    atomic_nums = set()
    degrees = set()
    hybridizations = set()
    formal_charges = set()
    chiral_tags = set()
    num_Hs = set()

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        for atom in mol.GetAtoms():
            atomic_nums.add(atom.GetAtomicNum())
            degrees.add(atom.GetTotalDegree())
            hybridizations.add(atom.GetHybridization())
            formal_charges.add(atom.GetFormalCharge())
            chiral_tags.add(int(atom.GetChiralTag()))
            num_Hs.add(int(atom.GetTotalNumHs()))

    return cls(
        atomic_nums=sorted(atomic_nums),
        degrees=sorted(degrees),
        formal_charges=sorted(formal_charges),
        chiral_tags=sorted(chiral_tags),
        num_Hs=sorted(num_Hs),
        hybridizations=sorted(hybridizations, key=lambda x: x.real),
    )
```

### 驗收條件
- `MultiHotAtomFeaturizer.from_smiles(smiles_list)` 可正確執行並回傳 instance
- 回傳的 instance 的 `__len__` 應反映掃描到的實際類別數量

---

## Step 4：重構 `src/features/chemprop.py`

### 目標
`ChemPropPyGFeaturizer` 改用 classmethod 模式，支援從資料動態建構與從檔案重建。

### 變更內容

移除：
- `__init__(self, atom_featurizer_mode: str)` 中依賴 mode string 的邏輯

新增：
- `__init__(self, atom_featurizer, bond_featurizer)` — 只接收已建構好的 featurizer objects
- `from_smiles(cls, smiles_list, save_path)` classmethod
- `from_vocab(cls, vocab_path)` classmethod
- `_save(self, save_path)` private method — 序列化掃描出的參數到 JSON

**不要動：**
- `featurize()` method
- `atom_dim`、`bond_dim`、`is_discrete` properties

### 序列化格式（儲存到 `save_path` 的 JSON 結構）

```json
{
    "featurizer_type": "ChemPropFeaturizer",
    "atomic_nums": [1, 6, 7, 8],
    "degrees": [0, 1, 2, 3, 4],
    "hybridizations": [2, 3, 4],
    "formal_charges": [-1, 0, 1],
    "chiral_tags": [0, 1, 2, 3],
    "num_Hs": [0, 1, 2, 3, 4]
}
```

> 注意：`hybridizations` 儲存為 int（`HybridizationType` 的 `.real` 值），載入時需轉回 enum。

### After

```python
class ChemPropPyGFeaturizer(BaseFeaturizer):

    def __init__(self, atom_featurizer: MultiHotAtomFeaturizer, bond_featurizer: MultiHotBondFeaturizer):
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer

    @classmethod
    def from_smiles(cls, smiles_list: List[str], save_path: str) -> "ChemPropPyGFeaturizer":
        """從訓練資料動態建構，並序列化到 save_path。"""
        atom_feat = MultiHotAtomFeaturizer.from_smiles(smiles_list)
        bond_feat = MultiHotBondFeaturizer()
        instance = cls(atom_feat, bond_feat)
        instance._save(save_path)
        return instance

    @classmethod
    def from_vocab(cls, vocab_path: str) -> "ChemPropPyGFeaturizer":
        """從序列化檔案重建，保證與訓練時行為一致。"""
        import json
        from rdkit.Chem.rdchem import HybridizationType

        with open(vocab_path, 'r') as f:
            data = json.load(f)

        # 將 int 還原為 HybridizationType enum
        hybridization_map = {h.real: h for h in HybridizationType.values.values()}
        hybridizations = [hybridization_map[v] for v in data["hybridizations"]]

        atom_feat = MultiHotAtomFeaturizer(
            atomic_nums=data["atomic_nums"],
            degrees=data["degrees"],
            formal_charges=data["formal_charges"],
            chiral_tags=data["chiral_tags"],
            num_Hs=data["num_Hs"],
            hybridizations=hybridizations,
        )
        return cls(atom_feat, MultiHotBondFeaturizer())

    def _save(self, save_path: str):
        import json, os
        from rdkit.Chem.rdchem import HybridizationType

        atom_feat = self.atom_featurizer
        # 從 atom_featurizer 反推出原始參數列表
        data = {
            "featurizer_type": "ChemPropFeaturizer",
            "atomic_nums": sorted(atom_feat.atomic_nums.keys()),
            "degrees": sorted(atom_feat.degrees.keys()),
            "formal_charges": sorted(
                atom_feat.formal_charges.keys(),
                key=lambda x: list(atom_feat.formal_charges.keys()).index(x)
            ),
            "chiral_tags": sorted(atom_feat.chiral_tags.keys()),
            "num_Hs": sorted(atom_feat.num_Hs.keys()),
            "hybridizations": [h.real for h in atom_feat.hybridizations.keys()],
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=4)

    @property
    def atom_dim(self) -> int:
        return len(self.atom_featurizer)

    @property
    def bond_dim(self) -> int:
        return len(self.bond_featurizer)

    @property
    def is_discrete(self) -> bool:
        return False

    def featurize(self, mol, labels=None, smiles=""):
        # 維持原有邏輯，不需要修改
        ...
```

### 驗收條件
- `ChemPropPyGFeaturizer.from_smiles(smiles_list, save_path)` 執行後，`save_path` 應存在且為合法 JSON
- `ChemPropPyGFeaturizer.from_vocab(save_path)` 重建後，`atom_dim` 和 `bond_dim` 應與原 instance 相同

---

## Step 5：重構 `src/features/__init__.py`

### 目標
工廠函數拆成兩個，分別對應訓練與推論。

### After

```python
from typing import List
from src.features.base import BaseFeaturizer


def get_featurizer(featurizer_type: str, smiles_list: List[str], save_path: str) -> BaseFeaturizer:
    """
    訓練用工廠函數。從訓練資料建構 Featurizer 並序列化。

    Args:
        featurizer_type: 'TokenFeaturizer' 或 'ChemPropFeaturizer'
        smiles_list: 訓練集的 SMILES list（只傳原始訓練集，不含驗證/測試集）
        save_path: 序列化檔案的儲存路徑
    """
    if featurizer_type == 'TokenFeaturizer':
        from src.features.token import TokenFeaturizer
        return TokenFeaturizer.from_smiles(smiles_list, save_path)

    elif featurizer_type == 'ChemPropFeaturizer':
        from src.features.chemprop import ChemPropPyGFeaturizer
        return ChemPropPyGFeaturizer.from_smiles(smiles_list, save_path)

    else:
        raise ValueError(f"Unknown featurizer type: {featurizer_type}")


def get_featurizer_from_vocab(featurizer_type: str, vocab_path: str) -> BaseFeaturizer:
    """
    推論用工廠函數。從序列化檔案重建 Featurizer。

    Args:
        featurizer_type: 'TokenFeaturizer' 或 'ChemPropFeaturizer'
        vocab_path: 序列化檔案的路徑（由訓練時的 get_featurizer 產生）
    """
    if featurizer_type == 'TokenFeaturizer':
        from src.features.token import TokenFeaturizer
        return TokenFeaturizer.from_vocab(vocab_path)

    elif featurizer_type == 'ChemPropFeaturizer':
        from src.features.chemprop import ChemPropPyGFeaturizer
        return ChemPropPyGFeaturizer.from_vocab(vocab_path)

    else:
        raise ValueError(f"Unknown featurizer type: {featurizer_type}")
```

### 驗收條件
- 舊的 `get_featurizer(config: DataConfig)` 簽名完全移除
- 兩個新函數可正確 dispatch 到對應的 classmethod

---

## Step 6：修改 `main.py`

### 目標
訓練端改用新的 `get_featurizer` 工廠函數。

### 變更內容

**Import 修改：**
```python
# Before
from src.features import get_featurizer

# After
from src.features import get_featurizer, get_featurizer_from_vocab
```

**初始化邏輯修改：**

找到以下這段程式碼（位於 `run_training` 函數內），**完整替換**：

```python
# Before（以下三個區塊全部替換）
featurizer = get_featurizer(cfg.data)

if hasattr(featurizer, 'prepare_data'):
    logger.info("Preparing featurizer (e.g., building vocabulary)...")
    train_smiles = [data[0] for data in train_smiles_data]
    featurizer.prepare_data(train_smiles)

vocab_save_path = os.path.join(run_dir, "vocab.json")
featurizer.save(vocab_save_path)
logger.info(f"Featurizer state saved to: {vocab_save_path}")
effective_vocab_path = vocab_save_path
```

```python
# After
vocab_save_path = os.path.join(run_dir, "vocab.json")
train_smiles = [data[0] for data in train_smiles_data]

logger.info(f"Initializing featurizer: {cfg.data.featurizer_type}...")
featurizer = get_featurizer(
    featurizer_type=cfg.data.featurizer_type,
    smiles_list=train_smiles,
    save_path=vocab_save_path
)
logger.info(f"Featurizer built and saved to: {vocab_save_path}")
effective_vocab_path = vocab_save_path
```

**不要動：**
- `cfg.data.featurizer_type` 的讀取邏輯
- `Trainer` 的初始化與後續邏輯

### 驗收條件
- `run_training` 函數內不再有 `hasattr(featurizer, 'prepare_data')` 的檢查
- `featurizer.save()` 的呼叫不再存在

---

## Step 7：修改 `src/inference/predictor.py`

### 目標
推論端改用 `get_featurizer_from_vocab`，移除對 `DataConfig` 的依賴。

### 變更內容

**Import 修改：**
```python
# Before
from src.features import get_featurizer
from src.config import DataConfig

# After
from src.features import get_featurizer_from_vocab
```

**`Predictor.__init__` 修改：**

找到以下程式碼，**替換**：

```python
# Before
data_config = DataConfig(vocab_path=vocab_path, featurizer_type=featurizer_type)
self.featurizer = get_featurizer(data_config)
```

```python
# After
self.featurizer = get_featurizer_from_vocab(
    featurizer_type=featurizer_type,
    vocab_path=vocab_path
)
```

**不要動：**
- `Predictor` 的其他所有邏輯
- `get_bde_predictions()` 和 `get_bde_predictions_with_embeddings()` 函數

### 驗收條件
- `Predictor.__init__` 不再建立 `DataConfig` object
- `DataConfig` 的 import 可從 `predictor.py` 移除（如果其他地方沒有用到）

---

## 全域驗收條件

完成所有步驟後，執行以下確認：

```bash
# 1. 確認現有測試仍可通過（部分測試可能需要同步更新）
pytest tests/test_featurizer.py
pytest tests/test_dataset.py
pytest tests/test_model.py

# 2. 確認 TokenFeaturizer 訓練與推論路徑
python -c "
from src.features.token import TokenFeaturizer
f = TokenFeaturizer.from_smiles(['CCO', 'CCC'], save_path='/tmp/test_vocab.json')
f2 = TokenFeaturizer.from_vocab('/tmp/test_vocab.json')
assert f.atom_dim == f2.atom_dim
assert f.bond_dim == f2.bond_dim
print('TokenFeaturizer OK')
"

# 3. 確認 ChemPropFeaturizer 訓練與推論路徑
python -c "
from src.features.chemprop import ChemPropPyGFeaturizer
f = ChemPropPyGFeaturizer.from_smiles(['CCO', 'CCC'], save_path='/tmp/test_chemprop.json')
f2 = ChemPropPyGFeaturizer.from_vocab('/tmp/test_chemprop.json')
assert f.atom_dim == f2.atom_dim
assert f.bond_dim == f2.bond_dim
print('ChemPropFeaturizer OK')
"
```

---

## 注意事項

- `tests/` 目錄下的測試檔案可能需要同步更新（fixture 的建構方式改變），**但不在本次重構範圍內**，請另行處理。
- `scripts/predict.py` 和 `scripts/visualize_embeddings.py` 不需要修改，因為它們透過 `Predictor` 間接使用 featurizer。
- `src/config.py` 的 `DataConfig` 不需要修改，`vocab_path` 和 `featurizer_type` 欄位仍被 `main.py` 使用來讀取設定值。
