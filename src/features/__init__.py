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