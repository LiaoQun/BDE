import os
import pytest
from src.features.token import TokenFeaturizer

def test_token_featurizer_from_smiles_and_vocab(tmp_path):
    save_path = str(tmp_path / 'test_vocab.json')
    f = TokenFeaturizer.from_smiles(['CCO', 'CCC'], save_path=save_path)
    f2 = TokenFeaturizer.from_vocab(save_path)
    
    assert f.atom_dim == f2.atom_dim
    assert f.bond_dim == f2.bond_dim
