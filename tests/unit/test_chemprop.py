import os
import pytest
from src.features.chemprop import ChemPropPyGFeaturizer

def test_chemprop_featurizer_from_smiles_and_vocab(tmp_path):
    save_path = str(tmp_path / 'test_chemprop.json')
    f = ChemPropPyGFeaturizer.from_smiles(['CCO', 'CCC'], save_path=save_path)
    f2 = ChemPropPyGFeaturizer.from_vocab(save_path)
    
    assert f.atom_dim == f2.atom_dim
    assert f.bond_dim == f2.bond_dim
