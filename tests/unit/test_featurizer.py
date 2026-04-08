import json
import os
import tempfile
import pytest
from rdkit import Chem
import torch # Import torch for dtype checks
from torch_geometric.data import Data

# Adjust the path to import from the new 'src' directory
from src.config import DataConfig
from src.features.token import (
    atom_featurizer,
    bond_featurizer,
    Tokenizer,
    TokenFeaturizer # Now testing TokenFeaturizer class directly
)
from src.features.chemprop import ChemPropPyGFeaturizer
from src.features import get_featurizer # Import the factory function

@pytest.fixture
def sample_mol():
    """Provides an ethanol molecule (CCO) for testing."""
    mol = Chem.MolFromSmiles("CCO")
    return Chem.AddHs(mol)

@pytest.fixture
def sample_vocab_file():
    """Creates a temporary vocabulary JSON file for testing."""
    vocab_data = {
        "atom_tokenizer": {
            "_data": {
                "unk": 1,
                "('C', 0, 0, rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED, False, 0, 1, 3)": 2, # Example C from CCO
                "('O', 0, 0, rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED, False, 0, 1, 1)": 3, # Example O from CCO
            },
            "num_classes": 3
        },
        "bond_tokenizer": {
            "_data": {
                "unk": 1,
                "C-C (rdkit.Chem.rdchem.BondType.SINGLE, False)": 2,
                "C-O (rdkit.Chem.rdchem.BondType.SINGLE, False)": 3,
                "O-C (rdkit.Chem.rdchem.BondType.SINGLE, False)": 4 # Flipped bond
            },
            "num_classes": 4
        }
    }
    # Use a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as f:
        json.dump(vocab_data, f)
        filepath = f.name
    
    yield filepath
    
    # Cleanup the file
    os.remove(filepath)

def test_atom_featurizer(sample_mol):
    """Tests the atom featurizer function."""
    # Test a carbon atom
    c0 = sample_mol.GetAtomWithIdx(0) # First C in CCO
    c0_features = atom_featurizer(c0)
    assert "'C'" in c0_features
    # The actual string is complex and can vary slightly with RDKit versions/internal details.
    # Testing for symbol presence is often sufficient for these string-based featurizers.

def test_bond_featurizer(sample_mol):
    """Tests the bond featurizer for both normal and flipped directions."""
    # C-C bond (index 0)
    bond_cc = sample_mol.GetBondWithIdx(0)
    
    feat_normal_cc = bond_featurizer(bond_cc, flipped=False)
    assert feat_normal_cc.startswith("C-C")
    assert "SINGLE, False" in feat_normal_cc

    feat_flipped_cc = bond_featurizer(bond_cc, flipped=True)
    assert feat_flipped_cc.startswith("C-C") # Still C-C for this bond
    assert "SINGLE, False" in feat_flipped_cc
    
    # C-O bond (index 1)
    bond_co = sample_mol.GetBondWithIdx(1) # C2-O3 bond
    feat_normal_co = bond_featurizer(bond_co, flipped=False)
    assert feat_normal_co.startswith("C-O")
    
    feat_flipped_co = bond_featurizer(bond_co, flipped=True)
    assert feat_flipped_co.startswith("O-C")
    assert feat_normal_co != feat_flipped_co # Flipped should be different for heteroatoms

def test_tokenizer_init_empty():
    """Tests tokenizer initialization without a vocab file."""
    tokenizer = Tokenizer()
    assert tokenizer.tokenize_atom("any_feature") == 1 # 'unk' is 1
    assert tokenizer.tokenize_bond("any_feature") == 1
    assert tokenizer.atom_num_classes == 1
    assert tokenizer.bond_num_classes == 1

def test_tokenizer_load_from_json(sample_vocab_file):
    """Tests tokenizer initialization from a predefined vocab file."""
    tokenizer = Tokenizer(vocab_filepath=sample_vocab_file)
    
    # Test known atom from fixture
    known_atom_feat = "('C', 0, 0, rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED, False, 0, 1, 3)"
    assert tokenizer.tokenize_atom(known_atom_feat) == 2
    
    # Test unknown atom
    assert tokenizer.tokenize_atom("unknown_atom_feature") == 1
    
    # Test known bond from fixture
    known_bond_feat = "C-C (rdkit.Chem.rdchem.BondType.SINGLE, False)"
    assert tokenizer.tokenize_bond(known_bond_feat) == 2
    
    # Test unknown bond
    assert tokenizer.tokenize_bond("unknown_bond_feature") == 1
    
    # Test loaded num_classes - 1 for unk, +1 for each unique feature string in fixture
    # Fixture defines 1 unique atom + 1 unk = 2 num_classes
    # Fixture defines 2 unique bonds + 1 unk = 3 num_classes
    assert tokenizer.atom_num_classes == 3 # unk + C + O
    assert tokenizer.bond_num_classes == 4 # unk + C-C + C-O + O-C (flipped)


def test_tokenizer_build_from_smiles():
    """Tests the dynamic vocabulary building functionality."""
    tokenizer = Tokenizer() # Start with an empty tokenizer
    smiles_list = ["C", "CO"]
    
    tokenizer.build_from_smiles(smiles_list)
    
    assert tokenizer.atom_num_classes > 1 # C and H will be added
    assert tokenizer.bond_num_classes > 1 # C-H, C-C, C-O will be added
    
    mol_co = Chem.MolFromSmiles("CO")
    mol_co = Chem.AddHs(mol_co)
    bond_co = mol_co.GetBondWithIdx(0) # C-O bond
    bond_feat_co = bond_featurizer(bond_co, flipped=False)
    assert tokenizer.tokenize_bond(bond_feat_co) > 1 # Should not be 'unk'

def test_tokenizer_save_and_load():
    """Tests saving the tokenizer's vocab and reloading it."""
    smiles_list = ["CCO", "CNC"]
    
    # Create and build a tokenizer
    tokenizer1 = Tokenizer()
    tokenizer1.build_from_smiles(smiles_list)
    
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as f:
        save_path = f.name
    
    tokenizer1.save(save_path)
    
    # Create a new tokenizer and load from the saved file
    tokenizer2 = Tokenizer(vocab_filepath=save_path)
    
    assert tokenizer1._atom_vocab == tokenizer2._atom_vocab
    assert tokenizer1._bond_vocab == tokenizer2._bond_vocab
    assert tokenizer1.atom_num_classes == tokenizer2.atom_num_classes
    assert tokenizer1.bond_num_classes == tokenizer2.bond_num_classes
    
    # Cleanup
    os.remove(save_path)

# --- New tests for BaseFeaturizer implementations and factory ---

@pytest.fixture
def data_config_token(sample_vocab_file):
    """Provides a DataConfig for TokenFeaturizer."""
    # Ensure vocab_path is accessible
    return DataConfig(
        base_data_paths=[], extra_data_paths=[], cross_validation="none",
        dataset_dir="dummy", target_columns=["bde"],
        test_size=0.1, val_size=0.1, random_seed=42, featurizer_type="TokenFeaturizer"
    )

@pytest.fixture
def data_config_chemprop():
    """Provides a DataConfig for ChemPropFeaturizer."""
    return DataConfig(
        base_data_paths=[], extra_data_paths=[], cross_validation="none",
        dataset_dir="dummy", target_columns=["bde"],
        test_size=0.1, val_size=0.1, random_seed=42, featurizer_type="ChemPropFeaturizer"
    )

def test_token_featurizer_properties(sample_vocab_file):
    """Tests properties of TokenFeaturizer."""
    featurizer = get_featurizer(
        featurizer_type="TokenFeaturizer",
        smiles_list=["CCO", "C"],
        save_path=sample_vocab_file
    )
    assert isinstance(featurizer, TokenFeaturizer)
    assert featurizer.is_discrete is True
    assert featurizer.atom_dim > 1
    assert featurizer.bond_dim > 1

def test_token_featurizer_featurize(sample_mol):
    """Tests featurize method of TokenFeaturizer."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as f:
        temp_vocab_path = f.name
        
    featurizer = get_featurizer(
        featurizer_type="TokenFeaturizer",
        smiles_list=[Chem.MolToSmiles(sample_mol)],
        save_path=temp_vocab_path
    )

    pyg_data = featurizer.featurize(sample_mol, smiles="CCO")
    assert isinstance(pyg_data, Data)
    assert pyg_data.x.dtype == torch.long
    assert pyg_data.edge_attr.dtype == torch.long
    assert pyg_data.x.shape[0] == sample_mol.GetNumAtoms()
    assert pyg_data.edge_attr.shape[0] == sample_mol.GetNumBonds() * 2 # Bidirectional edges
    
    # Check for labels if provided (optional). Mock list of float for multi-task support.
    labels = {(0, 1): [80.0], (1, 2): [90.0]} # C-C and C-O bond
    pyg_data_with_labels = featurizer.featurize(sample_mol, labels=labels, smiles="CCO")
    assert pyg_data_with_labels.y.dtype == torch.float
    assert pyg_data_with_labels.mask.dtype == torch.bool
    assert pyg_data_with_labels.y.shape == (sample_mol.GetNumBonds() * 2, 1) # Assumes 1 task in mock

    # Cleanup temp vocab file
    os.remove(temp_vocab_path)


def test_chemprop_featurizer_properties():
    """Tests properties of ChemPropFeaturizer."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as f:
        temp_vocab_path = f.name
        
    featurizer = get_featurizer(
        featurizer_type="ChemPropFeaturizer",
        smiles_list=["CCO", "C"],
        save_path=temp_vocab_path
    )
    assert isinstance(featurizer, ChemPropPyGFeaturizer)
    assert featurizer.is_discrete is False
    assert featurizer.atom_dim > 1
    assert featurizer.bond_dim > 1
    os.remove(temp_vocab_path)

def test_chemprop_featurizer_featurize(sample_mol):
    """Tests featurize method of ChemPropFeaturizer."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as f:
        temp_vocab_path = f.name
        
    featurizer = get_featurizer(
        featurizer_type="ChemPropFeaturizer",
        smiles_list=[Chem.MolToSmiles(sample_mol)],
        save_path=temp_vocab_path
    )
    
    pyg_data = featurizer.featurize(sample_mol, smiles="CCO")
    assert isinstance(pyg_data, Data)
    assert pyg_data.x.dtype == torch.float
    assert pyg_data.edge_attr.dtype == torch.float
    assert pyg_data.x.shape[0] == sample_mol.GetNumAtoms()
    assert pyg_data.edge_attr.shape[0] == sample_mol.GetNumBonds() * 2 # Bidirectional edges
    assert pyg_data.x.shape[1] == featurizer.atom_dim
    assert pyg_data.edge_attr.shape[1] == featurizer.bond_dim

    # Check for labels if provided (optional). Multi-task mock list layout.
    labels = {(0, 1): [80.0], (1, 2): [90.0]} # C-C and C-O bond
    pyg_data_with_labels = featurizer.featurize(sample_mol, labels=labels, smiles="CCO")
    assert pyg_data_with_labels.y.dtype == torch.float
    assert pyg_data_with_labels.mask.dtype == torch.bool
    assert pyg_data_with_labels.y.shape == (sample_mol.GetNumBonds() * 2, 1)

    os.remove(temp_vocab_path)

def test_get_featurizer_factory():
    """Tests the get_featurizer factory function."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as f:
        temp_vocab_path = f.name
        
    featurizer_token = get_featurizer(
        featurizer_type="TokenFeaturizer",
        smiles_list=["CCO"],
        save_path=temp_vocab_path
    )
    assert isinstance(featurizer_token, TokenFeaturizer)
    
    featurizer_chemprop = get_featurizer(
        featurizer_type="ChemPropFeaturizer",
        smiles_list=["CCO"],
        save_path=temp_vocab_path
    )
    assert isinstance(featurizer_chemprop, ChemPropPyGFeaturizer)
    os.remove(temp_vocab_path)

def test_get_featurizer_unknown_type():
    """Tests error handling for unknown featurizer type."""
    with pytest.raises(ValueError, match="Unknown featurizer type"):
        get_featurizer(
            featurizer_type="UnknownFeaturizer",
            smiles_list=["CCO"],
            save_path="dummy.json"
        )