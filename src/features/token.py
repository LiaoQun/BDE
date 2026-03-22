import json
import os
from typing import Dict, List, Optional, Tuple, Union
import math

import torch
from rdkit import Chem
from rdkit.Chem.rdchem import Atom, Bond
from torch_geometric.data import Data

from src.features.base import BaseFeaturizer


def get_ring_size(mol_obj: Union[Atom, Bond], max_size: int = 6) -> int:
    """
    Helper to determine the size of the smallest ring an atom or bond is in.
    """
    if not mol_obj.IsInRing():
        return 0

    min_ring_size = float('inf')
    for ring in mol_obj.GetOwningMol().GetRingInfo().AtomRings():
        if mol_obj.GetIdx() in ring:
            min_ring_size = min(min_ring_size, len(ring))

    if min_ring_size <= max_size:
        return min_ring_size
    else:
        return max_size


def atom_featurizer(atom: Atom) -> str:
    """
    Generates a feature string for an RDKit Atom object.
    """
    return str(
        (
            atom.GetSymbol(),
            atom.GetNumRadicalElectrons(),
            atom.GetFormalCharge(),
            atom.GetChiralTag(),
            atom.GetIsAromatic(),
            get_ring_size(atom, max_size=6),
            atom.GetDegree(),
            atom.GetTotalNumHs(includeNeighbors=True),
        )
    )


def bond_featurizer(bond: Bond, flipped: bool = False) -> str:
    """
    Generates a feature string for an RDKit Bond object.
    """
    if not flipped:
        atoms = "{}-{}".format(
            bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()
        )
    else:
        atoms = "{}-{}".format(
            bond.GetEndAtom().GetSymbol(), bond.GetBeginAtom().GetSymbol()
        )

    btype = str((bond.GetBondType(), bond.GetIsConjugated()))
    ring = f"R{get_ring_size(bond, max_size=6)}" if bond.IsInRing() else ""

    return " ".join([atoms, btype, ring]).strip()


class Tokenizer:
    """
    Handles the mapping of feature strings to integer IDs for atoms and bonds.
    """

    def __init__(self, vocab_filepath: str = None):
        self._atom_vocab: Dict[str, int] = {"unk": 1}
        self._bond_vocab: Dict[str, int] = {"unk": 1}
        self.atom_num_classes: int = 1
        self.bond_num_classes: int = 1

        if vocab_filepath:
            self.load(vocab_filepath)

    def load(self, filepath: str):
        """Loads vocabularies from a JSON file."""
        if not os.path.exists(filepath):
            # It's okay if file doesn't exist on init, we might build it later.
            # But if explicitly called, we warn or raise.
            print(f"Warning: Vocabulary file not found at: {filepath}. Starting with empty vocab.")
            return

        with open(filepath, 'r') as f:
            data = json.load(f)

        if "atom_tokenizer" in data and "_data" in data["atom_tokenizer"]:
            self._atom_vocab = data["atom_tokenizer"]["_data"]
            self.atom_num_classes = data["atom_tokenizer"].get("num_classes", max(self._atom_vocab.values()))

        if "bond_tokenizer" in data and "_data" in data["bond_tokenizer"]:
            self._bond_vocab = data["bond_tokenizer"]["_data"]
            self.bond_num_classes = data["bond_tokenizer"].get("num_classes", max(self._bond_vocab.values()))

    def _add_feature(self, feature_string: str, feature_type: str) -> int:
        vocab = self._atom_vocab if feature_type == 'atom' else self._bond_vocab
        if feature_string not in vocab:
            if feature_type == 'atom':
                self.atom_num_classes += 1
                vocab[feature_string] = self.atom_num_classes
            else:
                self.bond_num_classes += 1
                vocab[feature_string] = self.bond_num_classes
        return vocab[feature_string]

    def tokenize_atom(self, atom_feature_string: str) -> int:
        return self._atom_vocab.get(atom_feature_string, self._atom_vocab["unk"])

    def tokenize_bond(self, bond_feature_string: str) -> int:
        return self._bond_vocab.get(bond_feature_string, self._bond_vocab["unk"])

    def build_from_smiles(self, smiles_list: List[str]):
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            mol = Chem.AddHs(mol)

            for atom in mol.GetAtoms():
                self._add_feature(atom_featurizer(atom), 'atom')

            for bond in mol.GetBonds():
                self._add_feature(bond_featurizer(bond, flipped=False), 'bond')
                self._add_feature(bond_featurizer(bond, flipped=True), 'bond')

    def save(self, filepath: str):
        data = {
            "atom_tokenizer": {"_data": self._atom_vocab, "num_classes": self.atom_num_classes},
            "bond_tokenizer": {"_data": self._bond_vocab, "num_classes": self.bond_num_classes}
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)


class TokenFeaturizer(BaseFeaturizer):
    """
    Featurizer that uses a dictionary (Tokenizer) to map unique string representations
    of atoms and bonds to integer IDs.
    """
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


    def featurize(self, mol: Chem.Mol, 
                  labels: Optional[Dict[Tuple[int, int], List[float]]] = None,
                  smiles: str = "") -> Optional[Data]:
        """
        Converts a single RDKit Mol object into a PyG Data object.
        """
        # 1. Atom Features and Validity
        atom_feature_strings = [atom_featurizer(mol_atom) for mol_atom in mol.GetAtoms()]
        x = torch.LongTensor([self.tokenizer.tokenize_atom(s) for s in atom_feature_strings])
        atoms_are_valid = (x != 1).all().item()

        # 2. Edge Features, BDE Labels, and Validity
        is_training = labels is not None
        edge_indices, edge_attrs, bond_indices_map = [], [], []
        edge_bde_labels, edge_masks = [], []

        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            
            # Add forward and backward edges
            for (start_atom, end_atom, is_flipped) in [(u, v, False), (v, u, True)]:
                edge_indices.append((start_atom, end_atom))
                edge_attrs.append(self.tokenizer.tokenize_bond(bond_featurizer(bond, flipped=is_flipped)))
                bond_indices_map.append(bond.GetIdx())
                
                if is_training:
                    canonical_bond_key = tuple(sorted((u, v)))
                    bde_labels = labels.get(canonical_bond_key)
                    
                    if bde_labels is not None:
                        # Convert NaNs to 0 internally, and use mask to ignore them during loss computation
                        cleaned_labels = [float(lbl) if not math.isnan(lbl) else 0.0 for lbl in bde_labels]
                        masks = [not math.isnan(lbl) for lbl in bde_labels]
                        
                        edge_bde_labels.append(cleaned_labels)
                        edge_masks.append(masks)
                    else:
                        # No training labels available for this edge, but we might still need to match dimensions
                        # Assume 1 task dimension if labels is empty just to avoid breaking during inference init test
                        num_tasks = len(list(labels.values())[0]) if len(labels) > 0 else 1
                        edge_bde_labels.append([0.0] * num_tasks)
                        edge_masks.append([False] * num_tasks)

        if not edge_indices:
            return None

        edge_index = torch.LongTensor(edge_indices).t().contiguous()
        edge_attr = torch.LongTensor(edge_attrs)
        bonds_are_valid = (edge_attr != 1).all().item()
        
        # 3. Create Data Object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        # Attach training-specific attributes if available
        if is_training:
            data.y = torch.FloatTensor(edge_bde_labels)
            data.mask = torch.BoolTensor(edge_masks)
            
        # Attach inference-specific and common attributes
        data.bond_indices_map = torch.LongTensor(bond_indices_map)
        data.original_input_smiles = smiles
        data.is_valid = torch.tensor(atoms_are_valid and bonds_are_valid, dtype=torch.bool)
        
        return data

# Wrapper for backward compatibility if needed, though we should update call sites.
def mol_to_graph(mol, tokenizer, canonical_smiles, bde_labels_dict=None):
    # This is a temporary bridge.
    # We construct a TokenFeaturizer with the provided tokenizer to run featurize.
    featurizer = TokenFeaturizer(tokenizer)
    return featurizer.featurize(mol, bde_labels_dict, canonical_smiles)