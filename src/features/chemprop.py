from typing import Dict, List, Optional, Tuple
import math

from enum import Enum, auto
from typing import Sequence, TypeVar, Generic, List, Dict, Any, NamedTuple, Optional, Union, Tuple
from abc import abstractmethod
from collections.abc import Sized
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import Atom, Bond, HybridizationType, BondType
from torch_geometric.data import Data

from src.features.base import BaseFeaturizer

# --- Start: Shims for missing chemprop base classes ---

class MolGraph(NamedTuple):
    """A :class:`MolGraph` represents the graph featurization of a molecule."""
    V: np.ndarray
    E: np.ndarray
    edge_index: np.ndarray
    rev_edge_index: np.ndarray

S = TypeVar("S")
T = TypeVar("T")

class Featurizer(Generic[S, T]):
    """An :class:`Featurizer` featurizes inputs type ``S`` into outputs of type ``T``."""
    @abstractmethod
    def __call__(self, input: S, *args, **kwargs) -> T:
        """featurize an input"""

class VectorFeaturizer(Featurizer[S, np.ndarray], Sized):
    ...

class GraphFeaturizer(Featurizer[S, MolGraph]):
    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int]:
        ...

class EnumMapping(Enum):
    """Helper for Enum mapping, mimicking chemprop's structure."""
    @classmethod
    def get(cls, value: Any):
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls[value.upper()]
            except KeyError:
                pass
        raise ValueError(f"Invalid value '{value}' for {cls.__name__}")

# --- End: Shims ---


class MultiHotAtomFeaturizer(VectorFeaturizer[Atom]):
    """A :class:`MultiHotAtomFeaturizer` uses a multi-hot encoding to featurize atoms."""
    def __init__(
        self,
        atomic_nums: Sequence[int],
        degrees: Sequence[int],
        formal_charges: Sequence[int],
        chiral_tags: Sequence[int],
        num_Hs: Sequence[int],
        hybridizations: Sequence[int],
    ):
        self.atomic_nums = {j: i for i, j in enumerate(atomic_nums)}
        self.degrees = {i: i for i in degrees}
        self.formal_charges = {j: i for i, j in enumerate(formal_charges)}
        self.chiral_tags = {i: i for i in chiral_tags}
        self.num_Hs = {i: i for i in num_Hs}
        self.hybridizations = {ht: i for i, ht in enumerate(hybridizations)}

        self._subfeats: List[Dict] = [
            self.atomic_nums, self.degrees, self.formal_charges,
            self.chiral_tags, self.num_Hs, self.hybridizations,
        ]
        subfeat_sizes = [
            1 + len(self.atomic_nums), 1 + len(self.degrees), 1 + len(self.formal_charges),
            1 + len(self.chiral_tags), 1 + len(self.num_Hs), 1 + len(self.hybridizations),
            1, 1,
        ]
        self.__size = sum(subfeat_sizes)

    def __len__(self) -> int:
        return self.__size

    def __call__(self, a: Optional[Atom]) -> np.ndarray:
        x = np.zeros(self.__size)
        if a is None: return x

        feats = [
            a.GetAtomicNum(), a.GetTotalDegree(), a.GetFormalCharge(),
            int(a.GetChiralTag()), int(a.GetTotalNumHs()), a.GetHybridization(),
        ]
        i = 0
        for feat, choices in zip(feats, self._subfeats):
            j = choices.get(feat, len(choices))
            x[i + j] = 1
            i += len(choices) + 1
        x[i] = int(a.GetIsAromatic())
        x[i + 1] = 0.01 * a.GetMass()
        return x

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

    @classmethod
    def v1(cls, max_atomic_num: int = 100):
        return cls(
            atomic_nums=list(range(1, max_atomic_num + 1)), degrees=list(range(6)),
            formal_charges=[-1, -2, 1, 2, 0], chiral_tags=list(range(4)), num_Hs=list(range(5)),
            hybridizations=[HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3, HybridizationType.SP3D, HybridizationType.SP3D2],
        )

    @classmethod
    def v2(cls):
        return cls(
            atomic_nums=list(range(1, 37)) + [53], degrees=list(range(6)),
            formal_charges=[-1, -2, 1, 2, 0], chiral_tags=list(range(4)), num_Hs=list(range(5)),
            hybridizations=[HybridizationType.S, HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP2D, HybridizationType.SP3, HybridizationType.SP3D, HybridizationType.SP3D2],
        )

    @classmethod
    def organic(cls):
        return cls(
            atomic_nums=[1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53], degrees=list(range(6)),
            formal_charges=[-1, -2, 1, 2, 0], chiral_tags=list(range(4)), num_Hs=list(range(5)),
            hybridizations=[HybridizationType.S, HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3],
        )

class AtomFeatureMode(EnumMapping):
    V1 = auto()
    V2 = auto()
    ORGANIC = auto()

def get_multi_hot_atom_featurizer(mode: Union[str, AtomFeatureMode]) -> MultiHotAtomFeaturizer:
    mode_enum = AtomFeatureMode.get(mode)
    if mode_enum == AtomFeatureMode.V1:
        return MultiHotAtomFeaturizer.v1()
    elif mode_enum == AtomFeatureMode.V2:
        return MultiHotAtomFeaturizer.v2()
    elif mode_enum == AtomFeatureMode.ORGANIC:
        return MultiHotAtomFeaturizer.organic()
    else:
        raise RuntimeError("unreachable code reached!")


class MultiHotBondFeaturizer(VectorFeaturizer[Bond]):
    def __init__(
        self, bond_types: Optional[Sequence[BondType]] = None, stereos: Optional[Sequence[int]] = None
    ):
        self.bond_types = bond_types or [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]
        self.stereo = stereos or range(6)

    def __len__(self):
        return 1 + len(self.bond_types) + 2 + (len(self.stereo) + 1)

    def __call__(self, b: Bond) -> np.ndarray:
        x = np.zeros(len(self), int)
        if b is None:
            x[0] = 1
            return x

        i = 1
        bond_type = b.GetBondType()
        try:
            bt_bit = self.bond_types.index(bond_type)
            x[i + bt_bit] = 1
        except ValueError:
            pass # Keep as all-zero if bond type is not in the list
        i += len(self.bond_types)

        x[i] = int(b.GetIsConjugated())
        i += 1
        x[i] = int(b.IsInRing())
        i += 1
        
        try:
            stereo_bit = self.stereo.index(int(b.GetStereo()))
            x[i + stereo_bit] = 1
        except ValueError:
            x[i + len(self.stereo)] = 1 # Unknown stereo
        
        return x



class ChemPropPyGFeaturizer(BaseFeaturizer):
    """
    An adapter that uses Chemprop's featurizers (atom and bond) and converts
    the output into a PyTorch Geometric `Data` object.
    """

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
        # Chemprop featurizers produce continuous (one-hot) vectors
        return False

    def featurize(
        self,
        mol: Chem.Mol,
        labels: Optional[Dict[Tuple[int, int], List[float]]] = None,
        smiles: str = "",
    ) -> Optional[Data]:
        """
        Featurizes the molecule into a PyG Data object.

        Args:
            mol (Chem.Mol): RDKit molecule with hydrogens.
            labels (Optional[Dict...]]): Optional BDE labels for training.
            smiles (str): Canonical SMILES string.

        Returns:
            Optional[Data]: A PyG Data object or None if the molecule has no bonds.
        """
        # Atom features (V)
        atom_features = [self.atom_featurizer(atom) for atom in mol.GetAtoms()]
        x = torch.from_numpy(np.array(atom_features)).float()

        # Bond features (E) and edge_index
        edge_indices, edge_attrs, bond_indices_map = [], [], []
        
        is_training = labels is not None
        edge_bde_labels, edge_masks = [], []

        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_feat = self.bond_featurizer(bond)
            
            # Add forward and backward edges
            edge_indices.extend([(u, v), (v, u)])
            edge_attrs.extend([bond_feat, bond_feat])
            bond_indices_map.extend([bond.GetIdx(), bond.GetIdx()])

            if is_training:
                canonical_bond_key = tuple(sorted((u, v)))
                bde_labels = labels.get(canonical_bond_key)
                
                # Assign labels to both directed edges
                if bde_labels is not None:
                    # Convert NaNs to 0 internally, and use mask to ignore them during loss computation
                    cleaned_labels = [float(lbl) if not math.isnan(lbl) else 0.0 for lbl in bde_labels]
                    masks = [not math.isnan(lbl) for lbl in bde_labels]
                    
                    edge_bde_labels.extend([cleaned_labels, cleaned_labels])
                    edge_masks.extend([masks, masks])
                else:
                    # Assume 1 task dimension if labels is empty just to avoid breaking during inference init test
                    num_tasks = len(list(labels.values())[0]) if len(labels) > 0 else 1
                    zero_labels = [0.0] * num_tasks
                    false_masks = [False] * num_tasks
                    
                    edge_bde_labels.extend([zero_labels, zero_labels])
                    edge_masks.extend([false_masks, false_masks])
        
        if not edge_indices:
            return None

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.from_numpy(np.array(edge_attrs)).float()

        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.bond_indices_map = torch.tensor(bond_indices_map, dtype=torch.long)
        data.original_input_smiles = smiles
        data.is_valid = torch.tensor(True, dtype=torch.bool) # Assume valid if processed

        if is_training:
            data.y = torch.tensor(edge_bde_labels, dtype=torch.float)
            data.mask = torch.tensor(edge_masks, dtype=torch.bool)

        return data
