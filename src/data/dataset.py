import os
import logging # Import logging
from typing import Dict, List, Tuple

import torch
from torch_geometric.data import InMemoryDataset
from rdkit import Chem

from src.features.base import BaseFeaturizer

logger = logging.getLogger(__name__) # Get a logger for this module


class BDEDataset(InMemoryDataset):
    """
    A PyTorch Geometric InMemoryDataset for BDE prediction.
    Processes SMILES strings into PyG Data objects with atom and bond features,
    BDE labels, and a loss mask. All data is loaded into memory for faster training epochs.
    """
    def __init__(self, root: str, smiles_data: List[Tuple[str, Dict[Tuple[int, int], float]]], 
                 featurizer: BaseFeaturizer, transform=None, pre_transform=None):
        """
        Args:
            root (str): Root directory where the dataset will be saved.
            smiles_data (List[Tuple[str, Dict[Tuple[int, int], float]]]): A list of tuples,
                                                                          each containing a SMILES string
                                                                          and a dictionary of BDE labels.
            featurizer (BaseFeaturizer): An initialized Featurizer instance.
        """
        self.smiles_data = smiles_data
        self.featurizer = featurizer
        super().__init__(root, transform, pre_transform)
        # The following `torch.load` call triggers a security warning because `weights_only` defaults to `False`.
        # This is REQUIRED for InMemoryDataset as we are loading a complex tuple `(data, slices)`
        # which contains non-tensor objects (like the PyG Data object itself).
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> List[str]:
        # No raw files, as data is passed directly
        return []

    @property
    def processed_file_names(self) -> List[str]:
        # The processed data is saved in a single file.
        return ['data.pt']
    
    def download(self):
        # Data is passed directly, so no download needed
        pass

    def process(self):
        """
        Processes SMILES data into PyG Data objects and saves them as a single collated file.
        """
        data_list = []
        for i, (smiles, bde_labels_dict) in enumerate(self.smiles_data):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    logger.warning(f"Skipping SMILES '{smiles}' (index {i}) due to RDKit parse error.")
                    continue
                mol = Chem.AddHs(mol)
                
                # Use the abstract featurizer
                data = self.featurizer.featurize(mol, labels=bde_labels_dict, smiles=smiles)

                if data is not None:
                    data_list.append(data)
            except ValueError as e:
                logger.warning(f"Skipping SMILES '{smiles}' (index {i}) due to featurization error: {e}", exc_info=True)
                continue
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Collate all Data objects into a single large Data object and save.
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
