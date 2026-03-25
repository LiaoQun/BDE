"""
This module provides functions to generate standardized templates for BDE data annotation,
faithfully porting the core fragmentation logic from the original `alfabet` library.
"""
import logging
from collections import Counter
from typing import Dict, Iterator, List, Type

import pandas as pd
import rdkit
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm


logger = logging.getLogger(__name__)
# Suppress RDKit logging to keep the output clean
RDLogger.DisableLog("rdApp.*")


class Molecule:
    """A wrapper class for RDKit Mol objects to handle caching."""

    def __init__(self, mol: Type[Chem.Mol] = None, smiles: str = None) -> None:
        if not ((mol is not None) or (smiles is not None)):
            raise ValueError("mol or smiles must be provided")
        self._mol = mol
        self._smiles = smiles
        self._molH = None

    @property
    def mol(self) -> Type[Chem.Mol]:
        if self._mol is None:
            self._mol = Chem.MolFromSmiles(self._smiles)
        return self._mol

    @property
    def molH(self) -> Type[Chem.Mol]:
        if self._molH is None:
            self._molH = Chem.AddHs(self.mol)
        return self._molH

    @property
    def smiles(self) -> str:
        if self._smiles is None:
            self._smiles = Chem.MolToSmiles(self.mol)
        return self._smiles


def get_bond_type(bond: Type[Chem.Bond]) -> str:
    """Creates a standardized string representation for a bond type."""
    return "-".join(sorted((bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol())))


def count_atom_types(molecule: Molecule) -> Counter:
    """Returns a Counter dictionary of each atom type in the molecule."""
    return Counter([atom.GetSymbol() for atom in molecule.molH.GetAtoms()])


def count_stereocenters(molecule: Molecule) -> Dict[str, int]:
    """Counts both assigned and unassigned stereocenters in the molecule."""
    Chem.FindPotentialStereoBonds(molecule.mol)
    stereocenters = Chem.FindMolChiralCenters(molecule.mol, includeUnassigned=True)
    stereobonds = [
        bond for bond in molecule.mol.GetBonds()
        if bond.GetStereo() is not Chem.rdchem.BondStereo.STEREONONE
    ]

    return {
        "atom_assigned": len([center for center in stereocenters if center[1] != "?"]),
        "atom_unassigned": len([center for center in stereocenters if center[1] == "?"]),
        "bond_assigned": len(
            [bond for bond in stereobonds if bond.GetStereo() is not Chem.rdchem.BondStereo.STEREOANY]
        ),
        "bond_unassigned": len(
            [bond for bond in stereobonds if bond.GetStereo() is Chem.rdchem.BondStereo.STEREOANY]
        ),
    }


def check_stereocenters(molecule: Molecule) -> bool:
    """Checks if the molecule has a reasonable number of unassigned stereocenters."""
    stereocenters = count_stereocenters(molecule)
    if stereocenters["bond_unassigned"] > 0:
        return False

    max_unassigned = 1 if stereocenters["atom_assigned"] == 0 else 1
    return stereocenters["atom_unassigned"] <= max_unassigned


def _fragment_iterator(input_molecule: Molecule, skip_warnings: bool = False) -> Iterator[Dict]:
    """
    Iterates through valid bonds, fragments the molecule, and yields fragment data.
    This is a direct port of the logic from `alfabet.fragment.fragment_iterator`.
    """
    mol_stereo = count_stereocenters(input_molecule)
    if (mol_stereo["atom_unassigned"] != 0) or (mol_stereo["bond_unassigned"] != 0):
        logger.warning(f"Molecule {input_molecule.smiles} has undefined stereochemistry")
        if skip_warnings:
            return

    # Use a copy for kekulization to avoid modifying the original molH
    kekulized_mol = Chem.Mol(input_molecule.molH)
    Chem.Kekulize(kekulized_mol, clearAromaticFlags=True)

    for bond in kekulized_mol.GetBonds():
        # Filter 1: Only single, non-ring bonds
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue

        try:
            # Manually remove the bond to create radicals
            rw_mol = Chem.RWMol(kekulized_mol)
            a1_idx, a2_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            rw_mol.RemoveBond(a1_idx, a2_idx)

            # Prevent RDKit from auto-updating hydrogens on the affected atoms
            rw_mol.GetAtomWithIdx(a1_idx).SetNoImplicit(True)
            rw_mol.GetAtomWithIdx(a2_idx).SetNoImplicit(True)

            # Sanitize the molecule; this step generates the radical electrons
            Chem.SanitizeMol(rw_mol)

            # Convert the fragmented molecule to SMILES; fragments are separated by '.'
            fragmented_smiles = Chem.MolToSmiles(rw_mol, isomericSmiles=True)
            
            # Split fragments and canonicalize by sorting
            frags_smiles_list = sorted(fragmented_smiles.split("."), key=len,reverse=True)
            if len(frags_smiles_list) != 2:
                logger.warning(f"Fragmentation of {input_molecule.smiles} bond {bond.GetIdx()} did not yield 2 fragments. Got: {frags_smiles_list}")
                continue
            
            # Standardize fragment SMILES to be implicit-H for consistency with original
            standardized_frags = []
            for frag_smiles in frags_smiles_list:
                frag_mol = Chem.MolFromSmiles(frag_smiles)
                if frag_mol:
                    frag_mol = Chem.RemoveHs(frag_mol)
                    standardized_frags.append(Chem.MolToSmiles(frag_mol, isomericSmiles=True))
                else:
                    standardized_frags.append(frag_smiles) # Fallback if parsing fails

            frag1_smiles, frag2_smiles = standardized_frags
            frag1 = Molecule(smiles=frag1_smiles)
            frag2 = Molecule(smiles=frag2_smiles)

            # Stoichiometry check
            if (count_atom_types(frag1) + count_atom_types(frag2)) != count_atom_types(input_molecule):
                 logger.error(f"Atom count mismatch for {input_molecule.smiles} -> {frag1.smiles} + {frag2.smiles}")
                 continue

            yield {
                "molecule": input_molecule.smiles,
                "bond_index": bond.GetIdx(),
                "bond_type": get_bond_type(bond),
                "fragment1": frag1.smiles,
                "fragment2": frag2.smiles,
                "is_valid_stereo": check_stereocenters(frag1) and check_stereocenters(frag2),
            }

        except Exception as e:
            logger.error(f"Fragmentation error on {input_molecule.smiles} bond {bond.GetIdx()}: {e}")
            continue


def generate_fragment_template(smiles_list: List[str]) -> pd.DataFrame:
    """
    Generates a rich DataFrame template for BDE annotation from a list of SMILES,
    based on the fragmentation logic from `alfabet`.
    """
    records = []
    for smiles in tqdm(smiles_list, desc="Processing molecules"):
        try:
            # First, canonicalize the input SMILES to ensure consistency
            temp_mol = Chem.MolFromSmiles(smiles)
            if temp_mol is None:
                logger.error(f"Could not parse SMILES: {smiles}")
                continue
            canonical_smiles = Chem.MolToSmiles(temp_mol, isomericSmiles=True)
            
            # Use the canonical SMILES for all subsequent operations
            mol = Molecule(smiles=canonical_smiles)
            
            for fragment_data in _fragment_iterator(mol):
                # Ensure the original canonical SMILES is in the record
                fragment_data['molecule'] = canonical_smiles
                records.append(fragment_data)
        except Exception as e:
            logger.error(f"Failed to process molecule {smiles}: {e}")
    
    # Add the 'bde' column at the end
    df = pd.DataFrame(records)
    df['bde'] = None
    
    # Define final column order
    final_columns = [
        "molecule", "bond_index", "bond_type", 
        "fragment1", "fragment2", "is_valid_stereo", "bde"
    ]
    # Reorder columns, handling cases where no records were generated
    for col in final_columns:
        if col not in df:
            df[col] = None

    return df[final_columns]


