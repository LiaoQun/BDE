import os
import pandas as pd
from typing import List, Tuple, Dict
from rdkit import Chem
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def load_and_merge_data(
    data_paths: List[str],
    target_columns: List[str] = ['bde'],
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Loads data from a list of CSV file paths, merges them, canonicalizes SMILES,
    and cleans the data.
    Allows rows with missing values as long as at least one task target is present.

    Supports cross-file multitask scenarios where different files contribute
    different target columns (e.g., one file has 'bde', another has 'bdfe').
    Missing target columns are padded with NaN for downstream consistency.

    Args:
        data_paths: Paths to CSV (or .csv.gz) files to load and merge.
        target_columns: Column names for prediction targets.
        random_seed: Random seed used for SMILES-deduplication groupby tie-breaking.

    Returns:
        A single cleaned DataFrame with all records across the provided files.
    """
    if not data_paths:
        raise ValueError("No data paths provided in the configuration.")

    df_list = []
    logger.info("Loading data from the following paths:")
    for path in data_paths:
        if os.path.exists(path):
            logger.info(f" - Loading {path}...")
            try:
                df_list.append(pd.read_csv(path))
            except Exception as e:
                logger.warning(f"Could not read file {path}. Error: {e}. Skipping.", exc_info=True)
        else:
            logger.warning(f"Data file not found at: {path}. Skipping.")
    
    if not df_list:
        raise FileNotFoundError("No valid data files could be loaded from the specified paths.")

    logger.info("Merging and cleaning data...")
    merged_df = pd.concat(df_list, ignore_index=True)
    logger.info(f"Total records loaded: {len(merged_df)}")

    # Handle missing values: We keep the row as long as molecule, bond_index exist and AT LEAST ONE target exists.
    initial_rows = len(merged_df)
    
    # Must have molecule and bond_index
    merged_df.dropna(subset=['molecule', 'bond_index'], inplace=True)

    # Must have at least one valid target present for multi-task support.
    # Crucially: only check columns that ACTUALLY EXIST in this specific file.
    # In cross-file multitask scenarios, some files may not have all target
    # columns (e.g., an extra file with 'bdfe' but not 'bde').  Calling
    # dropna() on a non-existent column raises KeyError in pandas >= 1.3.
    existing_targets = [col for col in target_columns if col in merged_df.columns]
    if existing_targets:
        merged_df = merged_df.dropna(subset=existing_targets, how='all')
    else:
        logger.warning(
            f"None of the target columns {target_columns} found in this dataset. "
            "All rows will be kept without target filtering."
        )

    # Pad missing target columns with NaN so that pd.concat() downstream
    # always produces a DataFrame with a consistent column schema.
    for col in target_columns:
        if col not in merged_df.columns:
            merged_df[col] = float('nan')
            logger.debug(
                f"Added placeholder column '{col}' (all NaN) to align schema "
                "with target_columns."
            )

    if initial_rows > len(merged_df):
        logger.info(f"Dropped {initial_rows - len(merged_df)} rows missing critical key values or containing NO valid targets.")

    # --- Canonicalize SMILES ---
    logger.info("Canonicalizing SMILES strings...")
    
    def canonicalize(smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            return Chem.MolToSmiles(mol, canonical=True) if mol else None
        except Exception as e:
            logger.debug(f"Failed to canonicalize SMILES '{smi}': {e}", exc_info=True)
            return None

    initial_rows = len(merged_df)
    merged_df['molecule'] = merged_df['molecule'].apply(canonicalize)
    merged_df.dropna(subset=['molecule'], inplace=True)
    if initial_rows > len(merged_df):
        logger.info(f"Dropped {initial_rows - len(merged_df)} rows due to invalid/unparsable SMILES strings.")

    # --- Handle separate multi-task source combinations & multiple identical bond targets ---
    # Group by identical molecules and bonds to fuse multi-file target columns.
    # We use `.first()` here to keep the first non-null valid target for each task column,
    # effectively prioritizing data that was parsed first and ignoring subsequent overlaps.
    initial_rows = len(merged_df)
    merged_df = merged_df.groupby(['molecule', 'bond_index'], as_index=False).first()
    if initial_rows > len(merged_df):
        logger.info(f"Grouped {initial_rows - len(merged_df)} overlapping/duplicate records from multi-CSV splits (prioritizing first available targets).")

    logger.info(f"Final cleaned dataset contains {len(merged_df)} unique bond records.")
    return merged_df


def prepare_data(df: pd.DataFrame, target_columns: List[str] = ['bde']) -> List[Tuple[str, Dict[Tuple[int, int], List[float]]]]:
    """
    Processes a DataFrame into a list of (SMILES, labels_dict) tuples.
    Instead of single floats, labels are now lists of floats representing targets. NaN is allowed.
    """
    processed_smiles_data: List[Tuple[str, Dict[Tuple[int, int], List[float]]]] = []
    grouped_df = df.groupby('molecule')
    
    logger.info(f"Preparing BDE labels for {len(grouped_df)} unique molecules...")
    for smiles, mol_df in tqdm(grouped_df, desc="Processing molecules for labels"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Skipping molecule '{smiles}' due to RDKit parse error during label preparation.")
            continue
        mol = Chem.AddHs(mol)

        labels_dict = {}
        for _, row in mol_df.iterrows():
            bond_idx = int(row['bond_index'])
            
            try:
                if bond_idx >= mol.GetNumBonds():
                    logger.warning(f"Bond index {bond_idx} out of range for molecule '{smiles}'. Skipping bond.")
                    continue
                bond = mol.GetBondWithIdx(bond_idx)
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                canonical_bond_key = (min(u, v), max(u, v))
                
                # Extract multi-task target list. Keep NaN as float('nan') 
                targets = []
                for col in target_columns:
                    val = row.get(col, float('nan'))
                    targets.append(float(val))
                    
                labels_dict[canonical_bond_key] = targets
            except Exception as e:
                logger.warning(f"Error processing bond for {smiles} at bond_index {bond_idx}: {e}", exc_info=True)
                pass
                
        processed_smiles_data.append((smiles, labels_dict))
        
    return processed_smiles_data
