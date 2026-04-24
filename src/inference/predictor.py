"""
This module contains the Predictor class for running inference with a trained BDE model.
"""
from glob import glob
import os
import logging # Import logging
from typing import List, Dict, Union, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data, Batch

from src.curation.template_generator import generate_fragment_template
from src.features import get_featurizer_from_vocab
from src.models.mpnn import BDEModel
from src.config import load_config #

logger = logging.getLogger(__name__) # Get a logger for this module


class Predictor:
    """Handles loading a trained model and making BDE predictions."""

    def __init__(self, model_path: str, vocab_path: str, featurizer_type: str = 'TokenFeaturizer', 
                 atom_features: int = 128, num_messages: int = 6, num_tasks: int = 1,
                 target_columns: List[str] = ['bde'], device: str = 'cpu'):
        """
        Initializes the Predictor.

        Args:
            model_path (str): Path to the trained model checkpoint (.pt file).
            vocab_path (str): Path to the vocabulary file (.json) used during training.
            featurizer_type (str): Type of featurizer to use ('TokenFeaturizer' or 'ChemPropFeaturizer').
            atom_features (int): Hidden dimension size for the model.
            num_messages (int): Number of message passing layers.
            num_tasks (int): Number of tasks the model was trained on.
            target_columns (List[str]): Names of the target tasks.
            device (str): The device to run inference on ('cpu' or 'cuda').
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
        
        # Vocab path check might depend on featurizer type, but we'll check it if provided
        if vocab_path and not os.path.exists(vocab_path) and featurizer_type == 'TokenFeaturizer':
             raise FileNotFoundError(f"Vocabulary file not found at: {vocab_path}")

        self.device = torch.device(device)
        
        # Initialize Featurizer
        self.featurizer = get_featurizer_from_vocab(
            featurizer_type=featurizer_type,
            vocab_path=vocab_path
        )

        self.target_columns = target_columns

        # Re-create model architecture based on featurizer dimensions
        self.model = BDEModel(
            atom_input_dim=self.featurizer.atom_dim,
            bond_input_dim=self.featurizer.bond_dim,
            atom_features=atom_features,
            num_messages=num_messages,
            inputs_are_discrete=self.featurizer.is_discrete,
            num_tasks=num_tasks
        ).to(self.device)
        
        # The warning for weights_only=False is a security feature.
        # It's safe to set weights_only=True here as we are only loading model parameters.
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        
        # --- Compatibility: Rename keys from old models ---
        # Map old 'embedding' keys to new 'encoder' keys
        key_mapping = {
            "atom_embedding.weight": "atom_encoder.weight",
            "bond_embedding.weight": "bond_encoder.weight",
            "bond_mean_embedding.weight": "bond_bias_encoder.weight"
        }
        
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key_mapping.get(key, key)
            new_state_dict[new_key] = value
            
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        logger.info("Model and featurizer loaded successfully.")

    def predict(self, smiles_list: List[str], drop_duplicates: bool = True) -> pd.DataFrame:
        """
        Runs BDE prediction for a list of molecules in batch.

        Args:
            smiles_list (List[str]): A list of SMILES strings.
            drop_duplicates (bool): If True, remove predictions for bonds that
                                    result in the same set of fragments.

        Returns:
            pd.DataFrame: A concatenated DataFrame with predictions for all molecules.
        """
        # 1. Generate fragment info for all molecules in one go
        # This function is designed to handle a list of SMILES, reducing tqdm noise.
        all_fragments_df = generate_fragment_template(smiles_list)

        if all_fragments_df.empty:
            logger.info("No valid bonds found for prediction in the provided molecules.")
            return pd.DataFrame()

        # Get unique canonical SMILES from the generated fragments
        canonical_smiles_processed = all_fragments_df['molecule'].unique().tolist()
        
        all_data_list = []
        for canonical_smiles in canonical_smiles_processed:
            mol = Chem.MolFromSmiles(canonical_smiles)
            if mol is None:
                logger.warning(f"Failed to parse canonical SMILES: {canonical_smiles}. Skipping.")
                continue
            mol = Chem.AddHs(mol)
            
            # Use the abstract featurizer
            data = self.featurizer.featurize(mol, smiles=canonical_smiles)
            if data is None:
                logger.warning(f"Featurization failed for canonical SMILES: {canonical_smiles}. Skipping.")
                continue
            data.original_input_smiles = canonical_smiles # Ensure this is consistent with 'molecule' in df
            all_data_list.append(data)

        if not all_data_list:
            return pd.DataFrame()

        # 2. Batch featurized molecules and run model inference (single call)
        batch = Batch.from_data_list(all_data_list).to(self.device)
        
        with torch.no_grad():
            raw_predictions = self.model(batch) # Tensor of predictions for all edges in batch

        # 3. Map predictions back to original molecule and bond indices
        preds_records = []
        start_edge_idx = 0
        for data in all_data_list: # Iterate over the Data objects that were successfully featurized
            num_edges = data.edge_index.size(1) # Number of edges for this specific graph
            end_edge_idx = start_edge_idx + num_edges

            # Extract raw predictions for the current molecule's graph (shape: [num_edges, num_tasks])
            graph_preds = raw_predictions[start_edge_idx:end_edge_idx]
            # Retrieve the bond_indices_map created during featurization for this graph
            graph_bond_indices = data.bond_indices_map.cpu().numpy()

            # Create a temporary DataFrame to associate predictions with their original RDKit bond indices
            df_data = {
                'molecule': data.original_input_smiles, 
                'bond_index': graph_bond_indices,
                'is_valid': data.is_valid.item()
            }
            
            # Unpack each task into its own column
            np_preds = graph_preds.cpu().numpy()
            pred_col_names = []
            for i, target_name in enumerate(self.target_columns):
                col_name = f"{target_name}_pred"
                # If model hasn't been upgraded properly (edge case, old model file), handle gracefully
                val_arr = np_preds[:, i] if len(np_preds.shape) > 1 else np_preds
                df_data[col_name] = val_arr
                pred_col_names.append(col_name)

            graph_preds_df = pd.DataFrame(df_data)
            
            # Group by molecule and bond_index to average predictions for the two directed edges
            bde_preds_by_bond = graph_preds_df.groupby(['molecule', 'bond_index'])[pred_col_names + ['is_valid']].mean().reset_index()
            preds_records.append(bde_preds_by_bond)
            
            start_edge_idx = end_edge_idx
            
        final_bde_preds = pd.concat(preds_records, ignore_index=True) if preds_records else pd.DataFrame()

        # 4. Merge the averaged BDE predictions with the detailed fragment information
        # Merge with the single all_fragments_df
        result_df = pd.merge(all_fragments_df, final_bde_preds, on=['molecule', 'bond_index'], how='left')

        # Drop the now-redundant original target columns from the template if they exist
        for target in self.target_columns:
            if target in result_df.columns:
                result_df = result_df.drop(columns=[target])

        if drop_duplicates:
            # Sort fragments within each row to create a canonical key for deduplication
            fragments = result_df[['fragment1', 'fragment2']].values
            canonical_frag_pairs = [tuple(sorted(f)) for f in fragments]
            
            # Add a temporary column for deduplication
            temp_dedup_df = result_df.copy()
            temp_dedup_df['canonical_frag_pair'] = canonical_frag_pairs
            
            result_df = temp_dedup_df.drop_duplicates(subset=['molecule', 'canonical_frag_pair']).drop(columns=['canonical_frag_pair'])
            result_df = result_df.reset_index(drop=True)

        return result_df.reset_index(drop=True)

    @classmethod
    def from_run_dir(cls, run_dir, device='cpu'):
        """
        工廠方法：從訓練目錄自動初始化單一模型預測器。
        """
        config_path = os.path.join(run_dir, 'config.yaml')
        vocab_path = os.path.join(run_dir, 'vocab.json')
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"找不到設定檔: {config_path}")
            
        cfg = load_config(config_path) #
        model_path = os.path.join(run_dir, cfg.train.model_save_path) #
        
        return cls(
            model_path=model_path,
            vocab_path=vocab_path,
            featurizer_type=cfg.data.featurizer_type,
            atom_features=cfg.model.atom_features,
            num_messages=cfg.model.num_messages,
            num_tasks=cfg.model.num_tasks,
            target_columns=cfg.data.target_columns,
            device=device
        )


class EnsemblePredictor:
    """Manages multiple Predictor instances for ensemble inference.

    Wraps a list of ``Predictor`` objects and aggregates their predictions by
    computing per-bond mean and standard deviation.  The output interface is
    **always identical** regardless of how many models are held:

    * ``{task}_pred_mean`` — mean across all models (best estimate)
    * ``{task}_pred_std``  — std across all models (uncertainty proxy)

    When only one model is present ``pred_std`` is uniformly ``0.0``.

    The canonical way to create an instance for offline inference is via
    ``from_run_dir``, which auto-detects ``fold_*/`` sub-directories produced
    by K-Fold or LOO cross-validation and falls back to single-model mode when
    none are found.

    Attributes:
        predictors: Ordered list of loaded ``Predictor`` instances.
    """

    def __init__(self, predictors: List["Predictor"]) -> None:
        """Initialise with a pre-built list of predictors.

        Args:
            predictors: One or more loaded ``Predictor`` instances.
                        All predictors must share the same ``target_columns``.

        Raises:
            ValueError: If ``predictors`` is empty.
        """
        if not predictors:
            raise ValueError("EnsemblePredictor requires at least one Predictor.")
        self.predictors = predictors

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def target_columns(self) -> List[str]:
        """Target column names derived from the first underlying predictor."""
        return self.predictors[0].target_columns

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_run_dir(cls, run_dir: str, device: str = "cpu") -> "EnsemblePredictor":
        """Factory method: load predictor(s) from a training run directory.

        Scans ``run_dir`` for ``fold_*/`` sub-directories (produced by K-Fold
        or LOO cross-validation).  If found, one ``Predictor`` is loaded per
        fold and ensemble mode is activated.  If no fold directories exist, a
        single model is loaded from ``run_dir`` directly (single-model mode).

        ``config.yaml`` and ``vocab.json`` are always read from ``run_dir``
        (the run root); model checkpoints are found inside each ``fold_*/``
        sub-directory or directly in ``run_dir`` for non-CV runs.

        Args:
            run_dir: Root directory of a completed training run.
            device:  Torch device string (e.g. ``'cpu'`` or ``'cuda'``).

        Returns:
            An ``EnsemblePredictor`` ready for ``.predict()`` calls.

        Raises:
            FileNotFoundError: If ``config.yaml`` or ``vocab.json`` is missing.
            RuntimeError: If fold directories exist but no valid checkpoints
                are found inside them.
        """
        config_path = os.path.join(run_dir, "config.yaml")
        vocab_path = os.path.join(run_dir, "vocab.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Training config not found: {config_path}")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

        cfg = load_config(config_path)

        # Auto-detect fold sub-directories (fold_0, fold_1, …)
        fold_dirs = sorted(glob(os.path.join(run_dir, "fold_*")))

        if fold_dirs:
            logger.info(
                "EnsemblePredictor: detected %d fold(s) in '%s' — ensemble mode.",
                len(fold_dirs), run_dir,
            )
            predictors: List["Predictor"] = []
            for fold_dir in fold_dirs:
                model_path = os.path.join(fold_dir, cfg.train.model_save_path)
                if not os.path.exists(model_path):
                    logger.warning(
                        "EnsemblePredictor: model not found in '%s' — skipping.", fold_dir
                    )
                    continue
                predictors.append(
                    Predictor(
                        model_path=model_path,
                        vocab_path=vocab_path,
                        featurizer_type=cfg.data.featurizer_type,
                        atom_features=cfg.model.atom_features,
                        num_messages=cfg.model.num_messages,
                        num_tasks=cfg.model.num_tasks,
                        target_columns=cfg.data.target_columns,
                        device=device,
                    )
                )
            if not predictors:
                raise RuntimeError(
                    f"EnsemblePredictor: fold directories found but no valid model "
                    f"checkpoints could be loaded from '{run_dir}'."
                )
        else:
            logger.info(
                "EnsemblePredictor: no fold directories found in '%s' — single-model mode.",
                run_dir,
            )
            model_path = os.path.join(run_dir, cfg.train.model_save_path)
            predictors = [
                Predictor(
                    model_path=model_path,
                    vocab_path=vocab_path,
                    featurizer_type=cfg.data.featurizer_type,
                    atom_features=cfg.model.atom_features,
                    num_messages=cfg.model.num_messages,
                    num_tasks=cfg.model.num_tasks,
                    target_columns=cfg.data.target_columns,
                    device=device,
                )
            ]

        return cls(predictors)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        smiles_list: List[str],
        drop_duplicates: bool = True,
    ) -> pd.DataFrame:
        """Run ensemble prediction and return mean + uncertainty estimates.

        Each underlying ``Predictor`` predicts independently on ``smiles_list``.
        Predictions are aligned by ``(molecule, bond_index)`` key and aggregated:

        * ``{task}_pred_mean`` — mean across all models (best estimate)
        * ``{task}_pred_std``  — std across all models (uncertainty proxy)

        Fragment structural columns (``molecule``, ``bond_index``, ``fragment1``,
        ``fragment2``, ``is_valid``, …) are preserved from the first successful
        predictor.  When only one model is loaded, ``pred_std`` is ``0.0``.

        Args:
            smiles_list:     List of SMILES strings to predict on.
            drop_duplicates: If ``True``, bonds producing identical fragment
                             pairs are deduplicated (first occurrence kept).

        Returns:
            DataFrame containing fragment structural columns plus
            ``{task}_pred_mean`` and ``{task}_pred_std`` for every task.
            Returns an empty DataFrame if all predictors fail.
        """
        pred_cols = [f"{t}_pred" for t in self.target_columns]
        key_cols = ["molecule", "bond_index"]

        structural_df: Optional[pd.DataFrame] = None   # fragment info reference
        all_pred_dfs: List[pd.DataFrame] = []

        for i, predictor in enumerate(self.predictors):
            try:
                # Collect all bonds without deduplication for consistent alignment;
                # deduplication is applied once at the end if requested.
                pred_df = predictor.predict(smiles_list, drop_duplicates=False)
            except Exception as exc:
                logger.warning(
                    "EnsemblePredictor: predictor %d failed — skipping. Error: %s",
                    i, exc,
                )
                continue

            if pred_df.empty:
                logger.warning(
                    "EnsemblePredictor: predictor %d returned empty predictions.", i
                )
                continue

            if structural_df is None:
                # Capture structural / fragment columns from the first success
                non_pred_cols = [c for c in pred_df.columns if c not in pred_cols]
                structural_df = pred_df[non_pred_cols].copy()

            all_pred_dfs.append(pred_df)

        if structural_df is None or not all_pred_dfs:
            logger.warning("EnsemblePredictor: no valid predictions from any model.")
            return pd.DataFrame()

        # ── Stack predictions and compute mean / std ───────────────────────
        result_df = structural_df.copy()
        for col in pred_cols:
            stacked_arrays: List[np.ndarray] = []
            for pred_df in all_pred_dfs:
                # Align each model's predictions to the reference row order
                aligned = pd.merge(
                    structural_df[key_cols],
                    pred_df[key_cols + [col]],
                    on=key_cols,
                    how="left",
                )
                stacked_arrays.append(aligned[col].values.astype(float))

            arr = np.vstack(stacked_arrays)                   # [n_models, n_bonds]
            task_name = col[: -len("_pred")]                  # strip '_pred' suffix
            result_df[f"{task_name}_pred_mean"] = arr.mean(axis=0)
            result_df[f"{task_name}_pred_std"]  = arr.std(axis=0)

        # ── Optional deduplication (mirrors Predictor.predict logic) ──────
        if (
            drop_duplicates
            and "fragment1" in result_df.columns
            and "fragment2" in result_df.columns
        ):
            canonical_pairs = [
                tuple(sorted(pair))
                for pair in result_df[["fragment1", "fragment2"]].values
            ]
            result_df = result_df.copy()
            result_df["_canonical_frag_pair"] = canonical_pairs
            result_df = result_df.drop_duplicates(
                subset=["molecule", "_canonical_frag_pair"]
            ).drop(columns=["_canonical_frag_pair"])

        return result_df.reset_index(drop=True)


def get_bde_predictions(
    run_dir: str,
    smiles: Union[str, List[str]],
    device: str = "cpu",
    **kwargs,
) -> pd.DataFrame:
    """Convenience function: load ensemble from ``run_dir`` and predict.

    Delegates to ``EnsemblePredictor.from_run_dir``.  Auto-detects fold
    sub-directories and enables ensemble mode (mean + std) when present;
    falls back to single-model mode otherwise.

    Args:
        run_dir: Root directory of a completed training run.
        smiles:  A single SMILES string or a list of SMILES strings.
        device:  Torch device string (``'cpu'`` or ``'cuda'``).
        **kwargs: Additional keyword arguments forwarded to
                  ``EnsemblePredictor.predict`` (e.g. ``drop_duplicates``).

    Returns:
        DataFrame with ``{task}_pred_mean`` and ``{task}_pred_std`` columns
        plus structural fragment information.
    """
    ensemble = EnsemblePredictor.from_run_dir(run_dir, device=device)
    smiles_list = [smiles] if isinstance(smiles, str) else smiles
    return ensemble.predict(smiles_list, **kwargs)


def get_bde_predictions_with_embeddings(
    smiles: Union[str, List[str]],
    model_path: str,
    vocab_path: str,
    featurizer_type: str = 'TokenFeaturizer',
    atom_features: int = 128,
    num_messages: int = 6,
    num_tasks: int = 1,
    target_columns: List[str] = ['bde'],
    device: str = 'cpu'
) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
    """
    Gets BDE predictions and extracts intermediate, averaged bond embeddings from each MPNN layer.

    Args:
        smiles (Union[str, List[str]]): A single SMILES string or a list of SMILES strings.
        model_path (str): Path to the trained model checkpoint (.pt file).
        vocab_path (str): Path to the vocabulary file (.json) used during training.
        featurizer_type (str): Type of featurizer to use.
        num_messages (int): Number of message passing layers used in the model.
        device (str, optional): The device to run inference on.

    Returns:
        Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
            - A DataFrame containing final predictions and fragment info.
            - A dictionary where keys are layer indices and values are DataFrames
              containing ('molecule', 'bond_index', 'embedding') for that layer,
              averaged over directed edges.
    """
    if isinstance(smiles, str):
        smiles_list = [smiles]
    else:
        smiles_list = smiles

    try:
        predictor = Predictor(
            model_path=model_path,
            vocab_path=vocab_path,
            featurizer_type=featurizer_type,
            atom_features=atom_features,
            num_messages=num_messages,
            num_tasks=num_tasks,
            target_columns=target_columns,
            device=device
        )
        
        all_fragments_df = generate_fragment_template(smiles_list)
        if all_fragments_df.empty:
            logger.info("No valid bonds found for prediction in the provided molecules.")
            return pd.DataFrame(), {}
            
        canonical_smiles_processed = all_fragments_df['molecule'].unique().tolist()
        all_data_list = []
        for canonical_smiles in canonical_smiles_processed:
            mol = Chem.MolFromSmiles(canonical_smiles)
            if mol is None:
                logger.warning(f"Failed to parse canonical SMILES: {canonical_smiles}. Skipping.")
                continue
            mol = Chem.AddHs(mol)
            data = predictor.featurizer.featurize(mol, smiles=canonical_smiles)
            if data:
                data.original_input_smiles = canonical_smiles
                all_data_list.append(data)

        if not all_data_list:
            logger.info("No featurized data objects generated for prediction.")
            return pd.DataFrame(), {}
            
        batch = Batch.from_data_list(all_data_list).to(predictor.device)

        bond_embeddings_by_layer = {}
        hooks = []

        def get_bond_embeddings_hook(layer_idx):
            def hook(module, input, output):
                bond_embeddings_by_layer[layer_idx] = output[1].detach().cpu()
            return hook

        for i, layer in enumerate(predictor.model.interaction_layers):
            hooks.append(layer.register_forward_hook(get_bond_embeddings_hook(i)))

        with torch.no_grad():
            final_predictions = predictor.model(batch)

        for hook in hooks:
            hook.remove()

        # --- Process Final Predictions and Embeddings ---
        
        # Process final BDE predictions
        final_bde_preds_list = []
        start_edge_idx = 0
        for data in all_data_list:
            num_edges = data.edge_index.size(1)
            end_edge_idx = start_edge_idx + num_edges
            
            graph_preds_df = pd.DataFrame({
                'molecule': data.original_input_smiles,
                'bond_index': data.bond_indices_map.cpu().numpy()
            })
            
            np_preds = final_predictions[start_edge_idx:end_edge_idx].cpu().numpy()
            pred_col_names = []
            for i, target_name in enumerate(target_columns):
                col_name = f"{target_name}_pred"
                val_arr = np_preds[:, i] if len(np_preds.shape) > 1 else np_preds
                graph_preds_df[col_name] = val_arr
                pred_col_names.append(col_name)

            bde_preds_by_bond = graph_preds_df.groupby(['molecule', 'bond_index'])[pred_col_names].mean().reset_index()
            final_bde_preds_list.append(bde_preds_by_bond)
            
            start_edge_idx = end_edge_idx

        final_bde_preds_df = pd.concat(final_bde_preds_list, ignore_index=True) if final_bde_preds_list else pd.DataFrame()
        results_df = pd.merge(all_fragments_df, final_bde_preds_df, on=['molecule', 'bond_index'], how='left')

        # Process intermediate layer embeddings
        processed_embeddings = {}
        for layer_idx, embeddings_tensor in bond_embeddings_by_layer.items():
            layer_embeds_list = []
            start_edge_idx = 0
            for data in all_data_list:
                num_edges = data.edge_index.size(1)
                end_edge_idx = start_edge_idx + num_edges
                
                graph_embeds_df = pd.DataFrame({
                    'molecule': data.original_input_smiles,
                    'bond_index': data.bond_indices_map.cpu().numpy(),
                    'embedding': list(embeddings_tensor[start_edge_idx:end_edge_idx].numpy())
                })

                # Average embeddings for each bond
                avg_embeds = graph_embeds_df.groupby(['molecule', 'bond_index'])['embedding'].apply(lambda x: np.mean(np.stack(x), axis=0)).reset_index()
                layer_embeds_list.append(avg_embeds)
                
                start_edge_idx = end_edge_idx
            
            processed_embeddings[layer_idx] = pd.concat(layer_embeds_list, ignore_index=True) if layer_embeds_list else pd.DataFrame()

        return results_df, processed_embeddings

    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}", exc_info=True)
        return pd.DataFrame(), {}

