# BDE Prediction Model

This project contains a Graph Neural Network (GNN) model for predicting Bond Dissociation Energy (BDE), refactored into a PyTorch + PyTorch Geometric framework.

## How to Train

The entire training process is controlled via a centralized configuration file and launched from a single script. All outputs are automatically saved to a unique, timestamped directory for reproducibility.

### Step 1: Prepare Training Data

The model requires a CSV file (can be `.csv` or `.csv.gz`) with at least the following three columns:

-   `molecule`: The SMILES string representation of the molecule.
-   `bond_index`: The RDKit integer index of the bond you are labeling.
-   `bde`: The ground-truth Bond Dissociation Energy value for that bond.

To generate a template for your own molecules, you can use the `create_training_template.py` script:

```bash
# Example: Generate a template for ethane (CC) and methyl mercaptan (CS)
python scripts/create_training_template.py --smiles "CC" "CS" --output_path "data/my_new_data.csv"
```
After running, open the generated CSV file and fill in the `bde` column for the bonds you have data for.

### Step 2: Edit the Configuration File (`config.json`)

Open the `config.json` file in the root directory to define your experiment.

```json
{
  "data": {
    // 1. Specify the path(s) to your data file(s)
    "data_paths": [
      "data/my_new_data.csv"
    ],
    // 2. Set dataset split ratios
    "test_size": 0.1, // 10% for the test set
    "val_size": 0.1,  // 10% for the validation set
    // 3. (Optional) For quick tests, use a fraction of the data
    "sample_percentage": 1.0 // 1.0 uses 100% of the data
  },
  "model": {
    // Model architecture hyperparameters
    "atom_features": 128,
    "num_messages": 6
  },
  "train": {
    // 4. Training process hyperparameters
    "device": "cuda", // "cuda" to use GPU, "cpu" for CPU
    "epochs": 300,    // Total number of training epochs
    "lr": 0.001,      // Learning rate
    "batch_size": 64, // Adjust based on your GPU memory
    "model_save_path": "bde_model.pt", // Filename for the saved model
    "output_dir": "training_runs"      // Main directory for all outputs
  }
}
```

### Step 3: Execute the Training Script

Once the configuration is set, run `main.py` from your terminal:

```bash
# Ensure your Python/Conda environment is activated
python main.py
```

If you wish to use a different configuration file, you can specify its path:
```bash
python main.py --config_path "experiments/config_01.json"
```

### Step 4: Review Results

All artifacts from the run will be saved in a unique, timestamped directory inside `training_runs/` (e.g., `training_runs/20260106_...`).

Inside this directory, you will find:
-   **`bde_model.pt`**: The best-performing model weights from the training run.
-   **`config.json`**: A snapshot of the configuration used for this run.
-   **`training_log.csv`**: Epoch-by-epoch training and validation loss data.
-   **`training_curve.png`**: A plot of the loss curve.
-   **`parity_plot.png`**: A parity plot of predicted vs. true BDE values for evaluating model performance.
-   **`predictions_*.csv`**: Detailed prediction results for the train, validation, and test sets.
