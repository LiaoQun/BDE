import tempfile
import pytest
import torch
from torch_geometric.loader import DataLoader

from src.features import get_featurizer
from src.data.dataset import BDEDataset
from src.models.mpnn import BDEInteractionLayer, BDEModel

ATOM_FEATURES = 128


@pytest.fixture
def mock_smiles_data():
    return [
        ("CCO", {(0, 1): [75.0], (1, 2): [80.0]}),
        ("C=O", {(0, 1): [120.0]}),
    ]


@pytest.fixture
def featurizer_token(mock_smiles_data, tmp_path):
    save_path = str(tmp_path / "vocab.json")
    smiles_list = [smi for smi, _ in mock_smiles_data]
    return get_featurizer(
        featurizer_type="TokenFeaturizer",
        smiles_list=smiles_list,
        save_path=save_path,
    )


@pytest.fixture
def featurizer_chemprop(mock_smiles_data, tmp_path):
    save_path = str(tmp_path / "vocab_chemprop.json")
    smiles_list = [smi for smi, _ in mock_smiles_data]
    return get_featurizer(
        featurizer_type="ChemPropFeaturizer",
        smiles_list=smiles_list,
        save_path=save_path,
    )


@pytest.fixture
def sample_batch_token(mock_smiles_data, featurizer_token, tmp_path):
    dataset = BDEDataset(
        root=str(tmp_path / "token"),
        smiles_data=mock_smiles_data,
        featurizer=featurizer_token,
    )
    dataloader = DataLoader(dataset, batch_size=2)
    return next(iter(dataloader)), featurizer_token


@pytest.fixture
def sample_batch_chemprop(mock_smiles_data, featurizer_chemprop, tmp_path):
    dataset = BDEDataset(
        root=str(tmp_path / "chemprop"),
        smiles_data=mock_smiles_data,
        featurizer=featurizer_chemprop,
    )
    dataloader = DataLoader(dataset, batch_size=2)
    return next(iter(dataloader)), featurizer_chemprop


def test_bde_interaction_layer_shape(sample_batch_token):
    """Tests shape consistency of BDEInteractionLayer."""
    batch, featurizer = sample_batch_token
    layer = BDEInteractionLayer(atom_features=ATOM_FEATURES)

    atom_state = torch.rand(batch.x.shape[0], ATOM_FEATURES)
    bond_state = torch.rand(batch.edge_attr.shape[0], ATOM_FEATURES)

    new_atom_state, new_bond_state = layer(atom_state, batch.edge_index, bond_state)

    assert new_atom_state.shape == (batch.x.shape[0], ATOM_FEATURES)
    assert new_bond_state.shape == (batch.edge_attr.shape[0], ATOM_FEATURES)


@pytest.mark.parametrize("input_type", ["token", "chemprop"])
def test_bde_model_forward_pass_shape(input_type, request):
    """Tests forward pass shape for both discrete and continuous inputs."""
    batch, featurizer = request.getfixturevalue(f"sample_batch_{input_type}")

    model = BDEModel(
        atom_input_dim=featurizer.atom_dim,
        bond_input_dim=featurizer.bond_dim,
        atom_features=ATOM_FEATURES,
        num_messages=3,
        inputs_are_discrete=featurizer.is_discrete,
        num_tasks=1,
    )

    output = model(batch)

    # output shape: [num_edges, num_tasks]
    assert output.shape == (batch.edge_index.shape[1], 1)