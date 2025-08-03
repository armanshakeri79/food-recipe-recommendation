import pandas as pd
import torch
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from graph_generation.graph_dataset_generator import IdentityEncoder, ListEncoder


def test_identity_encoder_handles_optional_dtype():
    s = pd.Series([1, 2, 3])
    enc = IdentityEncoder()
    tensor = enc(s)
    assert tensor.shape == (3, 1)
    assert tensor.dtype == torch.int64


def test_list_encoder_handles_optional_dtype():
    s = pd.Series([[1, 2], [3, 4]])
    enc = ListEncoder()
    tensor = enc(s)
    assert tensor.shape == (2, 2)
    assert tensor.dtype == torch.int64
