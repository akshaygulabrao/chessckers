"""Export a ChesskersScorer state_dict to a flat binary the native C++ engine
loads (the `akshay-chessckers-0` fork's `nn.hpp`). Keeps the inference weights portable and dependency-free
— no protobuf/ONNX/safetensors, just length-prefixed float32 tensors.

Format (little-endian):
    int32  n_tensors
    repeat n_tensors:
        int32  name_len
        bytes  name (utf-8)
        int32  ndim
        int32  dims[ndim]
        float32 data[prod(dims)]   # row-major (PyTorch contiguous)
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Mapping

import torch


def export_state_dict(state_dict: Mapping[str, torch.Tensor], path: str | Path) -> None:
    items = list(state_dict.items())
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(items)))
        for name, tensor in items:
            t = tensor.detach().cpu().contiguous().float()
            nb = name.encode("utf-8")
            f.write(struct.pack("<i", len(nb)))
            f.write(nb)
            f.write(struct.pack("<i", t.dim()))
            for d in t.shape:
                f.write(struct.pack("<i", int(d)))
            f.write(t.numpy().tobytes())
