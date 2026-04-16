#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for FAT model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch
from collections import OrderedDict
from fuxictr.features import FeatureMap
from FAT import log_sinkhorn, SwishGLU, FATAttention, FATEncoderLayer, FAT


def create_dummy_feature_map(num_fields=4):
    feature_map = FeatureMap(dataset_id='test', data_dir='.')
    feature_map.num_fields = num_fields
    feature_map.total_features = num_fields * 10
    feature_map.input_length = num_fields
    feature_map.labels = ['label']
    feature_map.features = OrderedDict([
        (f'f{i}', {'type': 'categorical', 'source': '', 'vocab_size': 10})
        for i in range(1, num_fields + 1)
    ])
    feature_map.feature_specs = feature_map.features
    return feature_map


def test_log_sinkhorn():
    print("=" * 60)
    print("Test 1: log_sinkhorn")
    print("=" * 60)
    x = torch.randn(2, 3, 3)
    out = log_sinkhorn(x, n_iters=10, temperature=1.0)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    # Check that rows and columns sum to ~1
    row_sums = out.sum(dim=-1)
    col_sums = out.sum(dim=-2)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=5e-2), "Row sums not close to 1"
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=5e-2), "Col sums not close to 1"
    print(f"  Input shape:  {tuple(x.shape)}")
    print(f"  Output shape: {tuple(out.shape)}")
    print("  Doubly stochastic check passed")


def test_swish_glu():
    print()
    print("=" * 60)
    print("Test 2: SwishGLU")
    print("=" * 60)
    B, D = 4, 8
    glu = SwishGLU(input_dim=D, hidden_dim=16, output_dim=D, dropout=0.0)
    x = torch.randn(B, D, requires_grad=True)
    out = glu(x)
    assert out.shape == (B, D), f"Expected {(B, D)}, got {out.shape}"
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "SwishGLU grad error"
    print(f"  Input shape:  {tuple(x.shape)}")
    print(f"  Output shape: {tuple(out.shape)}")
    print("  Gradients OK")


def test_fat_attention():
    print()
    print("=" * 60)
    print("Test 3: FATAttention")
    print("=" * 60)
    B, T, D, nh = 4, 4, 16, 4
    att = FATAttention(seq_len=T, d_model=D, num_heads=nh, basis_num=4, att_emb_size=None,
                       sinkhorn_iters=5, dropout=0.0)
    X = torch.randn(B, T, D, requires_grad=True)
    out = att(X)
    assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"
    loss = out.sum()
    loss.backward()
    assert X.grad is not None, "FATAttention grad error"
    print(f"  Input shape:  {tuple(X.shape)}")
    print(f"  Output shape: {tuple(out.shape)}")
    print("  Gradients OK")


def test_fat_encoder_layer():
    print()
    print("=" * 60)
    print("Test 4: FATEncoderLayer")
    print("=" * 60)
    B, T, D, nh = 4, 4, 16, 4
    layer = FATEncoderLayer(seq_len=T, d_model=D, num_heads=nh, basis_num=4,
                            sinkhorn_iters=5, dim_feedforward=32, dropout=0.0)
    X = torch.randn(B, T, D, requires_grad=True)
    out = layer(X)
    assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"
    loss = out.sum()
    loss.backward()
    assert X.grad is not None, "FATEncoderLayer grad error"
    print(f"  Input shape:  {tuple(X.shape)}")
    print(f"  Output shape: {tuple(out.shape)}")
    print("  Gradients OK")


def test_fat_model():
    print()
    print("=" * 60)
    print("Test 5: FAT Model")
    print("=" * 60)
    num_fields = 4
    feature_map = create_dummy_feature_map(num_fields=num_fields)
    B = 4
    model = FAT(
        feature_map=feature_map,
        model_id="FAT_test",
        gpu=-1,
        learning_rate=1e-3,
        embedding_dim=16,
        num_transformer_layers=2,
        num_heads=4,
        basis_num=4,
        sinkhorn_iters=5,
        transformer_dropout=0.0,
        ffn_dim=32,
        use_pos_embedding=True,
        output_mlp_hidden_units=[32, 16],
        net_dropout=0.0,
        batch_norm=False,
        embedding_regularizer=None,
        net_regularizer=None,
        optimizer="adam",
        loss="binary_crossentropy",
        verbose=0,
        model_root="./checkpoints",
        metrics=["AUC"],
    )
    inputs = {f: torch.randint(0, 10, (B,)) for f in feature_map.features.keys()}
    inputs["label"] = torch.ones(B, 1)
    out_dict = model(inputs)
    y_pred = out_dict["y_pred"]
    assert y_pred.shape == (B, 1), f"Expected y_pred {(B, 1)}, got {y_pred.shape}"
    loss = model.loss_fn(y_pred, torch.ones(B, 1))
    loss.backward()
    print(f"  Batch size: {B}, num_fields: {feature_map.num_fields}")
    print(f"  y_pred shape: {tuple(y_pred.shape)}")
    print(f"  Loss: {loss.item():.4f}")
    print("  Gradients OK")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")


def test_model_zoo_import():
    print()
    print("=" * 60)
    print("Test 6: Model Zoo Import")
    print("=" * 60)
    import model_zoo
    assert hasattr(model_zoo, "FAT"), "FAT not found in model_zoo"
    print("  FAT successfully registered in model_zoo")


if __name__ == "__main__":
    test_log_sinkhorn()
    test_swish_glu()
    test_fat_attention()
    test_fat_encoder_layer()
    test_fat_model()
    test_model_zoo_import()
    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
