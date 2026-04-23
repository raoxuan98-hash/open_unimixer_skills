#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for HeteroAttention model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch
from collections import OrderedDict
from fuxictr.features import FeatureMap
from HeteroAttention import SwishGLU, HeteroSelfAttention, HeteroTransformerEncoderLayer, HeteroAttention


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


def test_swish_glu():
    print("=" * 60)
    print("Test 1: SwishGLU")
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


def test_hetero_self_attention():
    print()
    print("=" * 60)
    print("Test 2: HeteroSelfAttention")
    print("=" * 60)
    B, T, D, nh = 4, 5, 8, 2
    att = HeteroSelfAttention(seq_len=T, d_model=D, num_heads=nh, dropout=0.0)
    X = torch.randn(B, T, D, requires_grad=True)
    out = att(X)
    assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"
    loss = out.sum()
    loss.backward()
    assert X.grad is not None, "HeteroSelfAttention grad error"
    print(f"  Input shape:  {tuple(X.shape)}")
    print(f"  Output shape: {tuple(out.shape)}")
    print("  Gradients OK")


def test_hetero_transformer_encoder_layer():
    print()
    print("=" * 60)
    print("Test 3: HeteroTransformerEncoderLayer")
    print("=" * 60)
    B, T, D, nh = 4, 5, 8, 2
    layer = HeteroTransformerEncoderLayer(
        seq_len=T, d_model=D, num_heads=nh, dim_feedforward=16, dropout=0.0
    )
    X = torch.randn(B, T, D, requires_grad=True)
    out = layer(X)
    assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"
    loss = out.sum()
    loss.backward()
    assert X.grad is not None, "HeteroTransformerEncoderLayer grad error"
    print(f"  Input shape:  {tuple(X.shape)}")
    print(f"  Output shape: {tuple(out.shape)}")
    print("  Gradients OK")


def test_hetero_attention_model_cls():
    print()
    print("=" * 60)
    print("Test 4: HeteroAttention Model (use_cls_token=True)")
    print("=" * 60)
    feature_map = create_dummy_feature_map(num_fields=4)
    B = 4
    model = HeteroAttention(
        feature_map=feature_map,
        model_id="HeteroAttention_test1",
        gpu=-1,
        learning_rate=1e-3,
        embedding_dim=16,
        num_transformer_layers=2,
        num_heads=4,
        att_emb_size=None,
        transformer_dropout=0.0,
        ffn_dim=32,
        use_cls_token=True,
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


def test_hetero_attention_model_no_cls():
    print()
    print("=" * 60)
    print("Test 5: HeteroAttention Model (use_cls_token=False)")
    print("=" * 60)
    feature_map = create_dummy_feature_map(num_fields=4)
    B = 4
    model = HeteroAttention(
        feature_map=feature_map,
        model_id="HeteroAttention_test2",
        gpu=-1,
        learning_rate=1e-3,
        embedding_dim=16,
        num_transformer_layers=1,
        num_heads=4,
        att_emb_size=4,
        transformer_dropout=0.0,
        ffn_dim=32,
        use_cls_token=False,
        use_pos_embedding=False,
        output_mlp_hidden_units=[16],
        net_dropout=0.0,
        batch_norm=True,
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
    print(f"  y_pred shape: {tuple(y_pred.shape)}")
    print("  Forward pass OK (no cls token)")


def test_model_zoo_import():
    print()
    print("=" * 60)
    print("Test 6: Model Zoo Import")
    print("=" * 60)
    import model_zoo
    assert hasattr(model_zoo, "HeteroAttention"), "HeteroAttention not found in model_zoo"
    print("  HeteroAttention successfully registered in model_zoo")


if __name__ == "__main__":
    test_swish_glu()
    test_hetero_self_attention()
    test_hetero_transformer_encoder_layer()
    test_hetero_attention_model_cls()
    test_hetero_attention_model_no_cls()
    test_model_zoo_import()
    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
