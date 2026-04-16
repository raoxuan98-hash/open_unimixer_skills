#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for TokenMixerLarge model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch
from collections import OrderedDict
from fuxictr.features import FeatureMap
from TokenMixerLarge import SwishGLU, TokenSpecificSwishGLU, TokenMixerLargeLayer, TokenMixerLarge


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


def test_token_specific_swish_glu():
    print()
    print("=" * 60)
    print("Test 2: TokenSpecificSwishGLU")
    print("=" * 60)
    B, L, D, d_ff = 4, 4, 16, 32
    glu = TokenSpecificSwishGLU(seq_len=L, d_model=D, d_ff=d_ff, dropout=0.0)
    x = torch.randn(B, L, D, requires_grad=True)
    out = glu(x)
    assert out.shape == (B, L, D), f"Expected {(B, L, D)}, got {out.shape}"
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "TokenSpecificSwishGLU grad error"
    print(f"  Input shape:  {tuple(x.shape)}")
    print(f"  Output shape: {tuple(out.shape)}")
    print("  Gradients OK")


def test_token_mixer_large_layer():
    print()
    print("=" * 60)
    print("Test 3: TokenMixerLargeLayer")
    print("=" * 60)
    B, L, D = 4, 4, 16
    layer = TokenMixerLargeLayer(seq_len=L, d_model=D, d_ff=32, dropout=0.0)
    x = torch.randn(B, L, D, requires_grad=True)
    out = layer(x)
    assert out.shape == (B, L, D), f"Expected {(B, L, D)}, got {out.shape}"
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "TokenMixerLargeLayer grad error"
    # Verify alpha parameters exist
    assert hasattr(layer, 'alpha1') and hasattr(layer, 'alpha2')
    print(f"  Input shape:  {tuple(x.shape)}")
    print(f"  Output shape: {tuple(out.shape)}")
    print(f"  alpha1 softplus: {torch.nn.functional.softplus(layer.alpha1).mean().item():.4f}")
    print("  Gradients OK")


def test_token_mixer_large_model():
    print()
    print("=" * 60)
    print("Test 4: TokenMixerLarge Model")
    print("=" * 60)
    num_fields = 4
    feature_map = create_dummy_feature_map(num_fields=num_fields)
    B = 4
    model = TokenMixerLarge(
        feature_map=feature_map,
        model_id="TokenMixerLarge_test",
        gpu=-1,
        learning_rate=1e-3,
        embedding_dim=16,
        num_transformer_layers=2,
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
    print("Test 5: Model Zoo Import")
    print("=" * 60)
    import model_zoo
    assert hasattr(model_zoo, "TokenMixerLarge"), "TokenMixerLarge not found in model_zoo"
    print("  TokenMixerLarge successfully registered in model_zoo")


if __name__ == "__main__":
    test_swish_glu()
    test_token_specific_swish_glu()
    test_token_mixer_large_layer()
    test_token_mixer_large_model()
    test_model_zoo_import()
    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
