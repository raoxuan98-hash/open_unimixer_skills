#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for UniMixerLite model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch
from collections import OrderedDict
from fuxictr.features import FeatureMap
from UniMixerLite import log_sinkhorn, TokenSpecificSwishGLU, KroneckerMixer, UniMixerLiteLayer, UniMixerLite


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
    x = torch.randn(2, 4, 4)
    out = log_sinkhorn(x, n_iters=20, temperature=0.05)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    row_sums = out.sum(dim=-1)
    col_sums = out.sum(dim=-2)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=5e-2), "Row sums not close to 1"
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=5e-2), "Col sums not close to 1"
    print(f"  Input shape:  {tuple(x.shape)}")
    print(f"  Output shape: {tuple(out.shape)}")
    print("  Doubly stochastic check passed")


def test_kronecker_mixer():
    print()
    print("=" * 60)
    print("Test 2: KroneckerMixer")
    print("=" * 60)
    B, total_dim, block_size, basis_num = 4, 16, 4, 4
    mixer = KroneckerMixer(total_dim, block_size, basis_num=basis_num, sinkhorn_iters=5, dropout=0.0)
    x = torch.randn(B, total_dim, requires_grad=True)
    out = mixer(x)
    assert out.shape == (B, total_dim), f"Expected {(B, total_dim)}, got {out.shape}"
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "KroneckerMixer grad error"
    print(f"  Input shape:  {tuple(x.shape)}")
    print(f"  Output shape: {tuple(out.shape)}")
    print("  Gradients OK")


def test_unimixer_lite_layer():
    print()
    print("=" * 60)
    print("Test 3: UniMixerLiteLayer")
    print("=" * 60)
    B, T, D = 4, 4, 16
    layer = UniMixerLiteLayer(seq_len=T, d_model=D, block_size=4, basis_num=4,
                              d_ff=32, sinkhorn_iters=5, dropout=0.0)
    x = torch.randn(B, T, D, requires_grad=True)
    y = torch.randn(B, T, D, requires_grad=True)
    out_x, out_y = layer(x, y)
    assert out_x.shape == (B, T, D), f"Expected out_x {(B, T, D)}, got {out_x.shape}"
    assert out_y.shape == (B, T, D), f"Expected out_y {(B, T, D)}, got {out_y.shape}"
    loss = out_x.sum() + out_y.sum()
    loss.backward()
    assert x.grad is not None and y.grad is not None, "UniMixerLiteLayer grad error"
    print(f"  x/y shape:    {tuple(x.shape)}")
    print(f"  out_x shape:  {tuple(out_x.shape)}")
    print(f"  out_y shape:  {tuple(out_y.shape)}")
    print("  Gradients OK")


def test_unimixer_lite_model():
    print()
    print("=" * 60)
    print("Test 4: UniMixerLite Model")
    print("=" * 60)
    num_fields = 4
    feature_map = create_dummy_feature_map(num_fields=num_fields)
    B = 4
    model = UniMixerLite(
        feature_map=feature_map,
        model_id="UniMixerLite_test",
        gpu=-1,
        learning_rate=1e-3,
        embedding_dim=16,
        num_transformer_layers=2,
        block_size=4,
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


def test_block_size_assert():
    print()
    print("=" * 60)
    print("Test 5: Block Size Divisibility Assertion")
    print("=" * 60)
    feature_map = create_dummy_feature_map(num_fields=4)
    try:
        _ = UniMixerLite(
            feature_map=feature_map,
            model_id="UniMixerLite_test_fail",
            gpu=-1,
            learning_rate=1e-3,
            embedding_dim=16,
            num_transformer_layers=1,
            block_size=6,  # 4 * 16 = 64, 64 % 6 != 0
            basis_num=4,
            sinkhorn_iters=5,
            transformer_dropout=0.0,
            ffn_dim=32,
            use_pos_embedding=False,
            output_mlp_hidden_units=[16],
            net_dropout=0.0,
            batch_norm=False,
            optimizer="adam",
            loss="binary_crossentropy",
            verbose=0,
            model_root="./checkpoints",
            metrics=["AUC"],
        )
        assert False, "Expected AssertionError due to divisibility"
    except AssertionError as e:
        print(f"  Correctly raised AssertionError: {e}")


def test_model_zoo_import():
    print()
    print("=" * 60)
    print("Test 6: Model Zoo Import")
    print("=" * 60)
    import model_zoo
    assert hasattr(model_zoo, "UniMixerLite"), "UniMixerLite not found in model_zoo"
    print("  UniMixerLite successfully registered in model_zoo")


if __name__ == "__main__":
    test_log_sinkhorn()
    test_kronecker_mixer()
    test_unimixer_lite_layer()
    test_unimixer_lite_model()
    test_block_size_assert()
    test_model_zoo_import()
    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
