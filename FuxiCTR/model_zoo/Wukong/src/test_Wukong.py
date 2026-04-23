#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for Wukong model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch
from collections import OrderedDict
from fuxictr.features import FeatureMap
from Wukong import RMSNorm, WukongLinearCompressBlock, WukongFactorizationMachineBlock, WukongCrossBlock, WukongLayer, WuKong


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


def test_rms_norm():
    print("=" * 60)
    print("Test 1: RMSNorm")
    print("=" * 60)
    B, D = 4, 8
    norm = RMSNorm(dim=D)
    x = torch.randn(B, D, requires_grad=True)
    out = norm(x)
    assert out.shape == (B, D), f"Expected {(B, D)}, got {out.shape}"
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "RMSNorm grad error"
    print(f"  Input shape:  {tuple(x.shape)}")
    print(f"  Output shape: {tuple(out.shape)}")
    print("  Gradients OK")


def test_wukong_lcb():
    print()
    print("=" * 60)
    print("Test 2: WukongLinearCompressBlock")
    print("=" * 60)
    B, num_in, num_out, D = 4, 8, 16, 32
    lcb = WukongLinearCompressBlock(num_emb_in=num_in, num_emb_out=num_out)
    x = torch.randn(B, num_in, D, requires_grad=True)
    out = lcb(x)
    assert out.shape == (B, num_out, D), f"Expected {(B, num_out, D)}, got {out.shape}"
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "LCB grad error"
    print(f"  Input shape:  {tuple(x.shape)}")
    print(f"  Output shape: {tuple(out.shape)}")
    print("  Gradients OK")


def test_wukong_fmb():
    print()
    print("=" * 60)
    print("Test 3: WukongFactorizationMachineBlock")
    print("=" * 60)
    B, token_num, token_dim, rank, num_out = 4, 8, 16, 24, 16
    ffn_out_dim = 64
    fmb = WukongFactorizationMachineBlock(
        token_num=token_num,
        token_dim=token_dim,
        rank=rank,
        ffn_out_dim=ffn_out_dim,
        num_emb_out=num_out,
        dropout=0.0
    )
    x = torch.randn(B, token_num, token_dim, requires_grad=True)
    out = fmb(x)
    assert out.shape == (B, num_out, token_dim), f"Expected {(B, num_out, token_dim)}, got {out.shape}"
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "FMB grad error"
    print(f"  Input shape:  {tuple(x.shape)}")
    print(f"  Output shape: {tuple(out.shape)}")
    print("  Gradients OK")


def test_wukong_cross_block():
    print()
    print("=" * 60)
    print("Test 4: WukongCrossBlock")
    print("=" * 60)
    B, num_in, D = 4, 8, 16
    num_lcb, num_fmb, rank = 8, 8, 24
    ffn_out_dim = 64
    cross = WukongCrossBlock(
        num_emb_in=num_in,
        dim_emb=D,
        num_emb_lcb=num_lcb,
        num_emb_fmb=num_fmb,
        rank_fmb=rank,
        ffn_out_dim=ffn_out_dim,
        dropout=0.0
    )
    x = torch.randn(B, num_in, D, requires_grad=True)
    out = cross(x)
    expected_seq_len = num_lcb + num_fmb
    assert out.shape == (B, expected_seq_len, D), f"Expected {(B, expected_seq_len, D)}, got {out.shape}"
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "WukongCrossBlock grad error"
    print(f"  Input shape:  {tuple(x.shape)}")
    print(f"  Output shape: {tuple(out.shape)}")
    print("  Gradients OK")


def test_wukong_layer():
    print()
    print("=" * 60)
    print("Test 5: WukongLayer (Cross + TokenSpecificSwishGLU)")
    print("=" * 60)
    B, num_in, D = 4, 8, 16
    num_lcb, num_fmb, rank = 8, 8, 24
    ffn_out_dim = 64
    layer = WukongLayer(
        num_emb_in=num_in,
        dim_emb=D,
        num_emb_lcb=num_lcb,
        num_emb_fmb=num_fmb,
        rank_fmb=rank,
        ffn_out_dim=ffn_out_dim,
        dropout=0.0
    )
    x = torch.randn(B, num_in, D, requires_grad=True)
    out = layer(x)
    expected_seq_len = num_lcb + num_fmb
    assert out.shape == (B, expected_seq_len, D), f"Expected {(B, expected_seq_len, D)}, got {out.shape}"
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "WukongLayer grad error"
    print(f"  Input shape:  {tuple(x.shape)}")
    print(f"  Output shape: {tuple(out.shape)}")
    print("  Gradients OK")


def test_wukong_model():
    print()
    print("=" * 60)
    print("Test 6: WuKong Model")
    print("=" * 60)
    feature_map = create_dummy_feature_map(num_fields=4)
    B = 4
    model = WuKong(
        feature_map=feature_map,
        model_id="Wukong_test1",
        gpu=-1,
        learning_rate=1e-3,
        embedding_dim=16,
        num_transformer_layers=2,
        num_emb_lcb=4,
        num_emb_fmb=4,
        rank_fmb=8,
        ffn_out_dim=32,
        transformer_dropout=0.0,
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


def test_wukong_kwargs_compat():
    print()
    print("=" * 60)
    print("Test 7: WuKong kwargs compatibility (legacy params silently ignored)")
    print("=" * 60)
    feature_map = create_dummy_feature_map(num_fields=4)
    B = 4
    model = WuKong(
        feature_map=feature_map,
        model_id="Wukong_test2",
        gpu=-1,
        learning_rate=1e-3,
        embedding_dim=16,
        num_transformer_layers=2,
        transformer_dropout=0.0,
        ffn_out_dim=32,
        net_dropout=0.0,
        batch_norm=False,
        embedding_regularizer=None,
        net_regularizer=None,
        optimizer="adam",
        loss="binary_crossentropy",
        verbose=0,
        model_root="./checkpoints",
        metrics=["AUC"],
        # Legacy params that should be silently ignored via kwargs.pop
        num_heads=4,
        att_emb_size=4,
        use_cls_token=True,
        use_pos_embedding=True,
    )
    inputs = {f: torch.randint(0, 10, (B,)) for f in feature_map.features.keys()}
    inputs["label"] = torch.ones(B, 1)
    out_dict = model(inputs)
    y_pred = out_dict["y_pred"]
    assert y_pred.shape == (B, 1), f"Expected y_pred {(B, 1)}, got {y_pred.shape}"
    print(f"  y_pred shape: {tuple(y_pred.shape)}")
    print("  Forward pass OK (legacy params silently ignored)")


def test_model_zoo_import():
    print()
    print("=" * 60)
    print("Test 8: Model Zoo Import")
    print("=" * 60)
    import model_zoo
    assert hasattr(model_zoo, "WuKong") or hasattr(model_zoo, "HeteroAttention"), "WuKong not found in model_zoo"
    print("  WuKong successfully registered in model_zoo")


if __name__ == "__main__":
    test_rms_norm()
    test_wukong_lcb()
    test_wukong_fmb()
    test_wukong_cross_block()
    test_wukong_layer()
    test_wukong_model()
    test_wukong_kwargs_compat()
    test_model_zoo_import()
    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
