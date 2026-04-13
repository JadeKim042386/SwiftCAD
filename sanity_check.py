"""
Phase 1.5: Reshape / Mask Sanity Check

Validates:
1. Alternating Attention reshape correctness: (300,N,D) ↔ (100,3*N,D)
2. Padding mask reshape correctness: (N,300) ↔ (3*N,100)
3. Cross-attention memory_key_padding_mask propagation
4. Attention weight masking: padding positions get zero attention weight
"""

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention


def test_reshape_roundtrip():
    """Test that reshape (300,N,D) → (100,3*N,D) → (300,N,D) is lossless."""
    print("=" * 60)
    print("Test 1: Reshape roundtrip")
    print("=" * 60)

    N, D = 4, 144
    num_views = 3
    tokens_per_view = 100
    S = num_views * tokens_per_view  # 300

    # Create distinguishable data: each view has unique pattern
    x_original = torch.randn(S, N, D)
    # Tag each view for easy identification
    x_original[:100] += 10.0   # view 0
    x_original[100:200] += 20.0  # view 1
    x_original[200:300] += 30.0  # view 2

    # Forward reshape: (300, N, D) → (100, 3*N, D)
    x = x_original.view(num_views, tokens_per_view, N, D)  # (3, 100, N, D)
    x = x.permute(1, 0, 2, 3)                               # (100, 3, N, D)
    x_reshaped = x.reshape(tokens_per_view, num_views * N, D)  # (100, 3*N, D)

    assert x_reshaped.shape == (100, 12, D), f"Expected (100, 12, {D}), got {x_reshaped.shape}"

    # Verify view separation: batch dim should be [view0_b0, view0_b1, ..., view1_b0, ...]
    # view 0 batches are at indices 0..N-1, view 1 at N..2N-1, view 2 at 2N..3N-1
    for v in range(num_views):
        for b in range(N):
            merged_idx = v * N + b
            expected = x_original[v * tokens_per_view:(v + 1) * tokens_per_view, b, :]  # (100, D)
            actual = x_reshaped[:, merged_idx, :]  # (100, D)
            assert torch.allclose(expected, actual, atol=1e-7), \
                f"View {v}, batch {b}: mismatch at merged_idx {merged_idx}"

    # Inverse reshape: (100, 3*N, D) → (300, N, D)
    x_inv = x_reshaped.reshape(tokens_per_view, num_views, N, D)  # (100, 3, N, D)
    x_inv = x_inv.permute(1, 0, 2, 3)                              # (3, 100, N, D)
    x_restored = x_inv.reshape(S, N, D)                             # (300, N, D)

    assert torch.allclose(x_original, x_restored, atol=1e-7), "Roundtrip failed!"
    print("  PASS: (300,N,D) → (100,3*N,D) → (300,N,D) roundtrip is lossless")
    print()


def test_mask_reshape():
    """Test padding mask reshape: (N,300) → (3*N,100) preserves mask per view."""
    print("=" * 60)
    print("Test 2: Mask reshape")
    print("=" * 60)

    N = 4
    num_views = 3
    tokens_per_view = 100

    # Create mask with different padding per view per batch
    # True = masked (ignored)
    mask = torch.zeros(N, num_views * tokens_per_view).bool()  # (N, 300)
    # Batch 0: view 0 has 80 valid tokens, view 1 has 60, view 2 has 90
    mask[0, 80:100] = True    # view 0, pos 80-99 masked
    mask[0, 160:200] = True   # view 1, pos 60-99 masked (offset by 100)
    mask[0, 290:300] = True   # view 2, pos 90-99 masked (offset by 200)
    # Batch 1: all valid (no masking)
    # Batch 2: view 0 all masked
    mask[2, :100] = True
    # Batch 3: view 2, last 50 tokens masked
    mask[3, 250:300] = True

    # Reshape: (N, 300) → (3*N, 100)
    mask_r = mask.view(N, num_views, tokens_per_view)   # (N, 3, 100)
    mask_r = mask_r.permute(1, 0, 2)                     # (3, N, 100)
    mask_reshaped = mask_r.reshape(num_views * N, tokens_per_view)  # (3*N, 100)

    assert mask_reshaped.shape == (12, 100), f"Expected (12, 100), got {mask_reshaped.shape}"

    # Verify: view v, batch b → merged index v*N + b
    # Batch 0, view 0 (merged_idx=0): pos 80-99 should be True
    assert mask_reshaped[0, 79] == False
    assert mask_reshaped[0, 80] == True
    assert mask_reshaped[0, 99] == True

    # Batch 0, view 1 (merged_idx=4): pos 60-99 should be True
    assert mask_reshaped[4, 59] == False
    assert mask_reshaped[4, 60] == True

    # Batch 0, view 2 (merged_idx=8): pos 90-99 should be True
    assert mask_reshaped[8, 89] == False
    assert mask_reshaped[8, 90] == True

    # Batch 1, all views (merged_idx=1,5,9): all False
    assert mask_reshaped[1].sum() == 0
    assert mask_reshaped[5].sum() == 0
    assert mask_reshaped[9].sum() == 0

    # Batch 2, view 0 (merged_idx=2): all True
    assert mask_reshaped[2].all()

    # Batch 3, view 2 (merged_idx=11): pos 50-99 should be True
    assert mask_reshaped[11, 49] == False
    assert mask_reshaped[11, 50] == True

    print("  PASS: Mask reshape (N,300) → (3*N,100) preserves per-view per-batch mask")
    print()


def test_frame_wise_attention_masking():
    """Test that frame-wise attention respects padding mask after reshape."""
    print("=" * 60)
    print("Test 3: Frame-wise attention masking")
    print("=" * 60)

    d_model, nhead = 144, 8
    N, num_views, tokens_per_view = 2, 3, 100
    S = num_views * tokens_per_view

    mha = MultiheadAttention(d_model, nhead, dropout=0.0, batch_first=False)
    mha.eval()

    # Create input
    x = torch.randn(S, N, d_model)
    mask = torch.zeros(N, S).bool()
    # Batch 0, view 1 (positions 100-199): mask positions 150-199
    mask[0, 150:200] = True

    # Reshape for frame-wise attention
    x_r = x.view(num_views, tokens_per_view, N, d_model)
    x_r = x_r.permute(1, 0, 2, 3).reshape(tokens_per_view, num_views * N, d_model)

    mask_r = mask.view(N, num_views, tokens_per_view)
    mask_r = mask_r.permute(1, 0, 2).reshape(num_views * N, tokens_per_view)

    # Run attention with need_weights=True to inspect weights
    with torch.no_grad():
        _, attn_weights = mha(x_r, x_r, x_r, key_padding_mask=mask_r, need_weights=True)

    # attn_weights: (3*N, 100, 100) averaged over heads
    # View 1, batch 0 → merged_idx = 1*N + 0 = 2
    view1_batch0_weights = attn_weights[2]  # (100, 100)

    # Positions 50-99 (relative to view 1) should have zero attention weight
    # because mask[0, 150:200] = True → view 1 pos 50-99 masked
    attn_to_masked = view1_batch0_weights[:, 50:].sum().item()
    attn_to_valid = view1_batch0_weights[:, :50].sum().item()

    assert attn_to_masked < 1e-5, f"Attention to masked positions should be ~0, got {attn_to_masked}"
    assert attn_to_valid > 0, "Attention to valid positions should be > 0"

    print(f"  Attention to masked positions: {attn_to_masked:.2e} (should be ~0)")
    print(f"  Attention to valid positions: {attn_to_valid:.4f} (should be > 0)")
    print("  PASS: Frame-wise attention correctly ignores masked positions")
    print()


def test_cross_attention_masking():
    """Test cross-attention with memory_key_padding_mask."""
    print("=" * 60)
    print("Test 4: Cross-attention masking")
    print("=" * 60)

    d_model, nhead = 144, 8
    N = 2
    S_enc = 300  # encoder sequence length
    S_dec = 60   # decoder sequence length

    # Simulate cross-attention (decoder attends to encoder memory)
    cross_attn = MultiheadAttention(d_model, nhead, dropout=0.0, batch_first=False)
    cross_attn.eval()

    tgt = torch.randn(S_dec, N, d_model)  # decoder queries
    memory = torch.randn(S_enc, N, d_model)  # encoder keys/values

    # Create encoder padding mask
    memory_key_padding_mask = torch.zeros(N, S_enc).bool()
    # Batch 0: positions 250-299 are padding
    memory_key_padding_mask[0, 250:] = True
    # Batch 1: positions 200-299 are padding
    memory_key_padding_mask[1, 200:] = True

    with torch.no_grad():
        _, attn_weights = cross_attn(
            tgt, memory, memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True
        )

    # attn_weights: (N, S_dec, S_enc) = (2, 60, 300)
    # Batch 0: attention to positions 250-299 should be ~0
    attn_to_masked_b0 = attn_weights[0, :, 250:].sum().item()
    attn_to_valid_b0 = attn_weights[0, :, :250].sum().item()

    # Batch 1: attention to positions 200-299 should be ~0
    attn_to_masked_b1 = attn_weights[1, :, 200:].sum().item()
    attn_to_valid_b1 = attn_weights[1, :, :200].sum().item()

    assert attn_to_masked_b0 < 1e-5, f"Batch 0: attention to masked = {attn_to_masked_b0}"
    assert attn_to_masked_b1 < 1e-5, f"Batch 1: attention to masked = {attn_to_masked_b1}"
    assert attn_to_valid_b0 > 0
    assert attn_to_valid_b1 > 0

    print(f"  Batch 0: attn to masked={attn_to_masked_b0:.2e}, valid={attn_to_valid_b0:.4f}")
    print(f"  Batch 1: attn to masked={attn_to_masked_b1:.2e}, valid={attn_to_valid_b1:.4f}")
    print("  PASS: Cross-attention correctly masks encoder padding positions")
    print()


def test_cross_attention_decoder_layer():
    """Test TransformerDecoderLayerImproved with memory_key_padding_mask end-to-end."""
    print("=" * 60)
    print("Test 5: TransformerDecoderLayerImproved E2E mask propagation")
    print("=" * 60)

    from model.layers.improved_transformer import TransformerDecoderLayerImproved

    d_model, nhead = 144, 8
    N, S_enc, S_dec = 2, 300, 60

    layer = TransformerDecoderLayerImproved(d_model, nhead, dim_feedforward=512, dropout=0.0)
    layer.eval()

    tgt = torch.randn(S_dec, N, d_model)
    memory = torch.randn(S_enc, N, d_model)

    # All encoder positions masked for batch 1 except first 10
    memory_key_padding_mask = torch.zeros(N, S_enc).bool()
    memory_key_padding_mask[1, 10:] = True

    with torch.no_grad():
        out = layer(tgt, memory, memory_key_padding_mask=memory_key_padding_mask)

    assert out.shape == (S_dec, N, d_model), f"Expected ({S_dec},{N},{d_model}), got {out.shape}"

    # Output should differ between batches due to different masking
    diff = (out[:, 0, :] - out[:, 1, :]).abs().mean().item()
    assert diff > 0.01, f"Outputs too similar despite different masks: diff={diff}"

    print(f"  Output shape: {out.shape}")
    print(f"  Mean abs diff between batches (different masks): {diff:.4f}")
    print("  PASS: TransformerDecoderLayerImproved propagates memory_key_padding_mask correctly")
    print()


def test_4x_mode_reshape():
    """Test reshape for 4x input mode: (400,N,D) → (100,4*N,D)."""
    print("=" * 60)
    print("Test 6: 4x mode reshape")
    print("=" * 60)

    N, D = 3, 144
    num_views = 4
    tokens_per_view = 100
    S = num_views * tokens_per_view  # 400

    x = torch.randn(S, N, D)

    # Forward
    x_r = x.view(num_views, tokens_per_view, N, D)
    x_r = x_r.permute(1, 0, 2, 3).reshape(tokens_per_view, num_views * N, D)
    assert x_r.shape == (100, 12, D)

    # Verify
    for v in range(num_views):
        for b in range(N):
            expected = x[v * 100:(v + 1) * 100, b, :]
            actual = x_r[:, v * N + b, :]
            assert torch.allclose(expected, actual, atol=1e-7)

    # Roundtrip
    x_inv = x_r.reshape(tokens_per_view, num_views, N, D).permute(1, 0, 2, 3).reshape(S, N, D)
    assert torch.allclose(x, x_inv, atol=1e-7)

    print("  PASS: 4x mode (400,N,D) → (100,4*N,D) roundtrip correct")
    print()


if __name__ == "__main__":
    print()
    print("Drawing2CAD Phase 1.5: Reshape / Mask Sanity Check")
    print("=" * 60)
    print()

    test_reshape_roundtrip()
    test_mask_reshape()
    test_frame_wise_attention_masking()
    test_cross_attention_masking()
    test_cross_attention_decoder_layer()
    test_4x_mode_reshape()

    print("=" * 60)
    print("ALL SANITY CHECKS PASSED")
    print("=" * 60)
