"""
Example usage of Convex-CALDERA algorithm with evaluation metrics.

This script demonstrates how to use the new Convex-CALDERA implementation
with both penalty and constrained forms, and how to evaluate the results.
"""

import torch
import sys
import os
import numpy as np
import pandas as pd
sys.path.append('rank-constrained-regression-main')

from src.caldera.decomposition.convex_caldera import (
    convex_caldera_decompose as convex_caldera,
    ConvexCalderaParams
)
from src.caldera.utils.metrics import (
    evaluate_compression,
    plot_bit_allocation_heatmap,
    plot_accuracy_vs_bits,
    plot_loss_vs_rank,
    plot_singular_value_spectra,
    compute_singular_values
)
# ---- replace sample W/H with actual LLaMA-2-7B layer ----
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Example 1: Single weight matrix compression with penalty form
print("\n" + "="*60)
print("Example 1: Convex-CALDERA with Penalty Form")
print("="*60)







def build_layer_list(num_blocks=4):
    """Return list of 7 linear layer names per block."""
    names = []
    for i in range(num_blocks):
        prefix = f"model.layers.{i}"
        names += [
            # f"{prefix}.self_attn.q_proj",
            # f"{prefix}.self_attn.k_proj",
            #f"{prefix}.self_attn.v_proj",
            #f"{prefix}.self_attn.o_proj",
            f"{prefix}.mlp.gate_proj",
            f"{prefix}.mlp.up_proj",
            #f"{prefix}.mlp.down_proj",
        ]
    return names

def find_exact_module(model, layer_name: str):
    """Find module by exact name."""
    for name, mod in model.named_modules():
        if name == layer_name:
            return name, mod
    return None, None

def load_hessians(h_path: str):
    if os.path.exists(h_path):
        Hall = torch.load(h_path, map_location="cpu")
        print(f"Loaded Hessians: {h_path}, num_keys={len(Hall)}")
        return Hall
    print("No Hessians file found, will run with Identity (H=None).")
    return None

def get_hdiag(Hall, true_name: str, device, dtype=torch.float32):
    """Hall key style: 'language_model.' + true_name"""
    if Hall is None:
        return None
    key = "language_model." + true_name
    h = Hall.get(key, None)
    if h is None:
        return None
    # h is 1D CPU tensor from build_diag_hessians.py
    h = h.to(device=device, dtype=dtype)
    # safety
    h = torch.clamp(h, min=1e-8)
    return h



def calibrate_weight(W_orig_fp32: torch.Tensor, W_comp_fp32: torch.Tensor):
    """
    Calibrate compressed weight scale to match original weight scale.
    W_orig_fp32, W_comp_fp32: float32 tensors on same device.
    """
    # 1) match std
    scale = (W_orig_fp32.std() / (W_comp_fp32.std() + 1e-8)).clamp(0.1, 10.0)
    W_comp_fp32 = W_comp_fp32 * scale

    # 2) clamp max-abs outliers
    max0 = W_orig_fp32.abs().max()
    max1 = W_comp_fp32.abs().max()
    if max1 > 2.0 * max0:
        W_comp_fp32 = W_comp_fp32 * (2.0 * max0 / (max1 + 1e-8))

    return W_comp_fp32, float(scale.item()), float(max0.item()), float(max1.item())


def quick_logits_nan_check(model, tokenizer, device="cuda"):
    model.eval()
    inp = tokenizer("Hello world", return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inp).logits
    return bool(torch.isnan(logits).any().item()), bool(torch.isinf(logits).any().item())


def compress_many_layers(model, tokenizer, layer_names, params, device):
    did_plot = False
    results = {}
    cnt = 0

    for ln in layer_names:
        true_name, mod = find_exact_module(model, ln)
        if mod is None or (not hasattr(mod, "weight")):
            print("[SKIP] not found or no weight:", ln)
            continue

        # original
        W0 = mod.weight.detach().to(torch.float32).to(device)

        # decompose (Identity)
        decomp = convex_caldera(W=W0, H=None, params=params, device=device)

        # compressed
        W1 = decomp.W_compressed.detach().to(torch.float32).to(device)

        # DEBUG: before calibration
        print(f"[DBG pre] {ln}  orig_max={W0.abs().max().item():.3e} comp_max={W1.abs().max().item():.3e} "
              f"orig_std={W0.std().item():.3e} comp_std={W1.std().item():.3e}")

        # calibrate
        W1_cal, sc, max0, max1 = calibrate_weight(W0, W1)

        # DEBUG: after calibration
        print(f"[DBG post] {ln} scale={sc:.3f}  cal_max={W1_cal.abs().max().item():.3e} cal_std={W1_cal.std().item():.3e}")

        # write back
        mod.weight.data.copy_(W1_cal.to(mod.weight.dtype).to(mod.weight.device))

        cnt += 1
        nan_flag, inf_flag = quick_logits_nan_check(model, tokenizer)
        print(f"[CHECK] after {cnt} layers ({ln}): logits nan? {nan_flag} inf? {inf_flag}")
        if nan_flag or inf_flag:
            print("[STOP] NaN/Inf detected at layer:", ln)
            break


        results[ln] = decomp
                # ---- Plot singular value spectra (do once) ----
        if not did_plot:
            sv_original = compute_singular_values(W0.detach().float().cpu())
            sv_compressed = compute_singular_values(W1_cal.detach().float().cpu())

            save_path = f"singular_values_{ln.replace('.', '_')}_b{params.B_tot}.png"
            plot_singular_value_spectra(
                sv_original,
                sv_compressed,
                save_path=save_path
            )
            print("[PLOT] Saved:", save_path)
            did_plot = True

        print(f"[OK] {ln}  avg_bits={decomp.avg_bit_width:.2f}  resid_norm={decomp.residual_norm:.4f}")
        print(
            f"[OK] {ln}  avg_bits={decomp.avg_bit_width:.2f}  "
            f"eff_rank={decomp.effective_rank:.1f}  resid_norm={decomp.residual_norm:.4f}"
        )


    return results



num_blocks_to_compress = 4   
layer_list = build_layer_list(num_blocks=num_blocks_to_compress)
print("Will compress layers:", len(layer_list))


def make_params(B_tot: float):
    return ConvexCalderaParams(
        B_tot=B_tot,
        b_min=1.0,
        b_max=8.0,
        mu=0.1,
        tau_star=None,
        lambda_reg=0.01,
        k=1.0,
        discrete_bits=[1, 2, 3, 4, 8],
        solver_verbose=False,

        # make bit effects visible:
        inner_iters=1,
        target_rank=512,
        L_bits=8,
        R_bits=8,
        num_groups=8,
    )






#model_name = "meta-llama/Llama-2-13b-hf"
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

# Hessians
h_path = "Llama-2-13b_diag_Hessians_.pt"
Hall = load_hessians(h_path)

num_blocks_to_compress = 4
layer_list = build_layer_list(num_blocks=num_blocks_to_compress)

for B in [2.0]:
    print("\n" + "="*80)
    print(f"Compressing {len(layer_list)} layers with B_tot={B}")
    print("="*80)

    # IMPORTANT: reload fresh model each time (do NOT accumulate compression)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    params = make_params(B)

    _ = compress_many_layers(model=model, tokenizer=tokenizer, layer_names=layer_list, params=params, device=device)



    save_dir = f"./llama3_8b_convex_caldera_blocks{num_blocks_to_compress}_b{int(B)}"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Saved model to:", save_dir)





