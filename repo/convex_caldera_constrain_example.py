import torch
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------
# Path & imports
# ------------------------------------------------
sys.path.append("rank-constrained-regression-main")

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.caldera.decomposition.convex_caldera import (
    convex_caldera_decompose as convex_caldera,
    ConvexCalderaParams,
)

# ------------------------------------------------
# Basic setup
# ------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#model_name = "meta-llama/Llama-2-13b-hf"
model_name = "meta-llama/Llama-3.1-8B-Instruct"

bits_list = [2.0, 3.0, 4.0, 8.0]
target_ranks = [128, 64, 32]
num_blocks = 4  # same as your penalty run

# ------------------------------------------------
# Layer selection
# ------------------------------------------------
def build_layer_list(num_blocks: int = 4):
    names = []
    for i in range(num_blocks):
        prefix = f"model.layers.{i}"
        names += [
            f"{prefix}.mlp.gate_proj",
            f"{prefix}.mlp.up_proj",
            # f"{prefix}.mlp.down_proj",
        ]
    return names


layer_list = build_layer_list(num_blocks)
print("Compressing layers:", len(layer_list))


def find_exact_module(model, layer_name: str):
    for name, mod in model.named_modules():
        if name == layer_name:
            return mod
    return None


# ------------------------------------------------
# Convex Caldera params (constrained)
# ------------------------------------------------
def make_params_constrained(rank_budget: int, B_tot: float):
    return ConvexCalderaParams(
        B_tot=B_tot,
        b_min=2,
        b_max=8,
        tau_star=1.0,          # just to enter constrained branch
        mu=None,
        rank_budget=rank_budget,  # NEW
        lambda_reg=0.01,
        k=1.0,
        discrete_bits=[2, 3, 4, 8],
        inner_iters=1,
        target_rank=None,      
        L_bits=8,
        R_bits=8,
        num_groups=8,
    )


def recon_error_fro(W0: torch.Tensor, Wc: torch.Tensor) -> float:
    # returns ||W0 - Wc||_F^2  (squared Frobenius)
    diff = (W0 - Wc).float()
    return float((diff * diff).sum().item())

# ------------------------------------------------
# Quick NaN / Inf check
# ------------------------------------------------
def quick_logits_nan_check(model, tokenizer):
    model.eval()
    inp = tokenizer("Hello world", return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inp).logits

    return (
        torch.isnan(logits).any().item(),
        torch.isinf(logits).any().item(),
    )


# ------------------------------------------------
# Main constrained sweep
# ------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_auth_token=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

rows = []
for b in bits_list:
    for r in target_ranks:
        print("\n" + "=" * 80)
        print(f"Constrained run: target_rank={r}, B_tot={b}")
        print("=" * 80)
    
        # fresh model each rank
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        layer_errs = []
        layer_resids = []
        layer_effrs = []
        diverged = False
        layer_numel = []
        layer_fro2W = []
        for ln in layer_list:
            mod = find_exact_module(model, ln)
            if mod is None:
                print("[SKIP]", ln)
                continue
    
            W0 = mod.weight.detach().to(torch.float32).to(device)
            params = make_params_constrained(rank_budget=r, B_tot=b)
    
            decomp = convex_caldera(
                W=W0,
                H=None,
                params=params,
                device=device,
            )
            Wc = decomp.W_compressed.detach().to(device=device, dtype=torch.float32)
            layer_errs.append(recon_error_fro(W0, Wc))
            layer_resids.append(float(decomp.residual_norm))
            layer_effrs.append(float(decomp.effective_rank))
            layer_numel.append(int(W0.numel()))
            layer_fro2W.append(float((W0 * W0).sum().item()))
    
            mod.weight.data.copy_(
                decomp.W_compressed
                    .to(mod.weight.dtype)
                    .to(mod.weight.device)
            )
    
            nan_flag, inf_flag = quick_logits_nan_check(model, tokenizer)
    
            print(
                f"[OK] {ln} "
                f"rank_budget={r} "
                f"eff_rank={decomp.effective_rank:.1f} "
                f"resid={decomp.residual_norm:.2f} "
                f"nan={nan_flag}"
            )
    
            if nan_flag or inf_flag:
                print("[STOP] Divergence detected.")
                diverged = True
                break


        rows.append({
            "B_tot": float(b),
            "rank_budget": int(r),
            "mean_fro_sq": float(np.mean(layer_errs)) if len(layer_errs) else float("nan"),
            "mean_resid_norm": float(np.mean(layer_resids)) if len(layer_resids) else float("nan"),
            "mean_eff_rank": float(np.mean(layer_effrs)) if len(layer_effrs) else float("nan"),
            "diverged": bool(diverged),
            "num_layers_done": int(len(layer_errs)),
            "mean_numel_W": float(np.mean(layer_numel)),
            "mean_fro2_W": float(np.mean(layer_fro2W)),
        })
        print(float(np.mean(layer_errs)), float(np.mean(layer_fro2W)))
        save_dir = f"./llama3_8b_convex_caldera_constrained_r{r}_b{int(b)}"
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print("Saved to:", save_dir)
    
        del model
        torch.cuda.empty_cache()

df = pd.DataFrame(rows)
df.to_csv("constrained_rank_bit_error.csv", index=False)
print("Saved: constrained_rank_bit_error.csv")

# heatmap matrix: rows=rank, cols=bits
pivot = df.pivot(index="rank_budget", columns="B_tot", values="mean_fro_sq").sort_index(ascending=False)

plt.figure()
plt.imshow(pivot.values, aspect="auto")
plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
plt.yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
plt.xlabel("B_tot (avg bits)")
plt.ylabel("Rank budget")
plt.title("Constrained Convex-CALDERA: Rank–Bit–Error Landscape (H=I)")
plt.colorbar(label="mean ||W - W_hat||_F^2")
plt.tight_layout()
plt.savefig("constrained_rank_bit_error_heatmap.png", dpi=200)
print("Saved: constrained_rank_bit_error_heatmap.png")

