# convex_caldera.py
# ---------------------------------------------------------
# Convex-CALDERA: closed-form low-rank + simple bit allocation (no CVXPY)
#
# This module replaces the previous CVXPY-based solver with:
#   1) A closed-form proximal operator for the nuclear norm
#      (via singular value soft-thresholding).
#   2) A lightweight bit-allocation routine that can be
#      upgraded later to a CVXQ-style dual algorithm.
#
# Dependencies: numpy, torch
# ---------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch

'''
def normalize_h_diag(h: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize/robustify diagonal Hessian vector h to be used consistently across:
    - low-rank solve (weighted Frobenius)
    - H-aware residual quantization
    - H-weighted error checks

    Uses quantile clipping + median normalization (more robust to extreme tails).
    Returns CPU float tensor.
    """
    h = h.detach().float().cpu()
    lo = torch.quantile(h, 0.01)
    hi = torch.quantile(h, 0.99)
    h = torch.clamp(h, lo, hi)
    h = h / (torch.median(h) + 1e-8)
    h = torch.clamp(h, min=eps)
    return h
'''
def normalize_h_diag(
    h: torch.Tensor,
    eps: float = 1e-8,
    clip_q: Optional[Tuple[float, float]] = None,  # e.g. (0.001, 0.999)
    norm: str = "mean",                            # "mean" or "median"
    power: float = 1.0,                            # >1 amplifies differences
) -> torch.Tensor:
    """
    Preserve relative curvature differences.
    Returns CPU float tensor.
    """
    h = h.detach().float().cpu()

    # Optional: wide clipping only (avoid killing dynamic range)
    if clip_q is not None:
        lo = torch.quantile(h, clip_q[0])
        hi = torch.quantile(h, clip_q[1])
        h = torch.clamp(h, lo, hi)

    # Scale normalization (scalar only)
    if norm == "mean":
        h = h / (h.mean() + 1e-12)
    elif norm == "median":
        h = h / (torch.median(h) + 1e-12)
    else:
        raise ValueError(f"Unknown norm={norm}")

    # Amplify contrast if desired
    if power != 1.0:
        # keep positivity; power>1 increases spread, <1 decreases spread
        h = torch.pow(h.clamp(min=eps), power)

    return h.clamp(min=eps)

# ---------------------------------------------------------
# Hyper-parameters for Convex-CALDERA
# ---------------------------------------------------------

@dataclass
class ConvexCalderaParams:
    """
    Hyper-parameters for the Convex-CALDERA decomposition.

    Attributes
    ----------
    mu : float
        Nuclear-norm regularization weight in the objective:
            0.5 ||W - L||_F^2 + mu * ||L||_*
        Ignored if tau_star is not None and we enforce a
        trace / nuclear-norm constraint instead.
    tau_star : Optional[float]
        If not None, we enforce an approximate nuclear-norm
        / trace constraint by truncating singular values such
        that sum_i s_i <= tau_star.  When this is set, we do
        not use the soft-threshold parameter `mu` directly.
    lambda_reg : float
        Regularization coefficient in the CALDERA-type penalty
        for the residual quantization term (used only for
        logging / certificates here; the actual residual
        quantization happens in a separate module).
    kappa : float
        Scale factor for the rate–distortion penalty term.
    k : float
        Exponential rate parameter in exp(-k * b).  Kept here
        to match the original CALDERA notation.
    b_min : int
        Minimum bit-width allowed.
    b_max : int
        Maximum bit-width allowed.
    B_tot : float
        Total bit budget (average bit per weight or per group,
        depending on how you interpret it in your experiment).
        In this minimal implementation, for a single group we
        simply clamp b between [b_min, min(b_max, B_tot)].
    per_channel : bool
        If True, later you can extend the allocator to assign
        bits per-channel (e.g., per out-feature).  For now we
        only implement single-scalar b, but we keep this flag
        in case you want to upgrade to multi-group CVXQ style.
    """

    mu: float = 1e-3
    tau_star: Optional[float] = None
    rank_budget: Optional[int] = None
    lambda_reg: float = 1.0
    kappa: float = 1.0
    k: float = 1.0
    num_groups: int = 8

    b_min: int = 2
    b_max: int = 8
    B_tot: float = 4.0

    target_rank: Optional[int] = None   # e.g., 64/128. If set -> enforce hard rank-k
    L_bits: int = 8                     # bits for left factor
    R_bits: int = 8   
    Q_bits: int = 2          # backbone / residual bits
    inner_iters: int = 3     # alternating refinements

    per_channel: bool = False
    # Compatibility with old API
    discrete_bits: list = None
    solver: str = "SCS"
    solver_verbose: bool = False
    
    def __post_init__(self):
        if self.discrete_bits is None:
            self.discrete_bits = [2, 3, 4, 8, 16]


# ---------------------------------------------------------
# Low-rank part: nuclear-norm proximal operator via SVD
# ---------------------------------------------------------

def _soft_threshold_singular_values(
    S: np.ndarray,
    mu: float
) -> np.ndarray:
    """
    Soft-threshold singular values: s_i -> max(s_i - mu, 0).
    """
    return np.maximum(S - mu, 0.0)


def _project_to_nuclear_ball(S: np.ndarray, tau: float) -> np.ndarray:
    # project nonnegative singular values onto {s>=0, sum s <= tau}
    if tau <= 0:
        return np.zeros_like(S)
    if S.sum() <= tau:
        return S.copy()

    u = np.sort(S)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u - (cssv - tau) / (np.arange(len(u)) + 1) > 0)[0][-1]
    theta = (cssv[rho] - tau) / (rho + 1.0)
    return np.maximum(S - theta, 0.0)

def make_col_groups(n_cols: int, G: int):
    # 等分列为 G 组，返回 [(c0,c1), ...]
    edges = np.linspace(0, n_cols, G + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(G)]

def quantize_residual_h_aware_grouped(
    R: torch.Tensor,
    h_diag: torch.Tensor,
    bits_per_group: np.ndarray,
    groups,
    device: torch.device = None,
) -> torch.Tensor:
    """
    按列分组，每组用自己的 bit 做 H-aware residual quant。
    R: [out, in]
    h_diag: [in]
    bits_per_group: (G,)
    """
    Rq = R.clone()
    for (c0, c1), b in zip(groups, bits_per_group):
        R_slice = R[:, c0:c1]
        h_slice = h_diag[c0:c1]
        Rq[:, c0:c1] = quantize_residual_h_aware(
            R_slice, h_slice, num_bits=int(b), device=device
        )
    return Rq



def solve_convex_low_rank(
    W: torch.Tensor,
    params: ConvexCalderaParams
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Solve the low-rank convex problem in closed form via SVD.

        min_L  0.5 ||W - L||_F^2 + mu ||L||_*
        (or a trace / nuclear-norm style constraint via tau_star)

    Parameters
    ----------
    W : torch.Tensor (m x n)
        Weight matrix of a single linear / conv-equivalent layer.
        Can be on GPU; we will detach + move to CPU.
    params : ConvexCalderaParams
        Hyper-parameters controlling the nuclear-norm regularization.

    Returns
    -------
    L_star_np : np.ndarray
        Optimal low-rank component (same shape as W).
    R_star_np : np.ndarray
        Residual component W - L_star.
    stats : dict
        Dictionary of useful statistics (rank, nuclear norm, etc.).
    """
    # 1. Move to CPU + numpy
    W_np = W.detach().cpu().numpy()
    m, n = W_np.shape

    # 2. SVD
    # For typical EE274-sized layers, full SVD is OK. For truly
    # large LLM layers, you would want a truncated / randomized SVD.
    U, S, Vh = np.linalg.svd(W_np, full_matrices=False)

    # 3. Decide how to modify singular values
    if params.tau_star is not None:
        if params.rank_budget is not None:
            r = int(min(params.rank_budget, len(S)))
            S_new = np.zeros_like(S)
            S_new[:r] = S[:r]  # HARD top-r truncation
            mode = f"hard_rank_constraint_r{r}"
        else:
            S_new = _project_to_nuclear_ball(S, params.tau_star)
            mode = "trace_constraint"
    else:
        # Penalty-type formulation: proximal of mu * ||L||_*
        S_new = _soft_threshold_singular_values(S, params.mu)
        mode = "soft_threshold"

    # 4. Reconstruct L_star
    L_star_np = (U * S_new) @ Vh
    R_star_np = W_np - L_star_np

    # 5. Collect statistics
    rank = int((S_new > 0).sum())
    nuclear_norm = float(S_new.sum())
    frob_L = float(np.linalg.norm(L_star_np, ord="fro"))
    frob_R = float(np.linalg.norm(R_star_np, ord="fro"))

    stats = {
        "mode": mode,
        "rank": rank,
        "nuclear_norm": nuclear_norm,
        "frob_L": frob_L,
        "frob_R": frob_R,
        "shape": (m, n),
    }

    return L_star_np, R_star_np, stats

def solve_convex_low_rank_weighted_diagH(
    W: torch.Tensor,
    h_diag: Optional[torch.Tensor],
    params: ConvexCalderaParams,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Approximate solve of:
        min_L 0.5 || (W - L) @ diag(sqrt(h)) ||_F^2 + mu ||L||_*
    via right-scaling trick:
        W' = W D, solve min_{L'} 0.5||W' - L'||_F^2 + mu||L'||_*
        then L = L' D^{-1}
    """
    W_np = W.detach().cpu().numpy()
    m, n = W_np.shape

    if h_diag is None:
        # fallback to original
        U, S, Vh = np.linalg.svd(W_np, full_matrices=False)
        if params.tau_star is not None:
            if params.rank_budget is not None:
                r = int(min(params.rank_budget, len(S)))
                S_new = np.zeros_like(S)
                S_new[:r] = S[:r]
                mode = f"hard_rank_constraint_r{r}"
            else:
                S_new = _project_to_nuclear_ball(S, params.tau_star)
                mode = "trace_constraint"
        else:
            S_new = _soft_threshold_singular_values(S, params.mu)
            mode = "soft_threshold"
        L_star_np = (U * S_new) @ Vh
        R_star_np = W_np - L_star_np
        stats = {
            "mode": mode,
            "rank": int((S_new > 0).sum()),
            "nuclear_norm": float(S_new.sum()),
            "frob_L": float(np.linalg.norm(L_star_np, ord="fro")),
            "frob_R": float(np.linalg.norm(R_star_np, ord="fro")),
            "shape": (m, n),
            "used_H": False,
        }

        return L_star_np, R_star_np, stats

    h_t = normalize_h_diag(h_diag, clip_q=(0.001, 0.999), norm="mean", power=2.0)
    h = h_t.numpy()
    h_p01 = float(np.quantile(h, 0.01))
    h_p99 = float(np.quantile(h, 0.99))
    h_p99_over_p01 = h_p99 / max(h_p01, 1e-12)

    d = np.sqrt(h)              # (n,)
    dinv = 1.0 / d


    # right-scale W
    Wp = W_np * d[None, :]      # W' = W D

    # SVD on W'
    U, S, Vh = np.linalg.svd(Wp, full_matrices=False)

    if params.tau_star is not None:
        if params.rank_budget is not None:
            r = int(min(params.rank_budget, len(S)))
            S_new = np.zeros_like(S)
            S_new[:r] = S[:r]
            mode = f"hard_rank_constraint_weighted_diagH_r{r}"
        else:
            S_new = _project_to_nuclear_ball(S, params.tau_star)
            mode = "trace_constraint_weighted_diagH"
    
    else:
        S_new = _soft_threshold_singular_values(S, params.mu)
        mode = "soft_threshold_weighted_diagH"

    Lp = (U * S_new) @ Vh       # L' in scaled space
    L_star_np = Lp * dinv[None, :]  # L = L' D^{-1}

    R_star_np = W_np - L_star_np

    stats = {
        "mode": mode,
        "rank": int((S_new > 0).sum()),
        "nuclear_norm": float(S_new.sum()),
        "frob_L": float(np.linalg.norm(L_star_np, ord="fro")),
        "frob_R": float(np.linalg.norm(R_star_np, ord="fro")),
        "shape": (m, n),
        "used_H": True,
        "h_min": float(h.min()),
        "h_med": float(np.median(h)),
        "h_max": float(h.max()),
    }

    return L_star_np, R_star_np, stats



# ---------------------------------------------------------
# Bit allocation: minimal scalar version (upgradeable)
# ---------------------------------------------------------

def allocate_bits_scalar(
    params: ConvexCalderaParams
) -> Tuple[float, Dict[str, Any]]:
    """
    Improved bit allocator that respects the budget.
    
    Strategy: Allocate bits more intelligently based on B_tot
    - If B_tot is very small (< 2), still use at least 2 bits for R
    - Otherwise use B_tot directly
    """
    # More intelligent allocation: use B_tot as-is, but ensure minimum 2 bits
    b_star = float(np.clip(params.B_tot, params.b_min, params.b_max))

    stats = {
        "b_star": b_star,
        "b_min": params.b_min,
        "b_max": params.b_max,
        "B_tot": params.B_tot,
        "allocation_mode": "improved_clamp",
    }
    return b_star, stats

def factorize_low_rank_hard(
    L_star: torch.Tensor,
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hard rank-k factorization: L_star ≈ A @ B
    A: (m, r), B: (r, n), balanced via sqrt(S).
    """
    m, n = L_star.shape
    r = min(rank, m, n)
    U, S, Vh = torch.linalg.svd(L_star, full_matrices=False)
    U = U[:, :r]
    S = S[:r]
    Vh = Vh[:r, :]
    sqrtS = torch.sqrt(S)
    A = U * sqrtS[None, :]
    B = sqrtS[:, None] * Vh
    return A, B


def quantize_tensor(
    tensor: torch.Tensor,
    num_bits: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    Quantize a tensor to specified number of bits using symmetric quantization.
    
    Correct quantization: map [-max_val, max_val] to [-2^(b-1), 2^(b-1)-1]
    """
    if num_bits >= 16:
        return tensor
    
    if tensor.numel() == 0:
        return tensor
    
    if device is not None:
        tensor = tensor.to(device)
    
    # Find max absolute value
    max_val = torch.max(torch.abs(tensor))
    
    if max_val < 1e-10:
        return tensor
    
    # Number of levels: for b bits, we have 2^b - 1 levels
    # But we want to map to [-2^(b-1), 2^(b-1)-1], which is 2^b values total
    q_max = 2 ** (num_bits - 1) - 1  # e.g., for 8-bit: 127
    
    # Quantize: scale to [-q_max, q_max] and round
    # quantized = round(tensor / max_val * q_max)
    quantized = torch.round(tensor / max_val * q_max)
    
    # Clamp (should already be within bounds, but just in case)
    quantized = torch.clamp(quantized, -q_max, q_max)
    
    # Dequantize: scale back
    # dequantized = quantized / q_max * max_val
    dequantized = (quantized / q_max) * max_val
    
    return dequantized


def quantize_residual_per_row(R, num_bits, device=None):
    if num_bits >= 16 or R.numel() == 0:
        return R
    if device is not None:
        R = R.to(device)

    if num_bits == 1:
        # per-row sign quantization: ±t
        t = R.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)  # (out,1)
        return torch.sign(R) * t

    qmax = 2 ** (num_bits - 1) - 1  # 2bit->1, 3bit->3, 4bit->7
    t = R.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)

    scale = t / qmax                 # <-- 标准写法
    R_int = torch.round(R / scale).clamp(-qmax, qmax)
    return R_int * scale



def quantize_residual_h_aware(
    R: torch.Tensor,
    h_diag: torch.Tensor,
    num_bits: int,
    device: torch.device = None,
) -> torch.Tensor:
    """
    H-aware residual quantization for diagonal H.
    Right-scale columns by sqrt(h) before quantization, then scale back.

    This makes quantization minimize ||(R - Rq) diag(sqrt(h))||_F approximately.
    """
    if num_bits >= 16:
        return R
    if device is not None:
        R = R.to(device)
        h_diag = h_diag.to(device)

    h = normalize_h_diag(h_diag, clip_q=(0.001, 0.999), norm="mean", power=2.0).to(device=R.device)

    d = torch.sqrt(h)                    # (in,)
    Rw = R * d[None, :]                  # R D
    Rw_q = quantize_residual_per_row(Rw, num_bits=num_bits, device=device)
    return Rw_q / d[None, :]             # (R D)_q D^{-1}

# ---------------------------------------------------------
# Main decomposition wrapper
# ---------------------------------------------------------

class ConvexCalderaDecomposition:
    """
    High-level wrapper for Convex-CALDERA decomposition.

    Usage
    -----
    >>> params = ConvexCalderaParams(mu=1e-3, B_tot=4.0)
    >>> decomp = ConvexCalderaDecomposition(params)
    >>> out = decomp.decompose(W)   # W: torch.Tensor (m x n)
    >>> L_low_rank = out["L_low_rank"]      # torch.Tensor
    >>> R_residual = out["R_residual"]      # torch.Tensor
    >>> b_star = out["b_star"]              # float

    The actual quantization of R_residual should be handled in a
    separate quantization module (e.g., your existing LLM
    quantizer).  This class only computes the *convex* low-rank
    part + an initial bit allocation.
    """

    def __init__(self, params: ConvexCalderaParams):
        self.params = params

    def decompose(
        self,
        W: torch.Tensor,
        H_diag: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """
        Run the Convex-CALDERA decomposition on a single weight matrix.

        Parameters
        ----------
        W : torch.Tensor (m x n)
            Weight matrix to decompose.  Can live on CPU or GPU.
        H_sqrt : Optional[torch.Tensor]
            Placeholder for compatibility with earlier versions.
            In the simplest EE274 setting, we typically use
            H_sqrt = I, so this argument is not needed.
        device : Optional[torch.device]
            Device for the returned tensors.  If None, uses W.device.
        dtype : Optional[torch.dtype]
            dtype for the returned tensors.  If None, uses W.dtype.

        Returns
        -------
        result : dict
            {
              "L_low_rank": torch.Tensor,
              "R_residual": torch.Tensor,
              "b_star": float,
              "solver_stats": dict,
              "bit_stats": dict,
              "params": dict,
            }
        """
        if device is None:
            device = W.device
        if dtype is None:
            dtype = W.dtype

        # 1) Low-rank convex solve via SVD
        if H_diag is not None:
            if H_diag.ndim == 2:
                H_diag = torch.diag(H_diag)
            assert H_diag.numel() == W.shape[1], \
                f"len(H_diag)={H_diag.numel()} != W.shape[1]={W.shape[1]}"

            L_star_np, R_star_np, solver_stats = solve_convex_low_rank_weighted_diagH(
                W=W,
                h_diag=H_diag,
                params=self.params,
            )
        else:
            L_star_np, R_star_np, solver_stats = solve_convex_low_rank(
                W=W,
                params=self.params,
            )


        # 2) Bit allocation (minimal scalar version)
        b_star, bit_stats = allocate_bits_scalar(self.params)

        # 3) Convert back to torch tensors
        L_low_rank = torch.from_numpy(L_star_np).to(device=device, dtype=dtype)
        R_residual = torch.from_numpy(R_star_np).to(device=device, dtype=dtype)

        # 4) Optionally compute a simple "certificate"-like objective value
        #    This is purely for logging; the real certificate in CALDERA
        #    would involve the quantization of R_residual as well.
        # residual energy (pre-quantization)
        if H_diag is not None:
            h_t = normalize_h_diag(H_diag, clip_q=(0.001, 0.999), norm="mean", power=2.0).to(device=W.device, dtype=W.dtype)
            d_t = torch.sqrt(h_t)
            recon_error = 0.5 * float(torch.norm(R_residual * d_t[None, :], p="fro") ** 2)

        else:
            recon_error = 0.5 * float(torch.norm(R_residual, p="fro") ** 2)

        nuclear_norm = solver_stats["nuclear_norm"]
        penalty_low_rank = (self.params.mu * nuclear_norm) if self.params.mu is not None else 0.0

        # simple rate-distortion-style penalty term
        rd_penalty = self.params.lambda_reg * self.params.kappa * np.exp(
            -self.params.k * b_star
        )
        obj_value = recon_error + penalty_low_rank + float(rd_penalty)

        solver_stats["reconstruction_error"] = recon_error
        solver_stats["penalty_low_rank"] = penalty_low_rank
        solver_stats["rd_penalty"] = float(rd_penalty)
        solver_stats["objective_value"] = obj_value

        result = {
            "L_low_rank": L_low_rank,
            "R_residual": R_residual,
            "b_star": b_star,
            "solver_stats": solver_stats,
            "bit_stats": bit_stats,
            "params": asdict(self.params),
        }
        return result


# 返回对象类（兼容老 API）
@dataclass
class ConvexCalderaResult:
    """Object-based result for backward compatibility."""
    L_low_rank: torch.Tensor
    R_residual: torch.Tensor
    W_compressed: torch.Tensor
    solver_status: str
    solve_time: float
    avg_bit_width: float
    effective_rank: float
    duality_gap: float
    residual_norm: float
    objective_value: float
    b_star: float
    b_discrete: float


def choose_bits_discrete(params: ConvexCalderaParams) -> int:
    valid = [b for b in params.discrete_bits
             if params.b_min <= b <= params.b_max and b <= params.B_tot]
    if not valid:
        return int(np.ceil(np.clip(params.B_tot, params.b_min, params.b_max)))
    return max(valid)  

# ---------------------------------------------------------
# Convenience function (functional API)
# ---------------------------------------------------------

def convex_caldera_decompose(
    W: torch.Tensor,
    H: Optional[torch.Tensor] = None,
    params: Optional[ConvexCalderaParams] = None,
    device: Optional[torch.device] = None,
) -> ConvexCalderaResult:
    """
    Main Convex-CALDERA function (matches old API).
    
    Parameters
    ----------
    W : torch.Tensor
        Weight matrix to decompose.
    H : Optional[torch.Tensor]
        Hessian (currently unused in SVD version).
    params : Optional[ConvexCalderaParams]
        If None, uses defaults.
    device : Optional[torch.device]
        Output device.
    
    Returns
    -------
    result : ConvexCalderaResult
        Object with all metrics as attributes.
    """
    import time
    start_time = time.time()
    
    if params is None:
        params = ConvexCalderaParams()
    if device is None:
        device = W.device

    # Internal decomposition
    internal_decomp = ConvexCalderaDecomposition(params)
    result_dict = internal_decomp.decompose(W=W, H_diag=H, device=device)

    # ===== DEBUG: compare vs identity =====
    if H is not None:
        result_dict_id = internal_decomp.decompose(W=W, H_diag=None, device=device)

        L0 = result_dict_id["L_low_rank"].detach()
        L1 = result_dict["L_low_rank"].detach()
        
        print("mode0:", result_dict_id["solver_stats"].get("mode"))
        print("mode1:", result_dict["solver_stats"].get("mode"),
              "used_H=", result_dict["solver_stats"].get("used_H"))
        print("delta_L rel:", (torch.norm(L1 - L0) / (torch.norm(L0) + 1e-12)).item())
        if "h_min" in result_dict["solver_stats"]:
            ss = result_dict["solver_stats"]
            print(f"h_min / h_med / h_max = "
                  f"{ss['h_min']:.2e} / {ss['h_med']:.2e} / {ss['h_max']:.2e}")
            if "h_p99_over_p01" in ss:
                print(f"h_p99/p01 = {ss['h_p99_over_p01']:.2f}")
        W0 = (result_dict_id["L_low_rank"] + result_dict_id["R_residual"]).detach()
        W1 = (result_dict["L_low_rank"] + result_dict["R_residual"]).detach()
        print("delta_W rel:", (torch.norm(W1 - W0) / (torch.norm(W0) + 1e-12)).item())



    

    L = result_dict["L_low_rank"].to(device)
    W = W.to(device)
    solver_stats = result_dict["solver_stats"]
    h_diag = None
    if H is not None:
        h_diag = H if H.ndim == 1 else torch.diag(H)
        h_diag = h_diag.to(device)
    
    # 如果你想让 Q_bits 吃预算：就用离散选择；否则就用 params.Q_bits
    #Q_bits = params.Q_bits



    # -----------------------------
    # (NEW) G=8 group bits by columns
    # -----------------------------
    G_groups = int(getattr(params, "num_groups", 8))
    groups = make_col_groups(W.shape[1], G_groups)
    
    # 基准 bit（仍然沿用你原来的离散选择逻辑）
    base_bit = int(choose_bits_discrete(params))
    
    # 构造一个“混合 bits”的 b_discrete（长度=G_groups）
    # 例：base=2 -> [2,2,2,2,3,3,3,3]
    b_discrete = np.full((G_groups,), float(base_bit), dtype=float)
    half = G_groups // 2
    b_discrete[half:] = float(min(base_bit + 1, params.b_max))
    
    # snap 到离散集合（保证只取 discrete_bits 里的值）
    def snap_to_discrete(x: float) -> float:
        return float(min(params.discrete_bits, key=lambda t: abs(t - x)))
    
    b_discrete = np.array([snap_to_discrete(x) for x in b_discrete], dtype=float)
    
    # 简单预算修正：保证平均 bits <= B_tot（按最高 bit 的组逐个降一级）
    def mean_bits(bv: np.ndarray) -> float:
        return float(np.mean(bv))
    
    B_tot = float(params.B_tot)
    disc_sorted = sorted([int(x) for x in params.discrete_bits])
    
    while mean_bits(b_discrete) > B_tot + 1e-9:
        j = int(np.argmax(b_discrete))
        cur = int(b_discrete[j])
        lowers = [bb for bb in disc_sorted if bb < cur and bb >= params.b_min]
        if not lowers:
            break
        b_discrete[j] = float(max(lowers))



    
    
    L_hat = None
    R_hat = None
    
    for _ in range(params.inner_iters):
        # ---- (A) factorize low-rank into A,B with hard rank-k ----
        if params.target_rank is not None:
            A, B = factorize_low_rank_hard(L, rank=params.target_rank)
            A_q = quantize_tensor(A, num_bits=params.L_bits, device=device)
            B_q = quantize_tensor(B, num_bits=params.R_bits, device=device)
            L_hat = A_q @ B_q
        else:
            # fallback: quantize full low-rank matrix (not recommended)
            L_hat = quantize_residual_per_row(L, num_bits=params.L_bits, device=device)
    

        R = W - L_hat
        if h_diag is not None:
            R_hat = quantize_residual_h_aware_grouped(
                R=R,
                h_diag=h_diag,
                bits_per_group=b_discrete,
                groups=groups,
                device=device,
            )
        else:
            # 没有 H 的情况：也按组做 per-row 量化
            Rq = R.clone()
            for (c0, c1), b in zip(groups, b_discrete):
                Rq[:, c0:c1] = quantize_residual_per_row(
                    R[:, c0:c1], num_bits=int(b), device=device
                )
            R_hat = Rq

        
    
        # ---- (C) re-solve low-rank to fit W - R_hat (this is the "adapt-to-quant" step) ----
        W_minus_Q = (W - R_hat)
        if h_diag is not None:
            L_np, _, _ = solve_convex_low_rank_weighted_diagH(W=W_minus_Q, h_diag=h_diag, params=params)
        else:
            L_np, _, _ = solve_convex_low_rank(W=W_minus_Q, params=params)
        L = torch.from_numpy(L_np).to(device=device, dtype=W.dtype)
    
    W_comp = L_hat + R_hat


    
    # ======== H-weighted error check (debug only) ========
    def h_weighted_relerr(W, Wh, h):
        d = torch.sqrt(h).to(W.device)
        num = torch.norm((W - Wh) * d[None, :], p="fro")
        den = torch.norm(W * d[None, :], p="fro") + 1e-12
        return (num / den).item()

    if H is not None:
        h_diag_raw = H if H.ndim == 1 else torch.diag(H)
        h_diag_n = normalize_h_diag(h_diag_raw, clip_q=(0.001, 0.999), norm="mean", power=2.0).to(device=W.device, dtype=W.dtype)
        val = h_weighted_relerr(W, W_comp, h_diag_n)
        print(f"[CHECK] H-weighted relerr = {val:.6f}")

    
    
    # ========== 计算量化误差 ==========
    # 用原始 W 和量化后的 W_comp 计算误差
    quant_error = torch.norm(W - W_comp, p="fro").item()
    
    solve_time = time.time() - start_time
    
    # avg_bit_width：这里给你一个合理定义：按组 bits 的加权平均（按参数个数加权）
    out_dim = W.shape[0]
    p_counts = np.array([(c1 - c0) * out_dim for (c0, c1) in groups], dtype=float)
    p_g = p_counts / p_counts.sum()
    avg_bits = float(np.sum(p_g * b_discrete))
    
    print("\n[BITS DEBUG PRINT]")
    print("  B_tot =", params.B_tot)
    print("  discrete_bits =", params.discrete_bits)
    
    print("  num_groups =", len(b_discrete))
    print("  b_discrete =", b_discrete.tolist())
    
    unique_bits = np.unique(b_discrete)
    print("  unique(b_discrete) =", unique_bits.tolist())
    print("  num_unique_bits =", len(unique_bits))
    
    print("  sum(p_g * b_g) =", float(np.sum(p_g * b_discrete)))
    print("================================================================\n")


    return ConvexCalderaResult(
        L_low_rank=L_hat,
        R_residual=R_hat,
        W_compressed=W_comp,
        solver_status="optimal",
        solve_time=solve_time,
        avg_bit_width=float(avg_bits),
        effective_rank=float(params.target_rank if params.target_rank is not None else solver_stats["rank"]),
        duality_gap=0.0,
        residual_norm=quant_error,
        objective_value=solver_stats["objective_value"],
        b_star=float(avg_bits),
        b_discrete=b_discrete,  
    )
