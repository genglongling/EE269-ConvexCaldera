import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM


@torch.no_grad()
def estimate_hdiag_ex2_for_linear(model, layer_name, dataloader, max_batches=64):
    """
    Estimate h_diag (in_features,) as E[x^2] where x is the input activation to a Linear layer.
    Returns: (true_module_name, h_diag_cpu_float32)
    """
    target_module = None
    true_name = None
    for name, mod in model.named_modules():
        if name == layer_name or name.endswith(layer_name):
            target_module = mod
            true_name = name
            break
    if target_module is None:
        raise RuntimeError(f"Can't find layer {layer_name}")

    sum_sq = None  # CPU float64
    count = 0

    def pre_hook(mod, inputs):
        nonlocal sum_sq, count
        x = inputs[0]
        if not torch.is_tensor(x):
            return
        x = x.detach()
        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])  # (tokens, in_features)
        elif x.ndim == 2:
            pass
        else:
            return

        s = (x * x).sum(dim=0).to(dtype=torch.float64, device="cpu")  # (in_features,)
        if sum_sq is None:
            sum_sq = s.clone()
        else:
            sum_sq += s
        count += x.shape[0]

    handle = target_module.register_forward_pre_hook(pre_hook)

    model.eval()
    for bi, batch in enumerate(dataloader):
        if bi >= max_batches:
            break
        _ = model(**batch)

    handle.remove()

    if sum_sq is None or count == 0:
        raise RuntimeError("No activations captured; check layer_name or dataloader format.")

    h_diag = (sum_sq / count).to(torch.float32)  # CPU float32
    h_diag = torch.clamp(h_diag, min=1e-8)
    return true_name, h_diag


class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def build_dataloader(tokenizer, device, seqlen=2048, batch_size=1, num_texts=256):
    base_texts = [
        "Hello! This is a calibration sample.",
        "We estimate activation second moments for curvature-aware compression.",
        "The quick brown fox jumps over the lazy dog.",
        "Large language models can be compressed with low-rank and quantization.",
        "Convex optimization methods can allocate bits adaptively across dimensions.",
    ]
    texts = (base_texts * ((num_texts + len(base_texts) - 1) // len(base_texts)))[:num_texts]
    ds = TextDataset(texts)

    def collate_fn(batch_texts):
        enc = tokenizer(
            list(batch_texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=seqlen,
        )
        return {k: v.to(device) for k, v in enc.items()}

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-13b-hf")
    parser.add_argument("--token", type=str, default=None, help="HF token (or set HF_TOKEN env var)")
    parser.add_argument("--layers", type=str, nargs="+", default=["model.layers.0.mlp.gate_proj"])
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_texts", type=int, default=256)
    parser.add_argument("--max_batches", type=int, default=64)
    parser.add_argument("--out", type=str, default="Llama-2-13b_diag_Hessians.pt")
    args = parser.parse_args()

    hf_token = args.token or os.environ.get("HF_TOKEN", None)

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token,
    )

    embed_device = model.model.embed_tokens.weight.device

    dataloader = build_dataloader(
        tokenizer=tokenizer,
        device=embed_device,
        seqlen=args.seqlen,
        batch_size=args.batch_size,
        num_texts=args.num_texts,
    )

    out_dict = {}
    for layer in args.layers:
        true_name, h_diag = estimate_hdiag_ex2_for_linear(
            model=model,
            layer_name=layer,
            dataloader=dataloader,
            max_batches=args.max_batches,
        )

        key = "language_model." + true_name
        out_dict[key] = h_diag  # CPU tensor

        print(f"[OK] layer request: {layer}")
        print(f"     matched name: {true_name}")
        print(f"     saved key   : {key}")
        print(f"     h_diag shape: {tuple(h_diag.shape)}  min/mean/max: "
              f"{h_diag.min().item():.3e}/{h_diag.mean().item():.3e}/{h_diag.max().item():.3e}")

    torch.save(out_dict, args.out)
    print(f"\nSaved: {args.out}  (num layers: {len(out_dict)})")


if __name__ == "__main__":
    main()
