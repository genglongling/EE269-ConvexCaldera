import argparse
import json
from lm_eval import evaluator



'''
def check_weight_diff(base_model, model_dir, device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM

    layer_path = "model.layers.0.mlp.gate_proj.weight"

    def get_tensor(model, path):
        obj = model
        for p in path.split("."):
            obj = getattr(obj, p)
        return obj.detach()

    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="auto"
    )
    comp = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float16, device_map="auto"
    )

    w0 = get_tensor(base, layer_path).float().cpu()
    w1 = get_tensor(comp, layer_path).float().cpu()

    rel = torch.norm(w0 - w1) / (torch.norm(w0) + 1e-12)
    print(f"[CHECK] {layer_path} relative diff = {rel.item():.6f}")

    return rel.item()
'''

TASKS = [
    "wikitext",
    #"c4",
    #"winogrande",
    #"rte",
    #"piqa",
    #"arc_challenge",
]

def run_eval(model_dir, task, device="cuda"):
    print(f"\n====== Running task: {task} ======\n")

    results = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_dir},dtype=float16",
        tasks=[task],
        batch_size=2,
        device=device,
        #limit=1000
    )

    print(json.dumps(results["results"], indent=2))
    return results["results"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=None,
                    help="Optional: base/original model name or path to compare weights")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    summary = {}

    print("\n=== Evaluating model:", args.model_dir, "===\n")
    '''
    if args.base_model is not None:
        rel = check_weight_diff(args.base_model, args.model_dir, args.device)
        if rel < 1e-6:
            print("[WARNING] Relative diff is ~0. Your eval may not be using modified weights (or weights didn't change).")

    '''
    for task in TASKS:
        result = run_eval(args.model_dir, task, args.device)
        summary[task] = result

    print("\n=========== FINAL SUMMARY ===========\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
