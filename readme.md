## EE269 - Signal Processing and Quantization for Machine Learning

### Convex-CALDERA: Certified Rank–Bit Trade-offs for Post-Training LLM Quantization

#### Results

Main comparison across core baselines for LLaMA-2 (7B/13B). Convex-CALDERA uses the penalty formulation (rank≈full) with varying average bit budgets. Perplexity (PPL) is evaluated on WikiText-2 and C4; zero-shot accuracy is reported on WinoGrande (Wino), RTE, PiQA, and ARC-Challenge (ARC-C). Entries marked "—" indicate unavailable or diverged results.

| Method | Avg Bits | WikiText-2 (PPL↓) | C4 (PPL↓) | Wino↑ | RTE↑ | PiQA↑ | ARC-C↑ |
|--------|----------|-------------------|-----------|-------|------|-------|--------|
| **LLaMA-2 7B** | | | | | | | |
| LDA | 16.0 | 38.6 | 160.6 | 50.3 | 53.8 | 57.3 | 23.2 |
| Kernel SVM (RBF) | 16.0 | — | — | 49.2 | 52.7 | 50.2 | 27.9 |
| NMF + Linear Classifier | 16.0 | — | — | 49.5 | 52.7 | 50.3 | 27.9 |
| SCL–Scalar Quant (8-bit) | 8.0 | 574.4 | 239.0 | 65.0 | 53.8 | 72.4 | 35.1 |
| SCL–Lloyd–Max Quant (8-bit) | 8.0 | — | — | 49.7 | 50.5 | 52.4 | 21.1 |
| SCL–Vector Quant (8-bit VQ) | 8.0 | — | — | 49.5 | 52.7 | 52.6 | 21.2 |
| GPTQ (4-bit) | 4.0 | 9.47 | — | 69.6 | 64.3 | 77.2 | 42.5 |
| QLoRA–128 (NF4) | 4.0 | 9.03 | — | 69.5 | 63.2 | 77.8 | 43.1 |
| QuIP# (2-bit) | 2.0 | 7.73 | 10.0 | 61.7 | 57.8 | 69.6 | 29.9 |
| CALDERA (rank-128) | 2.2 | 6.76 | 8.83 | 63.8 | 59.9 | 75.1 | 34.6 |
| **Convex-CALDERA (Penalty)** | **4.0** | **8.76** | 17.42 | 68.8 | 63.9 | **78.2** | 42.4 |
| **Convex-CALDERA (Penalty)** | **3.0** | **8.80** | 17.51 | **69.4** | **63.2** | **78.2** | **42.6** |
| **Convex-CALDERA (Penalty)** | **2.0** | 9.42 | 18.96 | **68.4** | **63.9** | **78.1** | **42.1** |
| FP16 (uncompressed) | 16.0 | 5.12 | 6.63 | 67.3 | 63.2 | 78.5 | 40.0 |
| **LLaMA-2 13B** | | | | | | | |
| LDA | 16.0 | 170.5 | 498.8 | 50.7 | 53.1 | 56.3 | 23.3 |
| Kernel SVM (RBF) | 16.0 | — | — | 49.6 | 52.7 | 50.6 | 28.2 |
| NMF + Linear Classifier | 16.0 | — | — | 49.8 | 52.7 | 50.2 | 28.2 |
| SCL–Scalar Quant (8-bit) | 8.0 | 11.4 | 24.6 | 68.7 | 61.4 | 77.2 | 39.9 |
| SCL–Lloyd–Max Quant (8-bit) | 8.0 | — | — | 49.5 | 52.7 | 51.8 | 21.2 |
| SCL–Vector Quant (8-bit VQ) | 8.0 | — | — | 50.8 | 50.5 | 52.3 | 21.0 |
| GPTQ (4-bit) | 4.0 | 8.23 | — | 71.4 | 62.5 | 79.1 | 46.7 |
| QLoRA–128 (NF4) | 4.0 | 7.86 | — | 71.5 | 63.5 | 78.5 | 48.0 |
| QuIP# (2-bit) | 2.0 | 6.06 | 8.07 | 63.6 | 54.5 | 74.2 | 36.2 |
| CALDERA (rank-128) | 2.2 | 5.72 | 7.66 | 67.9 | 58.5 | 76.0 | 38.7 |
| **Convex-CALDERA (Penalty)** | **4.0** | **7.74** | 15.81 | **72.1** | **66.8** | **79.6** | **47.1** |
| **Convex-CALDERA (Penalty)** | **3.0** | **7.76** | 15.84 | **71.9** | **65.7** | **79.5** | **46.9** |
| **Convex-CALDERA (Penalty)** | **2.0** | 8.20 | 16.7 | **71.2** | **67.5** | **78.7** | **45.8** |
| FP16 (uncompressed) | 16.0 | 4.57 | 6.05 | 69.5 | 61.7 | 78.8 | 45.6 |

Main comparison for LLaMA-3 8B. Convex-CALDERA uses the penalty formulation (rank≈full) with varying average bit budgets. Perplexity (PPL) is measured on WikiText-2 and C4; zero-shot accuracy on WinoGrande (Wino), RTE, PiQA, and ARC-Challenge (ARC-C). Entries marked "—" indicate unavailable or diverged results.

| Method | Avg Bits | WikiText-2 (PPL↓) | C4 (PPL↓) | Wino↑ | RTE↑ | PiQA↑ | ARC-C↑ |
|--------|----------|-------------------|-----------|-------|------|-------|--------|
| LDA | 16.0 | — | — | 50.4 | 53.8 | 54.2 | 22.9 |
| Kernel SVM (RBF) | 16.0 | — | — | 51.5 | 47.3 | 51.7 | 24.4 |
| NMF + Linear Classifier | 16.0 | — | — | 51.1 | 47.3 | 52.0 | 24.1 |
| SCL–Scalar Quant (8-bit) | 8.0 | 9.4 | 22.5 | 73.9 | 75.5 | 79.2 | 51.4 |
| SCL–Lloyd–Max Quant (8-bit) | 8.0 | — | 278.1 | 64.8 | 55.6 | 70.1 | 33.4 |
| SCL–Vector Quant (8-bit VQ) | 8.0 | — | — | 51.5 | 47.3 | 54.1 | 21.8 |
| QuIP# (2-bit) | 2.0 | 10.9 | 11.8 | 66.5 | 57.0 | 69.6 | 31.0 |
| CALDERA (rank-128) | 2.2 | 9.21 | 10.5 | 69.7 | 63.1 | 74.4 | 36.3 |
| **Convex-CALDERA (Penalty)** | **4.0** | **8.84** | 20.27 | **74.2** | **74.0** | **80.3** | **53.7** |
| **Convex-CALDERA (Penalty)** | **3.0** | **8.95** | 20.70 | **74.4** | **74.0** | **80.0** | **53.6** |
| **Convex-CALDERA (Penalty)** | **2.0** | **11.33** | 28.39 | **71.7** | **64.6** | **75.0** | **45.5** |
| FP16 (uncompressed) | 16.0 | 5.54 | 7.01 | 73.5 | 68.6 | 79.7 | 50.2 |

---

### Quantization

**This code is based on the Lambda Labs platform's Launch Instance (GPU: NVIDIA H100 PCIe, Image: Lambda Stack 22.04).**

#### SCL Baselines

In the `scl` directory, there are three scripts: `scl_llama2_7b_quant.py`, `scl_llama2_13b_quant.py`, and `scl_llama3_8b_quant.py`, each corresponding to a different model. Taking the `llama2-7b` model as an example, to run

##### SCL–Scalar Quant (8-bit) 

```shell
python scl_llama2_7b_quant.py --method scalar_uniform
```

##### SCL–Lloyd–Max Quant (8-bit)

```shell
python scl_llama2_7b_quant.py \
  --method lloyd_max \
  --sample_size 200000 \
  --num_iters 25
```

##### SCL–Vector Quant (8-bit VQ)

```shell
python scl_llama2_7b_quant.py \
  --method vector_vq \
  --block_size 4 \
  --sample_size 200000 \
  --num_iters 25
```

#### Convex-CALDERA (penalty)

In the `repo` directory, run

```shell
python convex_caldera_example.py
```

Note: In `convex_caldera_example.py`, select the corresponding model, avg bits and saving directory.

It will output the relevant compression information in the console, and generate the `llama2_7b_convex_caldera_blocks4_b2` folder (as an example). Afterward, use the `eval_scl_llama2_7b.py` in `scl` to evaluate.

####  Convex-CALDERA (constrained)

In addition to the penalty form, we also provide scripts for the **constrained form** of Convex-CALDERA, which enforces an explicit hard rank constraint.

In the constrained setting, Convex-CALDERA performs **top-r singular value truncation** while optimizing under a fixed average bit budget.

To run constrained Convex-CALDERA experiments (e.g., reproducing Table 6 or rank–bit–error heatmaps), use the constrained script in the `repo` directory:

```
python convex_caldera_constrain_example.py
```

In the script, configure:

- bits_list e.g. [2.0, 3.0, 4.0, 8.0]
- target_ranks e.g. [128, 64, 32]
- model name and output directory

Each run will:

- compress the specified layers under a hard rank constraint

- report reconstruction error, effective rank, and stability diagnostics

- save the compressed model to a directory such as:

  ```
  llama2_13b_convex_caldera_constrained_r64_b2/
  ```

The resulting models can be evaluated using the same SCL evaluation scripts as in the penalty setting, e.g.,

```
python eval_scl_llama2_13b.py \
  --model_dir ../repo/llama2_13b_convex_caldera_constrained_r64_b2/
```

Note: Aggressive rank constraints at low bit-widths may lead to numerical instability. Divergent runs are detected and excluded from reported results.

#### Evaluation

The above commands will create a folder, e.g., `llama2_7b_convex_caldera_blocks4_b2/`, and then run the following command to evaluate:

```shell
python eval_scl_llama2_7b.py --model_dir ../repo/llama2_7b_convex_caldera_blocks4_b2/
```

Note: Replace the folder name after `--model_dir` accordingly. Meanwhile, in `eval_scl_llama2_7b.py`, within the `TASKS` list:

```python
TASKS = [
    "wikitext",
    "c4",
    "winogrande",
    "rte",
    "piqa",
    "arc_challenge",
]
```

Select the metrics you wish to evaluate. If you choose `c4`, it is recommended to add the parameter `limit=1000` in the `evaluator.simple_evaluate()` call. For other metrics, this parameter can typically be omitted. This adjustment prevents the evaluation of `c4` from becoming excessively slow.
