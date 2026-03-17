## Convex-CALDERA: Certified Rank–Bit Trade-offs for Post-Training LLM

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
