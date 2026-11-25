# GPU steps (Kaggle or Colab)

The PyTorch steps (FP16 baseline, sensitivity probe) are too slow on the CPU laptop
(measured: ~24 s per prompt for FP32 generation on an i5-1335U, so the probe would take about
19 hours). Run them once on a free GPU.

## Kaggle
1. New notebook, Settings -> Accelerator: **GPU T4 x2** (or P100), Internet: **On**.
2. Cell 1: `!git clone https://github.com/aceqbit/ToolQuant /kaggle/working/ToolQuant`
3. Cell 2: `!pip install -q torch transformers accelerate datasets scipy statsmodels numpy mlflow huggingface_hub`
   (Kaggle already has torch/transformers; the line is harmless.)
4. Cell 3: `!python /kaggle/working/ToolQuant/scripts/run_on_gpu.py --task all`
   - baseline: FP16 on probe + validation (`results/eval_fp16_*.json`)
   - probe: all 28 groups, 4-bit single-group perturbation, written after every group
     (`results/sensitivity_partial_*.json`; rerun the cell to resume after a disconnect)
5. Download `/kaggle/working/toolquant_results.zip`.

## Bring back
Unzip into `D:\ToolQuant\results\` (keep file names). Files expected:
- `eval_fp16_probe_<utc>.json`, `eval_fp16_validation_<utc>.json`
- `sensitivity_qwen3-0.6b_<utc>.json` and `.csv`

Then tell Claude Code to continue (prompt 1.4 accept, 1.5 maps, ...).
