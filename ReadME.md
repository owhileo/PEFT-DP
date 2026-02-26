# PEFT + DP-SGD on GLUE with TinyBERT

This repository provides the code for fine-tuning **TinyBERT** on the **GLUE benchmark** with optional **DP-SGD** using **Opacus**. It supports multiple **parameter-efficient fine-tuning (PEFT)** methods and their DP variants.

## Supported Datasets
- SST-2
- QNLI
- MNLI
- QQP

## Supported Methods
- Full fine-tuning  
- Last-layer fine-tuning  
- Soft prompt tuning  
- Prefix tuning  
- LoRA (Low-Rank Adaptation)  
- Soft prompt + LoRA  
- Prefix + LoRA  
- (IA)³  

> **Note:** Due to incompatibility between `(IA)³` and `opacus`, the DP-SGD implementation of `(IA)³` is provided separately in `run_ia3_dp.py`.

---

## Usage

### Standard Training (with or without DP)
```bash
python run.py \
  --method prefix \
  --dataset sst2 \
  --batchsize 32 \
  --learning_rate 1e-2 \
  --epsilon -1.0 \
  --grad 1.0 \
  --epochs 60 \
  --p_length 20
```

Set `--epsilon < 0` to disable DP.

Set `--epsilon > 0` to enable DP-SGD.

### (IA)³ with DP-SGD
```bash
python run_ia3_dp.py \
  --dataset sst2 \
  --batchsize 32 \
  --learning_rate 1e-2 \
  --epsilon 8 \
  --grad 1.0 \
  --epochs 60
```

### Hyper-parameter Tuning with WandB Sweeps

This repository includes scripts for hyper-parameter tuning using WandB Sweeps.

1. Generate sweep configurations:
```bash
python create_sweeps.py
```

2. Launch all sweep agents across available GPUs:
```bash
bash run_all_agents.sh
```