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


## Results

**Tiny-BERT with $\epsilon$=-1 (No Privacy)**

| Dataset      | soft-prompt | prefix  | LoRA    | full-finetuning | last-layer-finetuning | soft-prompt + LoRA | prefix + LoRA | (IA)(^3) |
| ------------ | ----------- | ------- | ------- | --------------- | --------------------- | ------------------ | ------------- | -------- |
| SST2         | 0.79931     | 0.81422 | 0.81995 | 0.80275         | 0.69151               | 0.80963            | 0.81422       | 0.77408  |
| QNLI         | 0.74135     | 0.73586 | 0.80761 | 0.69833         | 0.60260               | 0.80267            | 0.81384       | 0.77229  |
| QQP          | 0.75469     | 0.79978 | 0.80893 | 0.85404         | 0.82508               | 0.85966            | 0.85939       | 0.75417  |
| MNLI         | 0.54386     | 0.58563 | 0.61498 | 0.65441         | 0.43148               | 0.64442            | 0.63515       | 0.54030  |


**Tiny-BERT with $\epsilon$=8**

| Dataset      | soft-prompt | prefix  | LoRA    | full-finetuning | last-layer-finetuning | soft-prompt + LoRA | prefix + LoRA | (IA)(^3) |
| ------------ | ----------- | ------- | ------- | --------------- | --------------------- | ------------------ | ------------- | -------- |
| SST2         | 0.74197     | 0.75000 | 0.78555 | 0.79817         | 0.68349               | 0.77179            | 0.78899       | 0.76261  |
| QNLI         | 0.69815     | 0.70218 | 0.77906 | 0.59564         | 0.60077               | 0.76258            | 0.77558       | 0.76588  |
| QQP          | 0.72753     | 0.75434 | 0.79127 | 0.73309         | 0.63586               | 0.78875            | 0.78922       | 0.74892  |
| MNLI         | 0.45644     | 0.51258 | 0.59521 | 0.53785         | 0.38288               | 0.60795            | 0.60754       | 0.52277  |
