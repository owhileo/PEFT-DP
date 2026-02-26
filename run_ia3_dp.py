# IA3 + Manual DP-SGD Training Script 

import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import math
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)
from peft import IA3Config, get_peft_model, TaskType
from tqdm import tqdm
import wandb
from opacus.accountants import RDPAccountant

# -------------------- Arguments --------------------

def get_args():
    parser = argparse.ArgumentParser(description="IA3 + Manual DP-SGD")
    parser.add_argument("--project", type=str, default="IA3-DP")
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=5)

    parser.add_argument("--grad", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--epsilon", type=float, default=8.0)

    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["sst2", "qnli", "mnli", "qqp"]
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use")

    return parser.parse_args()


# -------------------- DP-SGD Core --------------------

def compute_noise_multiplier(
    target_epsilon,
    target_delta,
    sample_rate,
    epochs,
    steps_per_epoch,
    max_search=100
):
    """
    Solve noise multiplier σ for target (ε, δ).
    """
    accountant = RDPAccountant()
    total_steps = epochs * steps_per_epoch

    low, high = 0.1, 10.0

    for _ in range(max_search):
        mid = (low + high) / 2
        accountant.history = []

        for _ in range(total_steps):
            accountant.step(
                noise_multiplier=mid,
                sample_rate=sample_rate
            )

        eps = accountant.get_epsilon(delta=target_delta)

        if eps > target_epsilon:
            low = mid
        else:
            high = mid

    return high

def dp_sgd_step(model, optimizer, batch, grad, noise_multiplier):
    batch_size = batch["input_ids"].size(0)

    per_sample_grads = []

    for i in range(batch_size):
        optimizer.zero_grad()

        single_batch = {k: v[i:i+1] for k, v in batch.items()}
        outputs = model(**single_batch)
        loss = outputs.loss
        loss.backward()

        grad_dict = {}
        total_norm_sq = 0.0

        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                g = p.grad.detach().clone()
                grad_dict[name] = g
                total_norm_sq += g.norm(2).item() ** 2

        total_norm = math.sqrt(total_norm_sq)
        clip_coef = min(1.0, grad / (total_norm + 1e-6))

        for k in grad_dict:
            grad_dict[k] *= clip_coef

        per_sample_grads.append(grad_dict)

    optimizer.zero_grad()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        grads = [g[name] for g in per_sample_grads]
        stacked = torch.stack(grads, dim=0)
        grad = stacked.mean(dim=0)

        if noise_multiplier > 0:
            noise = torch.randn_like(grad) * (noise_multiplier * grad / batch_size)
            grad += noise

        p.grad = grad

    optimizer.step()


# -------------------- Main --------------------

def main():
    args = get_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    wandb.init(
        project=args.project,
        name=args.run_name if args.run_name else f"{args.dataset}-ia3-dpsgd",
        config=vars(args)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "prajjwal1/bert-tiny"

    dataset = load_dataset("glue", args.dataset)
    num_labels = 3 if args.dataset == "mnli" else 2

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def tokenize_fn(examples):
        if args.dataset == "sst2":
            return tokenizer(examples["sentence"], truncation=True, max_length=128)
        elif args.dataset == "qnli":
            return tokenizer(examples["question"], examples["sentence"], truncation=True, max_length=128)
        elif args.dataset == "mnli":
            return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, max_length=128)
        elif args.dataset == "qqp":
            return tokenizer(examples["question1"], examples["question2"], truncation=True, max_length=128)

    tokenized_ds = dataset.map(tokenize_fn, batched=True)
    tokenized_ds = tokenized_ds.rename_column("label", "labels")
    tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(
        tokenized_ds["train"],
        shuffle=True,
        batch_size=args.batchsize,
        collate_fn=data_collator,
        drop_last=True
    )

    eval_split = "validation_matched" if args.dataset == "mnli" else "validation"
    eval_loader = DataLoader(
        tokenized_ds[eval_split],
        batch_size=args.batchsize,
        collate_fn=data_collator
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_labels
    )

    model = get_peft_model(
        model,
        IA3Config(task_type=TaskType.SEQ_CLS, target_modules=["query", "value", "dense"])
    )

    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    wandb.run.summary["trainable_params"] = trainable_params

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate
    )

    N = len(tokenized_ds["train"])
    steps_per_epoch = len(train_loader)
    q = args.batchsize / N

    noise_multiplier = compute_noise_multiplier(
        target_epsilon=args.epsilon,
        target_delta=args.delta,
        sample_rate=q,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch
    )

    print(f"[DP] Solved noise_multiplier = {noise_multiplier:.4f}")
    wandb.log({"privacy/noise_multiplier": noise_multiplier})

    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            dp_sgd_step(
                model,
                optimizer,
                batch,
                grad=args.grad,
                noise_multiplier=noise_multiplier
            )

            global_step += 1

        # Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)

        acc = correct / total
        wandb.log({"eval/accuracy": acc, "epoch": epoch + 1}, step=global_step)
        print(f"[Epoch {epoch+1}] Accuracy={acc:.4f}")

    wandb.finish()


if __name__ == "__main__":
    main()
