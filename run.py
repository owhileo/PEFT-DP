import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)
from peft import (
    get_peft_model, LoraConfig, PrefixTuningConfig,
    PromptTuningConfig, IA3Config, TaskType,
    inject_adapter_in_model
)
from tqdm import tqdm
import wandb
from opacus import PrivacyEngine


# -------------------- Arguments --------------------

def get_args():
    parser = argparse.ArgumentParser(description="PEFT + Opacus DP-SGD on GLUE")
    parser.add_argument("--project", type=str, default="PEFT-DP-GLUE")
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument("--epsilon", type=float, default=-1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--grad", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--p_length", type=int, default=20)
    parser.add_argument(
        "--method", type=str, required=True,
        choices=[
            "soft-prompt", "prefix", "lora", "full-finetuning",
            "last-layer-finetuning", "soft-prompt+lora",
            "prefix+lora", "ia3"
        ]
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["sst2", "qnli", "mnli", "qqp"]
    )
    return parser.parse_args()


# -------------------- Main --------------------

def main():
    args = get_args()

    wandb.init(
        project=args.project,
        name=args.run_name if args.run_name else f"{args.dataset}-{args.method}-eps{args.epsilon}",
        config=vars(args)
    )

    model_id = "prajjwal1/bert-tiny"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # -------------------- PEFT --------------------

    if args.method == "full-finetuning":
        model.bert.embeddings.position_embeddings.weight.requires_grad = False

    elif args.method == "last-layer-finetuning":
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    elif args.method == "lora":
        model = get_peft_model(
            model, LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16)
        )

    elif args.method == "soft-prompt":
        model = get_peft_model(
            model, PromptTuningConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=args.p_length)
        )

    elif args.method == "prefix":
        model = get_peft_model(
            model, PrefixTuningConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=args.p_length)
        )

    elif args.method == "ia3":
        model = get_peft_model(
            model, IA3Config(task_type=TaskType.SEQ_CLS, target_modules=["query", "value", "dense"])
        )

    elif args.method == "soft-prompt+lora":
        lora_config = LoraConfig(r=8, target_modules=["query", "value"])
        model = inject_adapter_in_model(lora_config, model)

        prompt_config = PromptTuningConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=args.p_length)
        model = get_peft_model(model, prompt_config)

        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

    elif args.method == "prefix+lora":
        lora_config = LoraConfig(r=8, target_modules=["query", "value"])
        model = inject_adapter_in_model(lora_config, model)

        prefix_config = PrefixTuningConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=args.p_length)
        model = get_peft_model(model, prefix_config)

        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.run.summary["trainable_params"] = trainable_params


    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print("Trainable params:", sum(p.numel() for p in trainable_params))

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    # -------------------- Opacus DP --------------------

    privacy_engine = None
    if args.epsilon > 0:
        privacy_engine = PrivacyEngine()

        delta=args.delta if args.delta>0 else 1 / len(train_loader.dataset)

        model.train()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=args.epochs,
            target_epsilon=args.epsilon,
            target_delta=delta,
            max_grad_norm=args.grad,
        )

        print(f"[DP] Using Opacus | ε={args.epsilon}, δ={delta}")

    # -------------------- Training --------------------

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch in pbar:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            global_step += 1
            wandb.log({"train/loss": loss.item()}, step=global_step)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # -------------------- Eval --------------------

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

        if privacy_engine is not None:
            eps = privacy_engine.get_epsilon(args.delta)
            wandb.log({"privacy/epsilon": eps}, step=global_step)
            print(f"[Epoch {epoch+1}] Accuracy={acc:.4f}, ε={eps:.3f}")
        else:
            print(f"[Epoch {epoch+1}] Accuracy={acc:.4f}")

    wandb.finish()


if __name__ == "__main__":
    main()