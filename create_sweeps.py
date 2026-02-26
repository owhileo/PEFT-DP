import wandb
import sys

# Basic configuration
project_name = "PEFT-GLUE-DeepSearch"

# Automatically get the current Python executable
python_executable = sys.executable

datasets = ["sst2", "qnli", "mnli", "qqp"]
methods = ["prefix", "soft-prompt", "soft-prompt+lora", "prefix+lora"]

# Common sweep configuration
sweep_config = {
    "method": "bayes",
    "metric": {"name": "eval/accuracy", "goal": "maximize"},
    # Specify the execution command
    "command": [
        python_executable,  # Use the current environment's Python
        "run.py",
        "${args}"
    ],
    "parameters": {
        "learning_rate": {"values": [1e-2, 5e-2, 5e-3, 1e-3, 3e-2, 3e-3]},
        "p_length": {"values": [1, 10, 20, 50, 100]},
        "batchsize": {"values": [32, 128, 512, 1024, 2048]},
        "epsilon": {"value": -1.0},
        "epochs": {"value": 100}
    }
}

sweep_ids = []

# Get your WandB entity (make sure `wandb login` has been run)
entity = wandb.api.default_entity

for ds in datasets:
    for mt in methods:
        config = sweep_config.copy()
        config["name"] = f"search_{mt}_{ds}"
        config.update({"parameters": sweep_config["parameters"].copy()})  # Deep copy to avoid conflicts
        config["parameters"]["dataset"] = {"value": ds}
        config["parameters"]["method"] = {"value": mt}

        # Create sweep
        sweep_id = wandb.sweep(config, project=project_name)

        # Save full path: entity/project/sweep_id
        full_id = f"{entity}/{project_name}/{sweep_id}"
        sweep_ids.append(full_id)

# Save all sweep IDs
with open("sweep_ids.txt", "w") as f:
    for s_id in sweep_ids:
        f.write(f"{s_id}\n")

print(f"\nSuccessfully created {len(sweep_ids)} sweep jobs!")
print(f"Using Python executable: {python_executable}")
print("Full sweep IDs have been saved to sweep_ids.txt")