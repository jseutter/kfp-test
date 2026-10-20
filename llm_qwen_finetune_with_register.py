from kfp import dsl, compiler
from kfp.client import Client

# ----------------------------------------------------------------------
# STEP 1: Component - Fine-Tune Qwen LLM using QLoRA on NVIDIA GPU
# ----------------------------------------------------------------------
@dsl.component(
    base_image="quay.io/opendatahub-contrib/workbench-images:cuda-runtime-pytorch-c9s-py311_2023c_latest"
    # Note: packages_to_install is intentionally removed to handle it via subprocess below
)
def train_qwen_qlora_step(
    model_id: str,
    dataset_name: str,
    output_dir: str,
    epochs: int,
    output_model: dsl.Output[dsl.Model]
):
    import subprocess
    import sys
    
    print("--- Purging mixed C-extensions and reinstalling dependencies ---")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-cache-dir",
        "numpy==1.26.4", 
        "pyarrow<17.0.0",
        "pandas", 
        "transformers", 
        "datasets", 
        "peft", 
        "bitsandbytes", 
        "accelerate", 
        "trl"
    ])

    import os
    import torch
    import numpy as np
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments
    )
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer

    print(f"NumPy cleanly initialized: {np.__version__}")
    print("--- Checking GPU Availability ---")
    
    if torch.cuda.is_available():
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"Device Count: {torch.cuda.device_count()}")
    else:
        raise RuntimeError("No NVIDIA GPU detected! Ensure GPU tolerations/requests are set.")

    # 1. 4-bit Quantization Config (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # 2. Load Base Model & Tokenizer
    print(f"Loading base model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. Apply PEFT / LoRA Adapter
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 4. Load Sample Dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train[:100]")  # Subsampled for rapid demonstration

    # 5. Define Training Configuration
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="no"
    )

    # 6. Execute Fine-Tuning
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("--- Starting Fine-Tuning Job ---")
    trainer.train()

    # 7. Save Artifact Output path for KFP tracking
    print(f"Saving LoRA adapter to KFP Output path: {output_model.path}")
    trainer.model.save_pretrained(output_model.path)
    tokenizer.save_pretrained(output_model.path)


# ----------------------------------------------------------------------
# STEP 2: Component - Register Model into RHOAI Model Registry
# ----------------------------------------------------------------------
@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=["model-registry"]
)
def register_model_step(
    trained_model: dsl.Input[dsl.Model],
    model_name: str,
    version_name: str,
    registry_url: str
):
    from model_registry import ModelRegistry
    import os
    
    print(f"Connecting to Model Registry at: {registry_url}")
    token = os.environ.get("SA_TOKEN", "") 
    registry = ModelRegistry(server_address=registry_url, author="kfp-pipeline", token=token)

    print(f"Registering Model family: {model_name}")
    registered_model = registry.register_model(
        name=model_name,
        description="Automated fine-tune pipeline output"
    )

    print(f"Registering Version: {version_name} mapping to S3 path: {trained_model.uri}")
    registry.register_model_version(
        model=registered_model,
        version=version_name,
        description="Qwen QLoRA Adapter",
        artifacts={
            "model_weights": {
                "uri": trained_model.uri,
                "artifact_type": "model-artifact",
                "properties": {
                    "framework": "pytorch",
                    "format": "safetensors",
                    "pipeline_run": "true"
                }
            }
        }
    )
    print("Model successfully registered!")


# ----------------------------------------------------------------------
# STEP 3: Pipeline Definition with NVIDIA GPU Request
# ----------------------------------------------------------------------
@dsl.pipeline(
    name="qwen-llm-finetuning-pipeline",
    description="Fine-tune a Qwen LLM using QLoRA on an NVIDIA GPU cluster node."
)
def qwen_llm_pipeline(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    dataset_name: str = "imdb",
    epochs: int = 1
):
    # Instantiate the training step
    train_task = train_qwen_qlora_step(
        model_id=model_id,
        dataset_name=dataset_name,
        output_dir="/tmp/qwen-output",
        epochs=epochs
    )
    # Data is saved as an additional layer in a folder injected into this script.
    # A background pod will package and upload the files to S3.  View
    # using Run, click the node, and open I/O tab

    # CRITICAL FOR OPENSHIFT AI: Request 1x NVIDIA GPU resource
    train_task.set_gpu_limit(1)  # Limits & Requests 1 nvidia.com/gpu
    
    # Request adequate CPU & Memory for model loading
    train_task.set_cpu_limit("4")
    train_task.set_memory_limit("16Gi")


# ----------------------------------------------------------------------
# STEP 4: Compile Pipeline to YAML
# ----------------------------------------------------------------------
if __name__ == "__main__":
    compiled_yaml = "qwen_llm_pipeline.yaml"
    print(f"Compiling pipeline to {compiled_yaml}...")
    compiler.Compiler().compile(
        pipeline_func=qwen_llm_pipeline,
        package_path=compiled_yaml
    )
    print("Compilation complete!")

