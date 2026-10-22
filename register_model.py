from model_registry import ModelRegistry

# 1. Connect to your cluster's internal Model Registry route
# (Find this URL in the RHOAI Dashboard under Model Registry -> your registry name)
registry_url = "https://model-registry-service-my-registry.apps.cluster.com"
registry = ModelRegistry(registry_url)

# 2. Define the Model Name
model_name = "Qwen2.5-0.5B-FinTuned-Adapter"
registered_model = registry.register_model(
    name=model_name,
    description="Qwen2.5 0.5B instruct model fine-tuned on IMDB dataset via QLoRA."
)

# 3. Register the specific version and artifact location
# Point this to the exact S3/MinIO bucket path where KFP dumped your output_model
s3_uri = "s3://my-pipeline-bucket/artifacts/qwen-pipeline/run-1234/train-step/output_model/"

version = registry.register_model_version(
    model=registered_model,
    version="v1.0.0",
    description="Initial fine-tune on 100 IMDB samples, 1 epoch.",
    artifacts={
        "lora_weights": {
            "uri": s3_uri,
            "artifact_type": "model-artifact",
            "properties": {
                "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
                "framework": "pytorch",
                "format": "safetensors",
                "lora_rank": "8"
            }
        }
    }
)

print(f"Successfully registered {model_name} version {version.version} in the Model Registry!")
