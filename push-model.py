from model_registry import ModelRegistry

# Initialize connection to your Model Registry service
# Replace with your Model Registry service URL or internal cluster route
registry = ModelRegistry("http://modelmesh-serving:8080", author="developer")

# 1. Register a Model Name
registered_model = registry.register_model(
    "demo-mnist-model",
    description="A sample MNIST model for testing the registry",
)

# 2. Register a Model Version & Storage Location
model_version = registry.register_model_version(
    registered_model,
    version="v1.0.0",
    author="developer",
    description="Initial release",
    model_format_name="onnx",
    model_format_version="1",
    storage_key="aws-connection-s3", # Name of your Data Connection / Secret
    storage_path="models/mnist/v1/model.onnx", # S3 key / URI
)

print(f"Successfully registered model version: {model_version.name}")
