apiVersion: v1
kind: Pod
metadata:
  name: model-puller-test
  namespace: jstest
spec:
  restartPolicy: Never
  containers:
    - name: puller
      image: registry.access.redhat.com/ubi9/python-39:latest
      command:
        - /bin/bash
        - -c
        - |
          pip install model-registry --quiet
          python -c '
          from model_registry import ModelRegistry

          registry = ModelRegistry("http://modelmesh-serving:8080", author="pod-reader")
          model = registry.get_registered_model("demo-mnist-model")
          version = registry.get_model_version("demo-mnist-model", "v1.0.0")

          print("--- Model Details Retrieved Successfully ---")
          print(f"Model Name: {model.name}")
          print(f"Version: {version.name}")
          print(f"Storage Path: {version.storage_path}")
          '
