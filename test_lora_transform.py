# Run this in Jupyter Notebook, using PyTorch.  The output from the AI dashboard
# must be downloaded to the adapter_path below.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 1. Define paths
base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_path = "./my-qwen-adapter"  # Path to your extracted KFP output folder

print("Loading base model in 4-bit...")
# 2. Configure 4-bit quantization (Must match training setup)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# 3. Load the base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

print("Attaching LoRA adapter...")
# 4. Attach your fine-tuned adapter to the base model
model = PeftModel.from_pretrained(base_model, adapter_path)

print("Model ready! Generating response...")
# 5. Run a test inference
prompt = "Analyze this movie review and determine if it is positive or negative: 'The cinematography was beautiful, but the plot was completely nonsensical.'"

# Format the prompt using Qwen's chat template
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Tokenize and push to GPU
inputs = tokenizer(text_input, return_tensors="pt").to("cuda")

# Generate output
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True
    )

# Decode and print the result
response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print("\n--- Output ---")
print(response)

