import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#!/usr/bin/env python3
"""
Chat interface with EXPLICIT base model loading + LoRA adapter
"""
import os
from unsloth import FastLanguageModel
from peft import PeftModel
import torch

# Explicit paths
BASE_MODEL = "unsloth/llama-3.2-3b-instruct-bnb-4bit"
LORA_PATH = os.path.join(BASE_DIR, "Final-Dynamic-Model/final_model(Math)")

# EXACT system prompt from training
THINKING_SYSTEM_PROMPT = """You are a helpful assistant who thinks step by step through problems. When solving questions, show your reasoning process clearly using <think> tags, work through each step methodically, and then provide a clear final answer."""

print("🧠 Loading base model explicitly...")

# Step 1: Load BASE MODEL explicitly
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,  # Explicitly specified base
    max_seq_length=1536,
    dtype="bfloat16",
    load_in_4bit=True,
    device_map={"": 0},
    trust_remote_code=True,
)

print(f"📦 Loading LoRA adapter from: {LORA_PATH}")

# Step 2: Apply LoRA adapter on top of base model
model = PeftModel.from_pretrained(
    model,
    LORA_PATH,
    is_trainable=False  # Inference mode
)

# Step 3: Enable fast inference
FastLanguageModel.for_inference(model)

print("✅ Model ready with explicit base + LoRA!\n")
print("🎯 ELITE MATH + THINKING MODEL")
print("=" * 60)

# Chat loop
while True:
    question = input("\n🤔 You: ").strip()

    if question.lower() in ['quit', 'exit', 'q']:
        print("👋 Goodbye!")
        break

    if not question:
        continue

    # USE THE EXACT FORMAT FROM TRAINING
    full_prompt = f"<|system|>\n{THINKING_SYSTEM_PROMPT}\n\n<|user|>\n{question}\n\n<|assistant|>\n"

    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")

    print("\n🤖 Model: ", end="", flush=True)

    from transformers import TextStreamer
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=1024,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )
    print("\n")
