#!/usr/bin/env python3
"""
GSM8K Benchmark Script - Fixed Version
Removes buggy model kwargs and applies necessary patches
"""
import unsloth
# PATCH FIRST - This fixes the num_logits_to_keep error
import transformers.generation.utils as gen_utils

original_prepare = gen_utils.GenerationMixin._prepare_generation_config

def patched_prepare_generation_config(self, generation_config, use_model_defaults, **kwargs):
    config, model_kwargs = original_prepare(self, generation_config, use_model_defaults, **kwargs)
    model_kwargs.pop('num_logits_to_keep', None)  # Remove the problematic parameter
    return config, model_kwargs

gen_utils.GenerationMixin._prepare_generation_config = patched_prepare_generation_config
print("✅ Applied num_logits_to_keep fix")

# NOW your original imports and code
import os
import torch
import json
import re
from unsloth import FastLanguageModel
from datasets import load_dataset
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
FINAL_MODEL_PATH = "/home/ai_pc_user/gemma-grpo-project/Section-C/Elite-Math-Thinking-Merged"
RESULTS_FILE = "full_gsm8k_benchmark_results.jsonl"  # Saves progress here

# The EXACT system prompt used during training and successful inference
THINKING_SYSTEM_PROMPT = """You are a helpful assistant who thinks step by step through problems. When solving questions, show your reasoning process clearly using <think> tags, work through each step methodically, and then provide a clear final answer."""

# --- Helper Functions (Proven to work) ---
def load_model(model_path):
    print(f"\n🔧 Loading final merged model: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype="bfloat16",
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def extract_numerical_answer(text):
    boxed_match = re.search(r'\\boxed\{([\d\.\,]+)\}', text)
    if boxed_match:
        try: 
            return float(boxed_match.group(1).replace(',', ''))
        except ValueError: 
            pass
            
    numbers = re.findall(r'[\d,]*\.?\d+', text)
    if numbers:
        try: 
            return float(numbers[-1].replace(',', ''))
        except ValueError: 
            return None
    return None

# --- Main Benchmark Execution ---
if __name__ == "__main__":
    print("🚀 Starting FULL GSM8K Benchmark for Your Elite Model 🚀")
    model, tokenizer = load_model(FINAL_MODEL_PATH)
    
    print("\n--- Downloading full GSM8K test set ---")
    dataset = load_dataset("gsm8k", "main", split="test")
    print(f"✅ Loaded {len(dataset)} test questions.")
    
    # --- Resumable Logic ---
    processed_indices = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_indices.add(data['index'])
                except json.JSONDecodeError:
                    continue  # Skip corrupted lines
        print(f"✅ Resuming benchmark. Found {len(processed_indices)} previously completed samples.")
    
    # --- Run Evaluation Loop ---
    print("\n--- Starting evaluation loop (progress is saved after each question) ---")
    with open(RESULTS_FILE, 'a') as f:  # Open in append mode
        for i, example in tqdm(enumerate(dataset), total=len(dataset), desc="GSM8K Full Benchmark"):
            if i in processed_indices:
                continue  # Skip questions that are already done
            
            question = example['question']
            
            messages = [
                {"role": "system", "content": THINKING_SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ]
            
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to("cuda")
            
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            raw_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            model_answer_text = raw_response.split('[/INST]')[-1].strip()
            
            pred_answer = extract_numerical_answer(model_answer_text)
            ref_answer = extract_numerical_answer(example['answer'])
            
            is_correct = False
            if pred_answer is not None and ref_answer is not None:
                if abs(pred_answer - ref_answer) < 1e-3:
                    is_correct = True
            
            # Save the result for this single question immediately
            result_data = {
                "index": i,
                "question": question,
                "model_response": model_answer_text,
                "correct_answer_full": example['answer'],
                "predicted_answer_num": pred_answer,
                "reference_answer_num": ref_answer,
                "is_correct": is_correct
            }
            f.write(json.dumps(result_data) + '\n')
            f.flush()
    
    # --- Final Score Calculation ---
    print("\n\n--- ✅ Benchmark Complete! Calculating final score... ---")
    total_questions = 0
    correct_answers = 0
    with open(RESULTS_FILE, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                total_questions += 1
                if data.get('is_correct', False):
                    correct_answers += 1
            except json.JSONDecodeError:
                continue
    
    if total_questions > 0:
        accuracy = (correct_answers / total_questions) * 100
        print(f"\n--- 🏆 FINAL SCORE on Full GSM8K Test Set ({total_questions} samples) ---")
        print(f"Correct: {correct_answers} / {total_questions}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"\n🎯 Your Elite Math + Thinking Model Performance:")
        if accuracy >= 85:
            print("   🌟 OUTSTANDING! World-class mathematical reasoning!")
        elif accuracy >= 75:
            print("   ⭐ EXCELLENT! Very strong mathematical performance!")
        elif accuracy >= 65:
            print("   ✅ GOOD! Solid reasoning capabilities!")
        elif accuracy >= 50:
            print("   📈 FAIR! Shows mathematical understanding!")
        else:
            print("   🔧 NEEDS IMPROVEMENT! Consider additional training!")
    else:
        print("No results found to calculate score.")
    
    print(f"\n🎉 Benchmark completed! Results saved to: {RESULTS_FILE}")
