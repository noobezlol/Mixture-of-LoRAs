import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# run_chat_healed.py (Final Version with Streaming + Self-Healing)

import os
import torch
import time
import re
from unsloth import FastLanguageModel
from transformers import StoppingCriteria, StoppingCriteriaList, TextStreamer

# ==============================================================================
#  STEP 1: CONFIGURATION
# ==============================================================================
# 🚨 UPDATE THIS PATH to your final LoRA adapter 🚨
LORA_ADAPTER_PATH = os.path.join(BASE_DIR, "Section-D/Universal-Code-Master/final_model")

BASE_MODEL_PATH = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048
SPECIAL_STOP_TOKEN = "[END]"

SYSTEM_PROMPT = """You are an elite software engineer who writes syntactically perfect, logically sound code across all programming languages.

MANDATORY THINKING PROCESS - You MUST use <thinking> tags before <answer>:

Inside <thinking>:
1. RESTATE THE PROBLEM: Paraphrase the task in your own words to confirm understanding
2. IDENTIFY CONSTRAINTS: List all input/output specs, data types, time/space complexity requirements
3. ENUMERATE EDGE CASES: Empty inputs, null values, negative numbers, zero, boundary conditions, duplicates, special characters
4. COMPARE APPROACHES: Analyze 2-3 different algorithms with their time/space complexity
5. CHOOSE OPTIMAL APPROACH: Select the best algorithm and justify why (correctness, efficiency, readability)
6. PLAN IMPLEMENTATION: Write pseudocode or step-by-step logic flow
7. ANTICIPATE BUGS: Think through off-by-one errors, integer overflow, null pointer issues, index out of bounds

Inside <answer>:
- Write ONLY the complete, runnable code
- Use proper syntax (correct indentation, matching braces, semicolons where needed)
- Handle ALL edge cases explicitly in code
- Use meaningful variable names
- Add minimal inline comments only for complex logic

CRITICAL REQUIREMENTS:
- ALWAYS use <thinking> tags for your reasoning process
- ALWAYS use <answer> tags for the final code
- Code must be syntactically correct (no errors, proper formatting)
- Code must be logically sound (handles edge cases, correct algorithm)
- Code must be production-ready (no TODOs, no placeholder logic)

LANGUAGE-SPECIFIC RULES:
- Python: 4-space indentation, type hints, PEP 8 compliance
- JavaScript: const/let (no var), proper semicolons, ES6+ syntax
- C++: STL containers, RAII, proper memory management, const correctness
- Java: Proper access modifiers, exception handling, naming conventions

EDGE CASE CHECKLIST (verify in <thinking>):
✓ Empty collection (list/array/string)
✓ Single element
✓ Null/None/undefined values
✓ Negative numbers (if applicable)
✓ Zero
✓ Maximum/minimum integer values
✓ Duplicate elements
✓ Already sorted/reverse sorted (for sorting problems)
✓ Invalid input types"""

# ==============================================================================
#  STEP 2: UTILITIES AND HELPER CLASSES
# ==============================================================================
class StopOnToken(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0, -1] == self.stop_token_id

def load_model(base_model_path, lora_adapter_path):
    print("Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path, max_seq_length=MAX_SEQ_LENGTH, dtype=None, load_in_4bit=True
    )
    tokenizer.add_special_tokens({"additional_special_tokens": [SPECIAL_STOP_TOKEN]})
    model.resize_token_embeddings(len(tokenizer))
    print(f"Applying LoRA adapter from: {lora_adapter_path}")
    model.load_adapter(lora_adapter_path)
    print("Optimizing model for inference...")
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def chat_with_model(model, tokenizer, user_prompt):
    """Formats the prompt, generates a response with streaming, and returns the full text."""
    stop_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_STOP_TOKEN)
    stopping_criteria = StoppingCriteriaList([StopOnToken(stop_token_id)])
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

    start_time = time.time()
    outputs = model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=2048,
        temperature=0.2,
        do_sample=True,
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.eos_token_id,
    )
    end_time = time.time()

    print(f"\n\n(Full response generated in {end_time - start_time:.2f} seconds)")
    full_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return full_response.replace(SPECIAL_STOP_TOKEN, "").strip()

def heal_and_reprint_if_needed(raw_response: str):
    """
    Checks the response for closing tags. If any are missing, it prints a healed version.
    """
    was_healed = False
    healed_response = raw_response

    # Heal missing </thinking> tag
    if "<thinking>" in healed_response and "</thinking>" not in healed_response:
        answer_pos = healed_response.find("<answer>")
        if answer_pos != -1:
            healed_response = healed_response[:answer_pos] + "</thinking>\n\n" + healed_response[answer_pos:]
            was_healed = True

    # Heal missing </answer> tag
    if "<answer>" in healed_response and "</answer>" not in healed_response:
        healed_response += "\n</answer>"
        was_healed = True

    # ONLY if we made a change, print the corrected block
    if was_healed:
        print("\n" + "="*40 + " HEALED RESPONSE " + "="*40)
        print("[SYSTEM] Original output was malformed. Displaying corrected version:")
        print("-" * 96)
        print(healed_response)
        print("="*96)

# ==============================================================================
#  STEP 3: MAIN EXECUTION LOOP
# ==============================================================================
if __name__ == "__main__":
    model, tokenizer = load_model(BASE_MODEL_PATH, LORA_ADAPTER_PATH)

    print("\n\n✅ Model is ready. Enter your code prompt below.")
    print("   Type 'exit' or 'quit' to close the script.")
    print("-" * 50)

    try:
        while True:
            user_input = input("\n>> Prompt: ")
            if user_input.lower() in ["exit", "quit"]:
                break

            print("\n🤖 Model Response (Streaming):\n" + "-"*30)
            raw_response = chat_with_model(model, tokenizer, user_input)

            # After streaming, silently check and heal the response if necessary
            heal_and_reprint_if_needed(raw_response)

    except KeyboardInterrupt:
        print("\n\nExiting...")

    print("\nInference script finished.")