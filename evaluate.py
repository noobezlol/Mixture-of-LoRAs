import torch
import gc
import os
import json
from unsloth import FastLanguageModel
from peft import PeftModel
from tqdm import tqdm

# ==============================================================================
#  CONFIGURATION
# ==============================================================================
# Path to your fine-tune
YOUR_LORA_PATH = "/home/aurduinonucleo/mixture-of-loras/Section-D/Universal-Code-Master/final_model"
OUTPUT_FILE = "custom_50_benchmark_results.txt"

# Model IDs
MODEL_ID_BASE = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
MODEL_ID_QWEN = "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit"

# --- DISTINCT SYSTEM PROMPTS (FAIR FIGHT) ---

# 1. Base Model: Generic helpful assistant
PROMPT_BASE = "You are a helpful AI assistant. Solve the user's problem clearly and concisely."

# 2. Qwen 2.5 Coder: Expert Developer (Plays to its strength)
PROMPT_QWEN = "You are an expert software developer. Write high-quality, efficient, and bug-free code to solve the user's request. Provide a brief explanation."

# 3. Your Model: The "Thinking" Process (Plays to your fine-tuning)
PROMPT_MINE = """You are an elite software engineer who writes syntactically perfect, logically sound code across all programming languages.

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
#  THE "CLEAN-REASONING" 50-QUESTION DATASET
# ==============================================================================
BENCHMARK_DATASET = [
    # --- CATEGORY 1: CONSTRAINT TRAPS ---
    {"id": 1, "cat": "Constraint",
     "prompt": "Write a Python function `weird_sort(nums)` that sorts a list of integers, BUT odd numbers must remain in their original index positions. Only even numbers should be sorted in ascending order."},
    {"id": 2, "cat": "Constraint",
     "prompt": "Implement a `limited_stack` class. It works like a normal stack, but if the sum of elements inside exceeds 100, the `push` operation should automatically `pop` elements from the bottom until the sum is under 100."},
    {"id": 3, "cat": "Constraint",
     "prompt": "Write a function to reverse a string, but keep all vowels in their original positions. Example: 'design' -> 'nisedg'."},
    {"id": 4, "cat": "Constraint",
     "prompt": "Create a function that generates the Fibonacci sequence up to N, but replace any prime number in the sequence with the string 'PRIME'."},
    {"id": 5, "cat": "Constraint",
     "prompt": "Write a Python dictionary comprehension that maps numbers 1 to 10 to their squares, but ONLY if the square is an even number. If the square is odd, map it to 0."},
    {"id": 6, "cat": "Constraint",
     "prompt": "Write a function that takes a list of strings and sorts them based on the *third* letter of each word. If a word is shorter than 3 letters, put it at the end."},
    {"id": 7, "cat": "Constraint",
     "prompt": "Implement a counter that counts from 1 to 100, but skips numbers that contain the digit '3' (e.g., 3, 13, 30-39)."},
    {"id": 8, "cat": "Constraint",
     "prompt": "Write a function `merge_alternating(list1, list2)` that merges two lists by taking elements alternately. If one list runs out, reverse the remaining elements of the other list and append them."},
    {"id": 9, "cat": "Constraint",
     "prompt": "Create a class `OneTimeDict`. It behaves like a dictionary, but once a key is read (accessed), that key is automatically deleted."},
    {"id": 10, "cat": "Constraint",
     "prompt": "Write a function that finds the maximum number in a list, but you are NOT allowed to use `max()`, `sorted()`, `sort()`, or any comparison operators like `>` or `<`. (Hint: use subtraction and absolute values)."},

    # --- CATEGORY 2: FORMAT TORTURE ---
    {"id": 11, "cat": "Format",
     "prompt": "Write a Python Hello World script. However, you MUST NOT use the string 'Hello World' directly. You must construct it using ASCII character codes and the `chr()` function."},
    {"id": 12, "cat": "Format",
     "prompt": "Generate a Python function to calculate factorial. BUT, the entire function body must be written on a single line using lambda functions. No `def` allowed."},
    {"id": 13, "cat": "Format",
     "prompt": "Write a standard binary search function. However, all variable names must be fruits (e.g., 'apple' for left, 'banana' for right)."},
    {"id": 14, "cat": "Format",
     "prompt": "Create a JSON object representing a user profile. The keys must be 'u_id', 'u_name', and 'u_age'. Do NOT write any Python code, just output the raw JSON string inside markdown code blocks."},
    {"id": 15, "cat": "Format",
     "prompt": "Write a Python comment block describing the Theory of Relativity. Do not write any executable code."},
    {"id": 16, "cat": "Format",
     "prompt": "Write a SQL query to select users, but format the SQL query as a single Python string variable named `sql_query`. Do not explain the query."},
    {"id": 17, "cat": "Format",
     "prompt": "Write a Python function to add two numbers, but you must use a nested function architecture (a closure) to achieve it."},
    {"id": 18, "cat": "Format",
     "prompt": "Output the CSS code to center a div, but you must use Grid, not Flexbox. Provide ONLY the CSS."},
    {"id": 19, "cat": "Format",
     "prompt": "Write a Python list comprehension that produces the first 10 even numbers. However, you must wrap the list comprehension in a `try-except` block within the function."},
    {"id": 20, "cat": "Format",
     "prompt": "Write a bash script to list files, but all comments in the script must be written in French."},

    # --- CATEGORY 3: REAL WORLD MESSY ---
    {"id": 21, "cat": "Real World",
     "prompt": "I have a log string: '[ERROR] 2023-10-05 User:admin caused:Timeout'. Write a regex to extract the severity (ERROR), date, username, and error type into a dictionary."},
    {"id": 22, "cat": "Real World",
     "prompt": "Write a Pandas one-liner to filter a DataFrame `df`. Keep rows where column 'A' is greater than 10 OR column 'B' is less than 5, AND column 'C' is not Null."},
    {"id": 23, "cat": "Real World",
     "prompt": "Write a python script to rename all files in a folder. If a file is 'image.jpg', rename it to 'img_001.jpg', 'img_002.jpg', etc. It must handle padding zeros correctly based on the total file count."},
    {"id": 24, "cat": "Real World",
     "prompt": "Implement a 'rate_limiter' decorator in Python. It should allow a function to be called only 5 times every 10 seconds. If exceeded, raise an exception."},
    {"id": 25, "cat": "Real World",
     "prompt": "Write a function to validate a credit card number using the Luhn algorithm."},
    {"id": 26, "cat": "Real World",
     "prompt": "Given a list of dirty phone numbers like ['123-456-7890', '(123) 456 7890', '123.456.7890'], write a function to normalize them all to '1234567890'."},
    {"id": 27, "cat": "Real World",
     "prompt": "Parse a CSV string manually without using the `csv` library. The string deals with quoted fields containing commas. Example line: `1, \"Apple, Red\", $1.00`."},
    {"id": 28, "cat": "Real World",
     "prompt": "Write a function that takes a URL and returns the domain name (e.g., 'https://www.google.com/search' -> 'google.com'). Handle subdomains correctly."},
    {"id": 29, "cat": "Real World",
     "prompt": "Implement a simple exponential backoff strategy for a failing network request using a while loop."},
    {"id": 30, "cat": "Real World",
     "prompt": "Write a function to convert a nested JSON object into a flat dictionary where keys are separated by dots (e.g., {'a': {'b': 1}} -> {'a.b': 1})."},

    # --- CATEGORY 4: ALGORITHMIC & CREATIVE ---
    {"id": 31, "cat": "Creative",
     "prompt": "Write a Python script to print a pyramid of stars of height 5. It must be centered."},
    {"id": 32, "cat": "Creative",
     "prompt": "Write code to generate a random maze using Depth First Search (DFS). Represent the maze using '#' for walls and ' ' for paths."},
    {"id": 33, "cat": "Creative",
     "prompt": "Simulate a text-based traffic light system. It should loop forever, printing 'RED', waiting 3s, 'GREEN', waiting 3s, 'YELLOW', waiting 1s."},
    {"id": 34, "cat": "Creative",
     "prompt": "Write a Python class `VirtualPet`. It has `hunger` and `energy`. Methods: `feed()` decreases hunger, `play()` decreases energy. If energy < 0, it sleeps."},
    {"id": 35, "cat": "Creative",
     "prompt": "Write a function that takes a sentence and prints it out vertically, like a banner."},
    {"id": 36, "cat": "Algo",
     "prompt": "Implement the 'Sieve of Eratosthenes' to find all primes up to N, but optimize it to use a bitarray instead of a list of booleans for memory efficiency."},
    {"id": 37, "cat": "Algo",
     "prompt": "Write a function to check if two strings are anagrams, but you must do it in O(n) time and O(1) space (assuming fixed alphabet size)."},
    {"id": 38, "cat": "Algo",
     "prompt": "Implement a 'MinStack' that supports push, pop, top, and retrieving the minimum element in constant time O(1)."},
    {"id": 39, "cat": "Algo",
     "prompt": "Write a function to find the longest substring without repeating characters in a string."},
    {"id": 40, "cat": "Algo",
     "prompt": "Implement a basic version of the 'cd' (change directory) command logic. Given a current path and a command (like '../abc/./def'), resolve the new path."},

    # --- CATEGORY 5: SECURITY & EDGE CASES ---
    {"id": 41, "cat": "Security",
     "prompt": "Write a SQL query construction to select a user by ID, but protect it against SQL Injection without using an ORM. Use parameterized queries logic."},
    {"id": 42, "cat": "Security",
     "prompt": "Write a Python function to safely delete a file. It must check if the file exists and ensure the path is not outside the allowed directory (prevent directory traversal)."},
    {"id": 43, "cat": "Edge Case",
     "prompt": "Write a function to calculate the average of a list. Handle the edge cases: empty list, list with None values, and list with strings."},
    {"id": 44, "cat": "Edge Case",
     "prompt": "Implement division of two numbers, but handle division by zero gracefully by returning None, and ensure floating point precision issues are minimized."},
    {"id": 45, "cat": "Edge Case",
     "prompt": "Write a function to parse a date string 'YYYY-MM-DD'. Handle invalid dates like '2023-02-30' (February 30th) without crashing."},
    {"id": 46, "cat": "Security",
     "prompt": "Write a function to generate a secure random token of 32 bytes, encoded in URL-safe Base64."},
    {"id": 47, "cat": "Edge Case",
     "prompt": "Merge two dictionaries. If a key exists in both, the value should become a list containing both values. Handle cases where the original value is already a list."},
    {"id": 48, "cat": "Security",
     "prompt": "Sanitize a user input string to remove any potential HTML tags to prevent XSS attacks."},
    {"id": 49, "cat": "Algo",
     "prompt": "Write a function to detect if a binary tree is balanced. Return True or False."},
    {"id": 50, "cat": "Creative",
     "prompt": "Write a Python script that simulates a simple ATM state machine (Idle -> PIN -> Menu -> Withdraw -> Dispense). Use a while loop and user input."}
]


# ==============================================================================
#  STEP 2: INFERENCE ENGINE
# ==============================================================================
def generate_batch(model_name, model, tokenizer, system_prompt, tasks, use_stop_token=False):
    responses = []
    print(f"\n🚀 Generating responses for: {model_name}")

    stop_ids = [tokenizer.eos_token_id]
    if use_stop_token:
        stop_ids.append(tokenizer.convert_tokens_to_ids("[END]"))

    for task in tqdm(tasks, desc=f"Running {model_name}"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task['prompt']}
        ]
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=1200,
                temperature=0.1,
                do_sample=True,
                eos_token_id=stop_ids
            )

        decoded = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        if use_stop_token:
            decoded = decoded.replace("[END]", "")

        responses.append(decoded.strip())

    return responses


# ==============================================================================
#  STEP 3: RUNNING THE GAUNTLET
# ==============================================================================
results_storage = {"base": [], "qwen": [], "mine": []}

# --- RUN 1: BASE LLAMA 3.2 ---
print("\n🔵 Loading Base Llama 3.2...")
model, tokenizer = FastLanguageModel.from_pretrained(MODEL_ID_BASE, max_seq_length=2048, dtype=None, load_in_4bit=True)
FastLanguageModel.for_inference(model)
results_storage["base"] = generate_batch("Llama 3.2 Base", model, tokenizer, PROMPT_BASE, BENCHMARK_DATASET)

del model, tokenizer
gc.collect()
torch.cuda.empty_cache()

# --- RUN 2: QWEN 2.5 CODER ---
print("\n🟢 Loading Qwen 2.5 Coder...")
model, tokenizer = FastLanguageModel.from_pretrained(MODEL_ID_QWEN, max_seq_length=2048, dtype=None, load_in_4bit=True)
FastLanguageModel.for_inference(model)
results_storage["qwen"] = generate_batch("Qwen 2.5 Coder", model, tokenizer, PROMPT_QWEN, BENCHMARK_DATASET)

del model, tokenizer
gc.collect()
torch.cuda.empty_cache()

# --- RUN 3: YOUR FINE-TUNE ---
print("\n🟣 Loading YOUR Fine-Tuned Model...")
model, tokenizer = FastLanguageModel.from_pretrained(MODEL_ID_BASE, max_seq_length=2048, dtype=None, load_in_4bit=True)
tokenizer.add_special_tokens({"additional_special_tokens": ["[END]"]})
model.resize_token_embeddings(len(tokenizer))

if os.path.isdir(YOUR_LORA_PATH):
    print(f"   - Loading local adapter from: {YOUR_LORA_PATH}")
    model = PeftModel.from_pretrained(model, YOUR_LORA_PATH)
else:
    raise ValueError(f"Directory not found: {YOUR_LORA_PATH}")

FastLanguageModel.for_inference(model)
results_storage["mine"] = generate_batch("Your Expert", model, tokenizer, PROMPT_MINE, BENCHMARK_DATASET,
                                         use_stop_token=True)

# ==============================================================================
#  STEP 4: SAVE FOR GRADING
# ==============================================================================
print(f"\n💾 Saving results to {OUTPUT_FILE}...")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("=== 50-QUESTION CUSTOM BENCHMARK COMPARISON ===\n")
    f.write("Comparison: Base Llama 3.2 vs. Qwen 2.5 Coder vs. Custom Fine-Tune\n")
    f.write("Note: Each model was given a system prompt optimized for its specific training style.\n\n")

    for i, task in enumerate(BENCHMARK_DATASET):
        f.write(f"{'=' * 80}\n")
        f.write(f"PROBLEM {i + 1}: {task['cat']} - {task['prompt'][:60]}...\n")
        f.write(f"FULL PROMPT: {task['prompt']}\n")
        f.write(f"{'=' * 80}\n\n")

        f.write(f"--- MODEL A: Llama 3.2 Base ---\n")
        f.write(f"{results_storage['base'][i]}\n\n")

        f.write(f"--- MODEL B: Qwen 2.5 Coder ---\n")
        f.write(f"{results_storage['qwen'][i]}\n\n")

        f.write(f"--- MODEL C: My Fine-Tune ---\n")
        f.write(f"{results_storage['mine'][i]}\n\n")

        f.write("\n\n")

print("✅ Done! Run the LLM Judge on 'custom_50_benchmark_results.txt'.")