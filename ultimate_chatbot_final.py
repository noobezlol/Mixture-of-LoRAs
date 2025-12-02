# ultimate_chatbot_v6_isolated.py

import os
import unsloth
import torch
import time
import re
import os
from pathlib import Path
import copy  # Added for deepcopy
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline, StoppingCriteria, StoppingCriteriaList, TextStreamer
from unsloth import FastLanguageModel

# ==============================================================================
#  STEP 1: MASTER CONFIGURATION
# --- Automatic Path Setup (Docker & Portable Ready) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# --- Paths ---

# 1. Define Base Directory
# 1. Define Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Define Mapping: Local Folder -> Hugging Face ID
MODEL_MAP = {
    "router": {
        "local": os.path.join(BASE_DIR, "Classifier"),
        "cloud": "Ishaanlol/MoL-Router-DistilBERT"
    },
    "code": {
        "local": os.path.join(BASE_DIR, "Section-D/Universal-Code-Master/final_model"),
        "cloud": "Ishaanlol/MoL-Code-Expert-LoRA"
    },
    "math": {
        "local": os.path.join(BASE_DIR, "Final-Dynamic-Model/final_model(Math)"),
        "cloud": "Ishaanlol/MoL-Math-Expert-LoRA"
    }
}

# 3. Helper function to select the best path
def get_path(key):
    local_p = MODEL_MAP[key]["local"]
    if os.path.exists(local_p):
        print(f"   [Config] Found local model for '{key}': {local_p}")
        return local_p
    else:
        cloud_id = MODEL_MAP[key]["cloud"]

                if section.strip(): self.documents.append(section.strip())
        if not self.documents: return

        embeddings = self.embedding_model.encode(self.documents, show_progress_bar=False, convert_to_numpy=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings.astype('float32'))

    def retrieve(self, query, top_k=2):
        if not self.index or not self.documents: return []
        print(f"[{self.name} RAG System] Retrieving top-{top_k} documents...")
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True).astype('float32')
        _, indices = self.index.search(query_embedding, top_k)
        retrieved_docs = [self.documents[idx] for idx in indices[0] if idx < len(self.documents)]
        print(f"[{self.name} RAG System] -> Found {len(retrieved_docs)} relevant documents.")
        return retrieved_docs


class StopOnToken(StoppingCriteria):
    def __init__(self, stop_token_id): self.stop_token_id = stop_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0, -1] == self.stop_token_id


def load_expert_model(expert_name):
    """Lazy-loads an expert model into the cache the first time it's needed."""
    if model_cache.get(expert_name): return model_cache[expert_name]
    print(f"\n[System] Loading '{expert_name}' expert... (This may take a moment)")

    if expert_name == "other":
        model, tokenizer = FastLanguageModel.from_pretrained(model_name=BASE_MODEL_PATH, max_seq_length=MAX_SEQ_LENGTH,
                                                             dtype=None, load_in_4bit=True)
        FastLanguageModel.for_inference(model)
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(model_name=BASE_MODEL_PATH, max_seq_length=MAX_SEQ_LENGTH,
                                                             dtype=None, load_in_4bit=True)
        lora_path = CODE_LORA_PATH if expert_name == "code" else MATH_LORA_PATH
        if expert_name == "code":
            tokenizer.add_special_tokens({"additional_special_tokens": [CODE_STOP_TOKEN]})
            model.resize_token_embeddings(len(tokenizer))
        model.load_adapter(lora_path)
        FastLanguageModel.for_inference(model)

    model_cache[expert_name] = (model, tokenizer)
    print(f"[System] ✅ '{expert_name}' expert is now loaded.")
    return model_cache[expert_name]


def heal_and_reprint_if_needed(raw_response):
    was_healed, healed_response = False, raw_response
    if "<thinking>" in healed_response and "</thinking>" not in healed_response:
        answer_pos = healed_response.find("<answer>")
        if answer_pos != -1:
            healed_response = healed_response[:answer_pos] + "</thinking>\n\n" + healed_response[answer_pos:]
            was_healed = True
    if "<answer>" in healed_response and "</answer>" not in healed_response:
        healed_response += "\n</answer>"
        was_healed = True
    if was_healed:
        print("\n" + "=" * 40 + " HEALED RESPONSE " + "=" * 40)
        print("[SYSTEM] Original output was malformed. Displaying corrected version:")
        print(healed_response)
        print("=" * 96)


# ==============================================================================
#  STEP 3: ISOLATED EXPERT HANDLERS (Definitive Final Version)
# ==============================================================================
def handle_code_query(chat_history, rag_context=""):
    """Handles code queries with expert-specific history."""
    model, tokenizer = load_expert_model("code")
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # CRITICAL: Use deepcopy to prevent mutating the original history
    # This ensures RAG context is injected ONLY for this generation
    if rag_context:
        working_history = copy.deepcopy(chat_history)
        working_history[-1][
            'content'] = f"Use this reference:\n---REFERENCE---\n{rag_context}\n---END REFERENCE---\n\nQuestion: {working_history[-1]['content']}"
    else:
        working_history = chat_history

    messages = [{"role": "system", "content": CODE_SYSTEM_PROMPT}] + working_history
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

    stopping_criteria = StoppingCriteriaList([StopOnToken(tokenizer.convert_tokens_to_ids(CODE_STOP_TOKEN))])
    generation_config = {
        "streamer": streamer, "max_new_tokens": 8000, "temperature": 0.2,
        "do_sample": True, "stopping_criteria": stopping_criteria, "pad_token_id": tokenizer.eos_token_id
    }

    final_kwargs = inputs | generation_config
    outputs = model.generate(**final_kwargs)

    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    clean_response = response_text.replace(CODE_STOP_TOKEN, "").strip()
    heal_and_reprint_if_needed(clean_response)
    return clean_response


def handle_math_query(chat_history):
    """Handles math queries with expert-specific history."""
    model, tokenizer = load_expert_model("math")
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Build prompt for math expert
    prompt_text = f"<|system|>\n{MATH_SYSTEM_PROMPT}\n\n"
    for message in chat_history:
        prompt_text += f"<|{message['role']}|>\n{message['content']}\n\n"
    prompt_text += "<|assistant|>\n"

    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
    generation_config = {
        "streamer": streamer, "max_new_tokens": 1024, "temperature": 0.3,
        "top_p": 0.9, "do_sample": True, "repetition_penalty": 1.05, "pad_token_id": tokenizer.eos_token_id
    }
    final_kwargs = inputs | generation_config
    outputs = model.generate(**final_kwargs)
    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response_text.strip()


def handle_other_query(chat_history, rag_context=""):
    """Handles general queries with expert-specific history."""
    model, tokenizer = load_expert_model("other")
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # CRITICAL: Use deepcopy to prevent mutating the original history
    if rag_context:
        working_history = copy.deepcopy(chat_history)
        working_history[-1][
            'content'] = f"Use this reference:\n---REFERENCE---\n{rag_context}\n---END REFERENCE---\n\nQuestion: {working_history[-1]['content']}"
    else:
        working_history = chat_history

    messages = [{"role": "system", "content": OTHER_SYSTEM_PROMPT}] + working_history
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

    generation_config = {
        "streamer": streamer, "max_new_tokens": 1024, "temperature": 0.7,
        "do_sample": True, "pad_token_id": tokenizer.eos_token_id
    }
    final_kwargs = inputs | generation_config
    outputs = model.generate(**final_kwargs)
    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response_text.strip()


# ==============================================================================
#  STEP 4: MAIN EXECUTION LOOP (Expert-Specific History)
# ==============================================================================
if __name__ == "__main__":
    print("Initializing system...")
    try:
        model_cache["router"] = pipeline("text-classification", model=ROUTER_PATH, device=0)
        print("✅ Router loaded successfully.")
        model_cache["rag_code"] = KnowledgeRAG("Code", knowledge_base_path=CODE_KB_PATH)
        model_cache["rag_other"] = KnowledgeRAG("Other", knowledge_base_path=OTHER_KB_PATH)
    except Exception as e:
        print(f"❌ CRITICAL ERROR during initialization. Error: {e}")
        exit()

    # YOUR SUPERIOR ARCHITECTURE: Isolated memory for each expert
    expert_histories = {
        "code": [],
        "math": [],
        "other": []
    }

    print("\n\n" + "=" * 80)
    print("🚀 ULTIMATE Mixture-of-LoRAs + Multi-RAG Chatbot is running! 🚀")
    print("   You can override the router by ending your prompt with --code, --math, or --other")
    print("   Each expert now has isolated conversation memory - no more style contamination!")
    print("   Supports multi-line input. Press Enter on an empty line to submit.")
    print("=" * 80)

    try:
        while True:
            # --- MULTI-LINE INPUT FIX ---
            print("\n>> You (press Enter on an empty line to send):")
            lines = []
            while True:
                line = input()
                if not line: break
                lines.append(line)
            user_input_raw = "\n".join(lines).strip()
            # --- END FIX ---

            if user_input_raw.lower() in ["exit", "quit"]: break
            if not user_input_raw: continue

            user_input, expert_choice = user_input_raw, None
            override_flags = {"--code": "code", "--math": "math", "--other": "other"}

            for flag, expert in override_flags.items():
                if user_input.lower().strip().endswith(flag):
                    expert_choice, user_input = expert, user_input[:-len(flag)].strip()
                    print(f"\n[User Override] -> Routing to **{expert_choice.upper()}** expert.")
                    break

            if expert_choice is None:
                route_result_dict = model_cache["router"](user_input)[0]
                expert_choice = route_result_dict['label'].lower()
                print(f"\n[Router] -> Decided: **{expert_choice.upper()}** (Confidence: {route_result_dict['score']:.1%})")

            # Get the EXPERT-SPECIFIC history
            current_history = expert_histories[expert_choice]

            # Add user message to THIS expert's private history
            current_history.append({"role": "user", "content": user_input})
            print("-" * 50)

            rag_context_str = ""
            if expert_choice == "code":
                retrieved_docs = model_cache["rag_code"].retrieve(user_input, top_k=2)
                if retrieved_docs: rag_context_str = "\n\n".join(retrieved_docs)
            elif expert_choice == "other":
                retrieved_docs = model_cache["rag_other"].retrieve(user_input, top_k=3)
                if retrieved_docs: rag_context_str = "\n\n".join(retrieved_docs)

            start_time = time.time()
            print(f"🤖 {expert_choice.capitalize()} Expert:")

            # Pass the EXPERT-SPECIFIC history to the handler
            if expert_choice == "code":
                assistant_response = handle_code_query(current_history, rag_context=rag_context_str)
            elif expert_choice == "math":
                assistant_response = handle_math_query(current_history)
            else:
                assistant_response = handle_other_query(current_history, rag_context=rag_context_str)

            # Add assistant response to THIS expert's private history
            if assistant_response: # Only add if a response was generated
                current_history.append({"role": "assistant", "content": assistant_response})

            print(f"\n(Expert generation time: {time.time() - start_time:.2f} seconds)")

    except KeyboardInterrupt:
        print("\n\nExiting...")

    print("\nChat session finished.")