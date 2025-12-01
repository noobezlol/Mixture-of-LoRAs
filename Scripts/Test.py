import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# app.py
import os
import time
import copy
import json
import threading
import uvicorn
from pathlib import Path
from typing import Optional

# --- 1. IMPORTS & SETUP ---
try:
    import unsloth
    from unsloth import FastLanguageModel
except ImportError:
    pass

import torch
import faiss
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

# --- 2. CONFIGURATION (Adjust paths as needed) ---
ROUTER_PATH = os.path.join(BASE_DIR, "Classifier")
BASE_MODEL_PATH = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
CODE_LORA_PATH = os.path.join(BASE_DIR, "Final-Dynamic-Model/final_model(Code)")
MATH_LORA_PATH = os.path.join(BASE_DIR, "Final-Dynamic-Model/final_model(Math)")
CODE_KB_PATH = os.path.join(BASE_DIR, "knowledge_base/Code")
OTHER_KB_PATH = os.path.join(BASE_DIR, "knowledge_base/Base")

MAX_SEQ_LENGTH = 10000
CODE_STOP_TOKEN = "[END]"

CODE_SYSTEM_PROMPT = """You are an elite software engineer who writes syntactically perfect, logically sound code across all programming languages.

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
MATH_SYSTEM_PROMPT ="""You are a helpful assistant who thinks step by step through problems. When solving questions, show your reasoning process clearly using <think> tags, work through each step methodically, and then provide a clear final answer."""

OTHER_SYSTEM_PROMPT = """You are a helpful and friendly AI assistant."""

# Globals
model_cache = {"router": None, "code": None, "math": None, "other": None, "rag_code": None, "rag_other": None}
sessions = {}


# --- 3. BACKEND LOGIC ---

class KnowledgeRAG:
    def __init__(self, name, knowledge_base_path):
        self.name = name
        self.knowledge_base_path = Path(knowledge_base_path)
        if not self.knowledge_base_path.exists():
            self.knowledge_base_path.mkdir(parents=True, exist_ok=True)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2',
                                                   device='cuda' if torch.cuda.is_available() else 'cpu')
        self.documents = []
        self.index = None
        self._load_and_build()

    def _load_and_build(self):
        kb_files = sorted(self.knowledge_base_path.glob("*.txt"))
        for file_path in kb_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            sections = content.split('\n\n')
            for section in sections:
                if section.strip(): self.documents.append(section.strip())
        if not self.documents: return
        embeddings = self.embedding_model.encode(self.documents, show_progress_bar=False, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))

    def retrieve(self, query, top_k=2):
        if not self.index or not self.documents: return []
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True).astype('float32')
        _, indices = self.index.search(query_embedding, top_k)
        return [self.documents[idx] for idx in indices[0] if idx < len(self.documents)]


class StopOnToken(StoppingCriteria):
    def __init__(self, stop_token_id): self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs): return input_ids[0, -1] == self.stop_token_id


def load_expert_model(expert_name):
    if model_cache.get(expert_name): return model_cache[expert_name]
    print(f"[System] Loading '{expert_name}' expert...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL_PATH,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True
        )
        if expert_name != "other":
            lora_path = CODE_LORA_PATH if expert_name == "code" else MATH_LORA_PATH
            if expert_name == "code":
                tokenizer.add_special_tokens({"additional_special_tokens": [CODE_STOP_TOKEN]})
                model.resize_token_embeddings(len(tokenizer))
            model.load_adapter(lora_path)
        FastLanguageModel.for_inference(model)
        model_cache[expert_name] = (model, tokenizer)
        return model_cache[expert_name]
    except Exception as e:
        print(f"Error loading model (Running in mock mode?): {e}")
        return None, None


# --- 4. API SERVER ---

app = FastAPI(title="DeepSeek Clone")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    message: str
    session_id: str
    override: Optional[str] = None
    deep_think: bool = False


def get_session_history(session_id, expert_choice):
    if session_id not in sessions:
        sessions[session_id] = {"code": [], "math": [], "other": []}
    return sessions[session_id][expert_choice]


@app.on_event("startup")
def startup_event():
    print("Initializing Server Resources...")
    try:
        if os.path.exists(ROUTER_PATH):
            model_cache["router"] = pipeline("text-classification", model=ROUTER_PATH, device=0)
        if os.path.exists(str(CODE_KB_PATH)):
            model_cache["rag_code"] = KnowledgeRAG("Code", CODE_KB_PATH)
        if os.path.exists(str(OTHER_KB_PATH)):
            model_cache["rag_other"] = KnowledgeRAG("Other", OTHER_KB_PATH)
        print("✅ Core systems loaded.")
    except Exception as e:
        print(f"⚠️ Initialization Warning: {e}")


async def generate_stream(request: ChatRequest):
    # 1. Routing Logic
    expert_choice = "other"

    # If "DeepThink" button was clicked, force Math/Reasoning expert
    if request.deep_think:
        expert_choice = "math"
    elif request.override in ["code", "math", "other"]:
        expert_choice = request.override
    elif model_cache["router"]:
        try:
            route = model_cache["router"](request.message)[0]
            expert_choice = route['label'].lower()
        except:
            pass

    yield json.dumps({"type": "meta", "expert": expert_choice}) + "\n"

    # 2. Context & RAG
    history = get_session_history(request.session_id, expert_choice)
    history.append({"role": "user", "content": request.message})

    rag_text = ""
    if expert_choice == "code" and model_cache.get("rag_code"):
        docs = model_cache["rag_code"].retrieve(request.message)
        rag_text = "\n\n".join(docs)
    elif expert_choice == "other" and model_cache.get("rag_other"):
        docs = model_cache["rag_other"].retrieve(request.message)
        rag_text = "\n\n".join(docs)

    # 3. Inference
    model, tokenizer = load_expert_model(expert_choice)
    if not model:
        yield json.dumps({"type": "token", "content": "System Error: Model not loaded."}) + "\n"
        return

    working_history = copy.deepcopy(history)
    if rag_text:
        working_history[-1]['content'] = f"Reference:\n{rag_text}\n\nQuestion: {request.message}"

    prompt_str = ""
    sys_prompt = OTHER_SYSTEM_PROMPT
    if expert_choice == "code":
        sys_prompt = CODE_SYSTEM_PROMPT
    elif expert_choice == "math":
        sys_prompt = MATH_SYSTEM_PROMPT

    if expert_choice == "math":
        prompt_str = f"<|system|>\n{sys_prompt}\n\n"
        for msg in working_history:
            prompt_str += f"<|{msg['role']}|>\n{msg['content']}\n\n"
        prompt_str += "<|assistant|>\n"
    else:
        messages = [{"role": "system", "content": sys_prompt}] + working_history
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt_str, return_tensors="pt").to("cuda")

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=4000 if expert_choice == "code" else 1024,
        temperature=0.6,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    if expert_choice == "code":
        stop_list = StoppingCriteriaList([StopOnToken(tokenizer.convert_tokens_to_ids(CODE_STOP_TOKEN))])
        generation_kwargs["stopping_criteria"] = stop_list

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    full_response = ""
    for new_text in streamer:
        full_response += new_text
        clean_text = new_text.replace(CODE_STOP_TOKEN, "")
        if clean_text:
            yield json.dumps({"type": "token", "content": clean_text}) + "\n"
            await asyncio.sleep(0)

    history.append({"role": "assistant", "content": full_response.replace(CODE_STOP_TOKEN, "").strip()})
    yield json.dumps({"type": "done"}) + "\n"


@app.post("/chat_stream")
async def chat_endpoint(request: ChatRequest):
    return StreamingResponse(generate_stream(request), media_type="application/x-ndjson")


# --- 5. THE DEEPSEEK FRONTEND ---
@app.get("/", response_class=HTMLResponse)
async def get_ui():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepSeek Clone</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">

    <!-- Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

    <style>
        body { font-family: 'Inter', sans-serif; background-color: #ffffff; color: #1f2937; }

        /* Custom Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #e5e7eb; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #d1d5db; }

        /* Typography */
        .prose p { margin-bottom: 1em; line-height: 1.6; }
        .prose pre { background: #f3f4f6; border-radius: 8px; padding: 1em; overflow-x: auto; border: 1px solid #e5e7eb; }
        .prose code { background: #f3f4f6; padding: 2px 4px; border-radius: 4px; font-size: 0.9em; font-family: 'Menlo', monospace; color: #ef4444; }
        .prose pre code { background: transparent; color: inherit; padding: 0; }

        /* Thinking Block (Reasoning) */
        details.thinking { margin: 1em 0; border-left: 3px solid #e5e7eb; padding-left: 1em; }
        details.thinking summary { cursor: pointer; color: #6b7280; font-size: 0.9em; font-weight: 500; list-style: none; display: flex; align-items: center; gap: 0.5rem; }
        details.thinking summary:hover { color: #374151; }
        details.thinking summary::before { content: 'Expected'; font-family: "Font Awesome 6 Free"; font-weight: 900; content: "\f0eb"; } 
        details.thinking[open] summary::before { content: "\f0eb"; color: #4f46e5; }
        details.thinking div.content { margin-top: 0.5em; color: #6b7280; font-size: 0.9em; font-style: italic; }

        /* Layout Transitions */
        .input-centered { 
            position: absolute; 
            top: 50%; 
            left: 50%; 
            transform: translate(-50%, -50%); 
            width: 100%; 
            max-width: 48rem;
            transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .input-bottom { 
            position: fixed; 
            bottom: 0; 
            left: 16rem; /* Sidebar width */
            right: 0;
            width: auto;
            padding: 1.5rem;
            background: white;
            transform: translate(0, 0);
            max-width: 1000px;
            margin: 0 auto;
        }

        @media (max-width: 768px) {
            .input-bottom { left: 0; }
        }
    </style>
</head>
<body class="flex h-screen overflow-hidden">

    <!-- Sidebar -->
    <aside class="w-64 bg-[#f9fbfa] border-r border-gray-200 flex-none hidden md:flex flex-col justify-between z-20">
        <div class="p-4">
            <div class="flex items-center gap-2 mb-6 px-2">
                <div class="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center text-white font-bold text-sm">DS</div>
                <span class="font-semibold text-gray-700 tracking-tight">DeepSeek Clone</span>
            </div>
            <button onclick="location.reload()" class="w-full flex items-center gap-3 px-4 py-2.5 bg-white border border-gray-200 rounded-lg shadow-sm hover:bg-gray-50 transition text-sm font-medium text-gray-700 mb-6">
                <i class="fa-solid fa-plus text-gray-400"></i> New Chat
            </button>

            <div class="space-y-1">
                <div class="px-3 py-2 text-xs font-medium text-gray-400 uppercase">Recent</div>
                <!-- Mock History Items -->
                <button class="w-full text-left px-3 py-2 text-sm text-gray-600 rounded-md hover:bg-gray-100 truncate">
                    Code Analysis Python...
                </button>
                <button class="w-full text-left px-3 py-2 text-sm text-gray-600 rounded-md hover:bg-gray-100 truncate">
                    Calculus Problem Set 1
                </button>
            </div>
        </div>

        <div class="p-4 border-t border-gray-200">
            <div class="flex items-center gap-3 px-2 py-2 rounded-lg hover:bg-gray-100 cursor-pointer transition">
                <div class="w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center text-gray-500 text-xs"><i class="fa-solid fa-user"></i></div>
                <div class="text-sm font-medium text-gray-700">User</div>
            </div>
        </div>
    </aside>

    <!-- Main Content -->
    <main class="flex-1 flex flex-col relative bg-white">

        <!-- Mobile Header -->
        <div class="md:hidden p-4 flex justify-between items-center border-b border-gray-200">
            <span class="font-bold text-gray-700">DeepSeek</span>
            <button onclick="location.reload()" class="p-2"><i class="fa-solid fa-plus"></i></button>
        </div>

        <!-- Chat Scroll Area -->
        <div id="chat-container" class="flex-1 overflow-y-auto p-4 pb-32 scroll-smooth">
            <!-- Welcome Message (Hidden when chat starts) -->
            <div id="welcome-screen" class="h-full flex flex-col items-center justify-center -mt-10 transition-opacity duration-300">
                <div class="w-16 h-16 bg-white rounded-2xl shadow-sm border border-gray-100 flex items-center justify-center mb-6">
                     <div class="text-4xl text-blue-600"><i class="fa-solid fa-dragon"></i></div>
                </div>
                <h1 class="text-2xl font-semibold text-gray-800 mb-2">How can I help you?</h1>
            </div>

            <!-- Messages will be injected here -->
        </div>

        <!-- Input Area -->
        <div id="input-wrapper" class="input-centered px-4">
            <div class="relative bg-white rounded-3xl shadow-[0_0_15px_rgba(0,0,0,0.05)] border border-gray-200 transition-all duration-200 focus-within:shadow-lg focus-within:border-gray-300">
                <textarea id="user-input" rows="1" class="w-full bg-transparent border-0 text-gray-800 placeholder-gray-400 px-5 py-4 focus:ring-0 resize-none max-h-48 text-base" placeholder="Message DeepSeek..."></textarea>

                <!-- Toolbar -->
                <div class="flex justify-between items-center px-4 pb-3 pt-1">
                    <div class="flex gap-2">
                        <button id="deep-think-btn" class="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium bg-gray-100 text-gray-600 hover:bg-gray-200 transition border border-transparent">
                            <i class="fa-solid fa-brain"></i> DeepThink (R1)
                        </button>
                        <button class="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium bg-gray-100 text-gray-600 hover:bg-gray-200 transition">
                            <i class="fa-solid fa-globe"></i> Search
                        </button>
                    </div>

                    <button id="send-btn" onclick="sendMessage()" class="w-8 h-8 rounded-full bg-gray-200 text-gray-400 flex items-center justify-center transition hover:bg-blue-600 hover:text-white disabled:opacity-50">
                        <i class="fa-solid fa-arrow-up"></i>
                    </button>
                </div>
            </div>
            <div class="text-center mt-3">
                <p class="text-xs text-gray-400">AI generated content can be inaccurate.</p>
            </div>
        </div>

    </main>

    <script>
        const sessionId = "sess_" + Math.random().toString(36).substr(2, 9);
        const chatContainer = document.getElementById('chat-container');
        const welcomeScreen = document.getElementById('welcome-screen');
        const inputWrapper = document.getElementById('input-wrapper');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        const deepThinkBtn = document.getElementById('deep-think-btn');

        let hasStarted = false;
        let isDeepThinkActive = false;
        let isGenerating = false;

        // DeepThink Toggle
        deepThinkBtn.addEventListener('click', () => {
            isDeepThinkActive = !isDeepThinkActive;
            if(isDeepThinkActive) {
                deepThinkBtn.classList.replace('bg-gray-100', 'bg-blue-50');
                deepThinkBtn.classList.replace('text-gray-600', 'text-blue-600');
                deepThinkBtn.classList.replace('border-transparent', 'border-blue-200');
            } else {
                deepThinkBtn.classList.replace('bg-blue-50', 'bg-gray-100');
                deepThinkBtn.classList.replace('text-blue-600', 'text-gray-600');
                deepThinkBtn.classList.replace('border-blue-200', 'border-transparent');
            }
        });

        // Auto Resize
        userInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
            if (this.value.trim().length > 0) {
                sendBtn.classList.replace('bg-gray-200', 'bg-blue-600');
                sendBtn.classList.replace('text-gray-400', 'text-white');
            } else {
                sendBtn.classList.replace('bg-blue-600', 'bg-gray-200');
                sendBtn.classList.replace('text-white', 'text-gray-400');
            }
        });

        // Enter Key
        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        async function sendMessage() {
            const text = userInput.value.trim();
            if (!text || isGenerating) return;

            // Transition UI
            if (!hasStarted) {
                hasStarted = true;
                welcomeScreen.style.display = 'none';
                inputWrapper.classList.remove('input-centered');
                inputWrapper.classList.add('input-bottom');
            }

            // Reset Input
            userInput.value = '';
            userInput.style.height = 'auto';
            isGenerating = true;
            sendBtn.disabled = true;

            // User Message
            appendMessage('user', text);

            // AI Placeholder
            const aiContentDiv = appendMessage('ai', '');

            try {
                const response = await fetch('/chat_stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        message: text, 
                        session_id: sessionId, 
                        deep_think: isDeepThinkActive 
                    })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let fullText = "";

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    const lines = decoder.decode(value, { stream: true }).split('\\n');
                    for (const line of lines) {
                        if (!line.trim()) continue;
                        try {
                            const data = JSON.parse(line);
                            if (data.type === 'token') {
                                fullText += data.content;
                                renderContent(aiContentDiv, fullText);
                                chatContainer.scrollTop = chatContainer.scrollHeight;
                            }
                        } catch (e) {}
                    }
                }
            } catch (e) {
                aiContentDiv.innerHTML = `<span class="text-red-500">Error: ${e.message}</span>`;
            }

            isGenerating = false;
            sendBtn.disabled = false;
            hljs.highlightAll();
        }

        function appendMessage(role, text) {
            const row = document.createElement('div');
            row.className = 'w-full max-w-3xl mx-auto mb-6 flex gap-4';

            if (role === 'user') {
                row.innerHTML = `
                    <div class="flex-1 flex justify-end">
                        <div class="bg-gray-100 text-gray-800 px-4 py-2.5 rounded-2xl rounded-tr-sm max-w-[85%] leading-relaxed whitespace-pre-wrap">${text}</div>
                    </div>
                `;
            } else {
                const contentId = 'ai-' + Math.random().toString(36).substr(2, 9);
                row.innerHTML = `
                    <div class="w-8 h-8 rounded-full bg-blue-600 flex-shrink-0 flex items-center justify-center text-white text-xs font-bold shadow-sm">DS</div>
                    <div class="flex-1 min-w-0">
                        <div class="font-medium text-sm text-gray-900 mb-1">DeepSeek Clone</div>
                        <div id="${contentId}" class="prose prose-sm max-w-none text-gray-800">Thinking...</div>
                    </div>
                `;
            }
            chatContainer.appendChild(row);
            return role === 'ai' ? document.getElementById(row.querySelector('.prose').id) : null;
        }

        function renderContent(element, text) {
            let html = "";
            // Logic for <thinking> tags
            const thinkRegex = /<thinking>([\\s\\S]*?)<\\/thinking>/g;
            const thinkMatch = thinkRegex.exec(text);
            let cleanText = text;

            if (thinkMatch) {
                html += `<details class="thinking" open>
                            <summary>Reasoning Process</summary>
                            <div class="content">${thinkMatch[1]}</div>
                         </details>`;
                cleanText = text.replace(thinkRegex, "");
            } else if (text.includes("<thinking>")) {
                 const parts = text.split("<thinking>");
                 html += parts[0];
                 html += `<details class="thinking" open>
                            <summary>Reasoning Process...</summary>
                            <div class="content">${parts[1]}</div>
                         </details>`;
                 element.innerHTML = html;
                 return;
            }

            cleanText = cleanText.replace(/<answer>/g, "").replace(/<\\/answer>/g, "");
            html += marked.parse(cleanText);
            element.innerHTML = html;
        }
    </script>
</body>
</html>
    """


if __name__ == "__main__":
    print("🚀 DeepSeek UI running at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)