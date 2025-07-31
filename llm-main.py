#!/usr/bin/env python3
import requests
import json
import time
import re

from transformers import T5Tokenizer, T5ForConditionalGeneration


# Save locally for conversion
try:
    tokenizer = T5Tokenizer.from_pretrained("./t5-small")
    model = T5ForConditionalGeneration.from_pretrained("./t5-small")
except:
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model.save_pretrained("./t5-small")
    tokenizer.save_pretrained("./t5-small")


# Configuration
MODEL_CONFIG = {
    "think": "deepseek-r1:1.5b",
    "deep": "deepseek-r1:7b-qwen-distill-q4_K_M",
    "text": "gemma:2b" 
}

# Utility: Remove <think> blocks
def strip_think(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def get_summary(text):
    print(f"[Debug][Summarizer] Input length: {len(text)} chars")
    input_text = "lightly summarize the following text: " + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=2048, truncation=True)
    outputs = model.generate(**inputs, max_length=1024, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"[Debug][Summarizer] Summary length: {len(summary)} chars")
    print(f"[Debug][Summarizer] Summary preview: {summary[:100]}...")
    return summary

# Build full message list
def build_messages(system, memory, user_input):
    messages = system.copy()
    if "@recap" in user_input:
        # Append memory as a system-style recap block
        if memory:
            recap = "=== PREVIOUS CONVERSATION ===\n"
            for i, (user_msg, assistant_msg) in enumerate(memory, 1):
                recap += f"User ({i}): {user_msg.strip()}\n"
                recap += f"Assistant ({i}): {assistant_msg.strip()}\n"
            messages.append({"role": "system", "content": recap.strip()})

    # Now add the current user input
    messages.append({"role": "user", "content": "\n === USER QUERY ===\n" + user_input})
    return messages

# One-time warm-up for model
def warmup_model(model: str):
    print(f"[Debug] Warming up model: {model}")
    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True
        }
        r = requests.post("http://localhost:11434/v1/chat/completions",
                          headers={"Content-Type": "application/json"},
                          json=payload, stream=True)
        for line in r.iter_lines(decode_unicode=True):
            if line and "[DONE]" in line:
                break
        r.close()
        print(f"[Debug] Warm-up complete for: {model}")
    except Exception as e:
        print(f"[Error] Warm-up failed: {e}")

# Streaming helper
def ollama_stream(messages: list, model: str):
    url = "http://localhost:11434/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer ollama"
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": True
    }

    with requests.post(url, headers=headers, json=payload, stream=True) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data:"):
                continue
            data = raw_line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    yield delta["content"]
            except Exception as e:
                print(f"[Warning] Stream decode failed: {e}")
                continue

# Model selector
def select_model(prompt: str) -> str:
    if "@think" in prompt:
        return MODEL_CONFIG["think"]
    elif "@deep" in prompt:
        return MODEL_CONFIG["deep"]
    return MODEL_CONFIG["text"]

# ===== Main LLM Router =====
class LLMRouter:
    def __init__(self, max_turns=5):
        self.memory = []  # [(user_msg, assistant_msg)]
        self.max_turns = max_turns
        self.warmed_up_models = set()

        # System context
        self.system_messages = [
            {"role": "system", "content": (
                '''=== INSTRUCTIONS ===
                * Answer user prompt strictly based on the context and the information given by the user. 
                * NOTE: Your purpose is to answer questions based on the given prompt and/or context ONLY. Do not make anything up as all information is provided in the context or/and the user prompt. ONLY ask questions if really necessary (i.e. if some terminology is unclear).'''
            )},
            {"role": "system", "content": (
                '''=== CONTEXT ===:
                Consider the following as an overview of the Tuas Terminal Phase 2 Project in Singapore and use of Settlement Plates:
                * A large land-reclamation project in Singapore which comprises the construction of wharf line for the future Tuas megaport.
                * 365 hectares of land reclamation, constructing a 9km wharf line.
                * Settlement Plates are instruments installed every 1600 square metres to measure the change in ground level under sand surcharge. 
                * Sand surcharge is used to weigh the reclaimed land, comprising mostly clay and sand, downwards which results in settlement and improvement of the reclaimed soil properties.
                * Currently, 1900 Settlement Plates are installed at the project
                * Settlement plates are named in a format similar to 'F3-R03a-SM-04' where 'F3' Indicates it belongs to the project, R03a is a specific area within the project, 'SM' indicates it is a Settlement Plates and the last two digits is the plate's index number. Do not comment on these names as they are fixed and require no interpretation.
                * The Settlement Plates are crucial for completion of Soil Improvement Works, where the settlement measurements have to meet certain criteria prior to being approved for removal.
                * The criteria is Asaoka DOC greater than 90%, a ground level above 16.9mCD and rate of settlement less than 4mm.

                === GUIDELINES ===
                
                When assessing the overview for Settlement Plates, note the following:
                * A settlement plate measures the ground settlement in millimetres (mm) where a larger negative value means more settlement from a baseline elevation. This settlement is a result of consolidation, where the soil improves under a surcharge load.
                * The settlement plate has a standard naming format like 'F3-R03a-SM-01', where 'R03a' denotes a region within the project, 'SM' means Settlement plate and '01 denotes which Settlement Plate is being referred to. The last two digits are an index number. Do not comment on these names as they are fixed.
                * Settlement is expected to vary from Settlement Plate to Settlement Plate as the soil layering under each Settlement Plate is unique, and may behave differently even under the same ground level.
                * Surcharge load is a certain thickness of sand which weighs the ground down, thereby causing settlement and improvement of the underlying soil's properties.
                * The "Latest_GL" is the last reported ground elevation in units 'mCD'. A larger number indicates the particular plate is loaded more, and hence should record more settlement.
                * Each Settlement Plate is surcharged on a particular date known as the "Surcharge_Complete_Date" which indicates the date from when the major of the settlement occurs.
                * The Holding_period is the period of time in days, between the "Surcharge_Complete_Date" and "Latest_Date" when the settlement was last reported. A longer 'Holding_Period' usually means that the settlement has had time to taper off. Shorter periods may result in more ongoing settlement.
                * The "Asaoka_DOC" denotes the Degree of Consolidation (DOC) based on the Asaoka Assessment method. It is a measure in units %, of by how much the ground has consolidated. A DOC of 100 % means no more settlement is expected, while a DOC between 90 % and 100 % means the settlement is tapering and a DOC less than 90% indicate on-going settlement. DOC less than 90 % non-compliant to the requirements.
                * The 'Latest_GL' should be a minimum 16.9mCD to be compliant with Port specifications.
                * When asked for a summary or overview, make sure to provide the Settlement Plate ID, along with the respective "latest_Settlement", "Latest_GL", "Latest_Date", "Asaoka_DOC", "Holding_period" which is what users are interested in. You do not need to comment on the format of the Settlement Plate ID as this is just a reference identifier.
                * When asked for a summary or overview of the Settlement Plate data, provide a table at the end of your response.'''
            )}
        ]

        # Warm up default model
        #warmup_model(MODEL_CONFIG["text"])
        self.warmed_up_models.add(MODEL_CONFIG["text"])

    def handle_user(self, user_input: str):
        cleaned_input = strip_think(user_input)
        model = select_model(cleaned_input)

        # Warm up if not already
        if model not in self.warmed_up_models:
            #warmup_model(model)
            self.warmed_up_models.add(model)

        messages = build_messages(self.system_messages, self.memory, cleaned_input)
        print(f"[Debug] Selected model: {model}")
        print(f"[Debug] Message count as a json: {len(messages)}")
        print(f"[Debug] Full message length as a string: {len(str(messages))}")
        print(f"[Debug] Full message: \n\n {messages} \n\n")

        start = time.time()
        response = ""
        first = True
        for token in ollama_stream(messages, model):
            if first:
                print("\r", end="", flush=True)
                ends = time.time()
                first = False
            print(token, end='', flush=True)
            response += token
        duration = time.time() - start
        firsttok = ends-start
        print(f"\n[Debug] Inference time: {duration:.2f}s with model: {model}")
        print(f"\n[Debug] First Token time: {firsttok:.2f}s with model: {model}")

        # Update memory
        summary = get_summary(response)  # lightweight summarizer
        self.memory.append((cleaned_input, summary))
        self.memory = self.memory[-self.max_turns:]
        return response

# ======= Run ========
if __name__ == "__main__":
    router = LLMRouter(max_turns=3)

    combined = '''
@analysis Please give me an descriptive full analysis of all the given settlement plates, finish by a tabular summary of all plates. Dont think too much, all guidelines are provided:

  "F3-R11a-SM-01": {
    "Surcharge_Complete_date": "2025-04-24",
    "Datetime": "2025-07-22",
    "latest_Settlement": "-2.617",
    "Latest_GL": "20.77",
    "Asaoka_DOC": "100",
    "Holding_period": "89"
  },
  "F3-R11a-SM-02": {
    "Surcharge_Complete_date": "2025-04-26",
    "Datetime": "2025-07-22",
    "latest_Settlement": "-2.136",
    "Latest_GL": "23.22",
    "Asaoka_DOC": "100",
    "Holding_period": "87"
  },
  "F3-R11a-SM-03": {
    "Surcharge_Complete_date": "2025-04-24",
    "Datetime": "2025-07-22",
    "latest_Settlement": "-2.133",
    "Latest_GL": "23.32",
    "Asaoka_DOC": "100",
    "Holding_period": "89"
  },
  "F3-R11a-SM-04": {
    "Surcharge_Complete_date": "2025-04-24",
    "Datetime": "2025-07-22",
    "latest_Settlement": "-2.658",
    "Latest_GL": "23.29",
    "Asaoka_DOC": "100",
    "Holding_period": "89"
  },
  "F3-R11a-SM-05": {
    "Surcharge_Complete_date": "2025-04-24",
    "Datetime": "2025-07-22",
    "latest_Settlement": "-2.225",
    "Latest_GL": "23.31",
    "Asaoka_DOC": "100",
    "Holding_period": "89"
  },
  "F3-R11a-SM-06": {
    "Surcharge_Complete_date": "2025-04-24",
    "Datetime": "2025-07-22",
    "latest_Settlement": "-2.303",
    "Latest_GL": "23.22",
    "Asaoka_DOC": "100",
    "Holding_period": "89"
  },
  "F3-R11a-SM-07": {
    "Surcharge_Complete_date": "2025-04-26",
    "Datetime": "2025-07-22",
    "latest_Settlement": "-2.137",
    "Latest_GL": "23.27",
    "Asaoka_DOC": "100",
    "Holding_period": "87"
  },
  "F3-R11a-SM-08": {
    "Surcharge_Complete_date": "2025-04-26",
    "Datetime": "2025-07-22",
    "latest_Settlement": "-2.518",
    "Latest_GL": "23.14",
    "Asaoka_DOC": "100",
    "Holding_period": "87"
  }
'''
    router.handle_user(combined)

    while True:
        user_input = input("\n>>: ")
        if user_input.strip().lower() == 'exit':
            break
        router.handle_user(user_input)
