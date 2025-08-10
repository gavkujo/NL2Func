# Helper to fuse system instructions and functional guidelines
def fused_system_message(system, func_guidelines=None):
    """
    Returns a single string with system instructions, background/context, and functional guidelines (if any), separated by clear headers.
    """
    parts = []
    if system and len(system) > 0:
        parts.append(system[0]["content"].strip())
    if system and len(system) > 1:
        parts.append(system[1]["content"].strip())
    if func_guidelines:
        parts.append(func_guidelines.strip())
    return '\n\n'.join(parts)
# Function-specific guidelines/context mapping
FUNCTION_GUIDELINES = {
    "Asaoka_data": '''
=== FUNCTIONAL INSTRUCTIONS ===
You will receive a JSON list of settlement plate data relevant to the user query. Based on this data and the following instructions, answer the user’s query precisely.

Key Concepts:
- Settlement plates measure ground settlement in meters (m). More negative values indicate greater settlement from a baseline elevation.
- Settlement results from soil consolidation under surcharge load (a sand layer that compresses the ground).
- Settlement plates follow a naming format like 'F3-R03a-SM-01': 
  - 'R03a' identifies the region,
  - 'SM' denotes Settlement Plate,
  - '01' is the plate index.
  Do not comment on or interpret these IDs—they are fixed references.
- Settlement varies between plates due to unique soil layering beneath each plate, even under identical ground conditions.
- '7day_rate' indicates settlement amount over the past 7 days (in mm).
- 'Latest_GL' is the latest ground elevation (in mCD); higher values indicate heavier surcharge loading and should correlate with more settlement.
- 'Surcharge_Complete_Date' marks when the major surcharge loading finished.
- 'Holding_period' is the number of days between 'Surcharge_Complete_Date' and 'Latest_Date'. Longer holding periods typically mean settlement is tapering.
- 'Asaoka_DOC' is the Degree of Consolidation (%) from the Asaoka Assessment method:
  - 100% = full consolidation (no further settlement expected),
  - 90–100% = settlement tapering,
  - below 90% = ongoing settlement and non-compliance.
- Compliance criteria:
  - 'Latest_GL' ≥ 16.9 mCD,
  - '7day_rate' ≤ 4 mm.

Output Requirements:
- When asked for summaries or overviews, provide a table listing each Settlement Plate’s ID along with raw values for:
  - latest_Settlement,
  - Latest_GL,
  - Latest_Date,
  - Asaoka_DOC,
  - Holding_period,
  - 7day_rate.
- Do not interpret or comment on the format of the Settlement Plate ID.

Always tailor your response strictly to these parameters.
''',
    "reporter_Asaoka": '''
=== FUNCTIONAL INSTRUCTIONS ===
- You have been tasked with creating a PDF report for the user-specified list of plates.
- The PDF report is already generated and ready.
- Your only responsibility is to inform the user that the PDF report is now available for download.
- No further explanation or commentary is needed.

''',
    "plot_combi_S": '''
=== FUNCTIONAL INSTRUCTIONS ===
- You have been tasked with creating a settlement graph for the user-specified list of plates.
- The graph is already generated and ready.
- Your only responsibility is to inform the user that the graph is now available for download.
- No further explanation or commentary is needed.
''',
    "SM_overview":'''
=== FUNCTIONAL INSTRUCTIONS ===
You are provided with a JSON list of settlement plates and must answer user queries using the following guidelines:

- Settlement plates measure ground settlement in meters (m), with larger negative values indicating more settlement from baseline elevation due to soil consolidation under surcharge loading.
- Naming format is 'F3-R03a-SM-01' (region, plate type, index). Do not comment on these IDs.
- Settlement varies plate-to-plate due to differing soil layers.
- '7day_rate' is settlement in millimeters over the past 7 days.
- 'Latest_GL' is ground elevation (mCD); higher values indicate heavier surcharge and more settlement.
- 'Surcharge_Complete_Date' marks when surcharge loading was completed.
- 'Holding_period' is days between surcharge completion and latest measurement; longer periods imply more ground treatment and typically less future settlement.
- 'Asaoka_DOC' (Degree of Consolidation %) indicates:
  - 100% means full consolidation,
  - 90–100% indicates tapering settlement,
  - less than 90% indicates ongoing settlement and non-compliance.
- Compliance criteria:
  - 'Latest_GL' ≥ 16.9 mCD,
  - '7day_rate' ≤ 4 mm.
- There is an inverse relationship between '7day_rate' and 'Holding_period': larger holding periods usually mean smaller settlement rates.

Output Instructions:
- When asked for a summary or overview, provide a table listing each Settlement Plate’s ID along with raw values for:
  - latest_Settlement,
  - Latest_GL,
  - Latest_Date,
  - Asaoka_DOC,
  - Holding_period,
  - 7day_rate.
- Do not comment on or interpret the plate IDs.
- ALWAYS provide a descriptive analysis of the data at the end of your response, even if not explicitly requested.
'''
}
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
    "text": "yi:6b-chat" #gemma:2b
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
def build_messages(system, memory, user_input, classifier_data=None, func_guidelines=None, func_output=None):
    messages = []
    # Fuse system instructions and functional guidelines
    fused_content = fused_system_message(system, func_guidelines)
    # Fuse output and convo history if present
    block = ""
    if classifier_data or func_output:
        if classifier_data:
            block += f"Function: {classifier_data.get('Function')}\nParams: {classifier_data.get('Params')}\n"
        if func_output:
            block += f"Output: {func_output}\n"
    if "@recap" in user_input and memory:
        recap = "=== PREVIOUS CONVERSATION ===\n"
        for i, (user_msg, assistant_msg) in enumerate(memory, 1):
            recap += f"User ({i}): {user_msg.strip()}\n"
            recap += f"Assistant ({i}): {assistant_msg.strip()}\n"
        block += recap
    # Add the fused system message
    if block:
        fused_content += '\n\n' + block.strip()
    messages.append({"role": "system", "content": fused_content})
    # Always add the user query last, but REMOVE @recap from the message
    cleaned_user_input = user_input.replace("@recap", "").strip()
    messages.append({"role": "user", "content": "\n === USER QUERY ===\n" + cleaned_user_input})
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
                '''
                === SYSTEM INSTRUCTIONS ===
                * Always answer the user's query strictly based on the information provided in the current context.
                * Background information is provided only for light reference and should never override the functional instructions or the user query.
                * Follow this priority order when referencing information:
                    1. USER QUERY (highest priority)
                    2. PREVIOUS CONVERSATION HISTORY (if any)
                    3. FUNCTIONAL INSTRUCTIONS (guidelines specific to each task) 
                    4. USER DATA (to be interpreted based on the functional instructions)
                    5. BACKGROUND INFORMATION (lowest priority)

                * Do NOT hallucinate or fabricate any information. Only ask clarifying questions if a term or instruction is unclear.
                * Always structure your responses clearly and logically.
                * Maintain a professional, concise, and factual tone unless otherwise specified by the user.
                * The project background is:
                    - Tuas Terminal Phase 2 Project, Singapore: large-scale land reclamation and wharf construction.
                    - 365 hectares of reclaimed land, 9km wharf line.
                    - 1900 settlement plates installed every 1600 sqm to monitor settlement under sand surcharge.
                    - Settlement plates measure ground level changes critical for soil improvement verification.
                    - Compliance criteria include Asaoka DOC > 90%, ground level > 16.9mCD, and settlement rate ≤ 4mm.

                * When the user refers to previous conversation points, respond accordingly using the conversation history.
'''
            )},
            {"role": "system", "content": (
            '''
            === FUNCTIONAL INSTRUCTIONS ===
            You are a helpful chatbot tasked to continue the conversation based on the User query and conversation history ONLY.
            * Main goal is to continue the conversation logically. 
            * Refer only to conversation history and the above functional instructions.
            * If user references previous conversation points, respond accordingly.
            ''')}
        ]

        # Warm up default model
        #warmup_model(MODEL_CONFIG["text"])
        self.warmed_up_models.add(MODEL_CONFIG["text"])


    def handle_user(self, user_input: str, func_name=None, classifier_data=None, func_output=None, stream=False):
        cleaned_input = strip_think(user_input)
        model = select_model(cleaned_input)

        if model not in self.warmed_up_models:
            self.warmed_up_models.add(model)

        func_guidelines = FUNCTION_GUIDELINES.get(func_name)
        messages = build_messages(
            self.system_messages,
            self.memory,
            cleaned_input,
            classifier_data=classifier_data,
            func_guidelines=func_guidelines,
            func_output=func_output
        )
        print("FINAL MESSAGE \n", messages)

        if stream:
            # Streaming mode: yield tokens as they arrive
            response = ""
            for token in ollama_stream(messages, model):
                yield token
                response += token
            # After streaming, update memory
            #summary = get_summary(response)
            summary = response
            cleaned_input = cleaned_input.replace("@recap", "").strip()
            self.memory.append((cleaned_input, summary))
            self.memory = self.memory[-self.max_turns:]
        else:
            # Non-streaming mode: collect all tokens, then return
            response = ""
            for token in ollama_stream(messages, model):
                response += token
            summary = get_summary(response)
            self.memory.append((cleaned_input, summary))
            self.memory = self.memory[-self.max_turns:]
            return response
    


# ======= Run ========
if __name__ == "__main__":
    router = LLMRouter(max_turns=3)

    combined = ''' prompt '''
    router.handle_user(combined)

    while True:
        user_input = input("\n>>: ")
        if user_input.strip().lower() == 'exit':
            break
        router.handle_user(user_input)
