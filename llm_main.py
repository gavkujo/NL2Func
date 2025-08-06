# Function-specific guidelines/context mapping
FUNCTION_GUIDELINES = {
    "Asaoka_data": '''
=== FUNCTIONAL INSTRUCTIONS ===:
You are given an output json data list of relevant settlement plates. Based on the user query, data, and the following guidelines, answer the user's query. When assessing the overview for Settlement Plates, note the following:
* A settlement plate measures the ground settlement in metres (m) where a larger negative value means more settlement from a baseline elevation. This settlement is a result of consolidation, where the soil improves under a surcharge load.
* The settlement plate has a standard naming format like 'F3-R03a-SM-01', where 'R03a' denotes a region within the project, 'SM' means Settlement plate and '01 denotes which Settlement Plate is being referred to. The last two digits are an index number. Do not comment on these names as they are fixed.
* Settlement is expected to vary from Settlement Plate to Settlement Plate as the soil layering under each Settlement Plate is unique, and may behave differently even under the same ground level.
* Surcharge load is a certain thickness of sand which weighs the ground down, thereby causing settlement and improvement of the underlying soil's properties.
* The '7day_rate' is the amount of settlement that has occurred over the last 7 days. 
* The "Latest_GL" is the last reported ground elevation in units 'mCD'. A larger number indicates the particular plate is loaded more, and hence should record more settlement.
* Each Settlement Plate is surcharged on a particular date known as the "Surcharge_Complete_Date" which indicates the date from when the major of the settlement occurs.
* The Holding_period is the period of time in days, between the "Surcharge_Complete_Date" and "Latest_Date" when the settlement was last reported. A longer 'Holding_Period' usually means that the settlement has had time to taper off. Shorter periods may result in more ongoing settlement.
* The "Asaoka_DOC" denotes the Degree of Consolidation (DOC) based on the Asaoka Assessment method. It is a measure in units %, of by how much the ground has consolidated. A DOC of 100 % means no more settlement is expected, while a DOC between 90 % and 100 % means the settlement is tapering and a DOC less than 90% indicate on-going settlement. DOC less than 90 % non-compliant to the requirements.
* The 'Latest_GL' should be a minimum 16.9mCD to be compliant with Port specifications.
* The '7day_rate' has to be less than or equal to 4 to be compliant. If this value is greater, indicate that the plate is non compliant.
* When asked for a summary or overview, make sure to provide the Settlement Plate ID, along with the respective "latest_Settlement", "Latest_GL", "Latest_Date", "Asaoka_DOC", "Holding_period" and "7day_rate" which is what users are interested in. Make sure to show the raw values and not interpreted values of these key parameters.
* You do not need to comment on the format of the Settlement Plate ID as this is just a reference identifier. 
* When asked for a summary or overview of the Settlement Plate data, provide a table at the end of your response.
''',
    "reporter_Asaoka": '''
=== FUNCTIONAL INSTRUCTIONS ===:
When preparing the Asaoka report for a set of Settlement Plates, note the following. The report will be prepared automatically:
* A settlement plate measures the ground settlement in metres (m) where a larger negative value means more settlement from a baseline elevation. This settlement is a result of consolidation, where the soil improves under a surcharge load.
* The settlement plate has a standard naming format like 'F3-R03a-SM-01', where 'R03a' denotes a region within the project, 'SM' means Settlement plate and '01 denotes which Settlement Plate is being referred to. The last two digits are an index number. Do not comment on these names as they are fixed.
* The Surcharge Completion Date is also called the SCD
* The Assessment Start Date is also called the ASD
* The ASD has to be after the SCD
''',
    "plot_combi_S": '''
=== FUNCTIONAL INSTRUCTIONS ===:
* You were tasked to create a settlement graph for a given list of plates by the user. The plot graph has been made for the user so your task is to tell the user that the graph is downloadable for the user now.
* Thats all
''',
    "SM_overview":'''
=== FUNCTIONAL INSTRUCTIONS ===:
You are given an output json data list of relevant settlement plates. Based on the user query, data, and the following guidelines, answer the user's query. When assessing the overview for Settlement Plates, note the following:
* A settlement plate measures the ground settlement in metres (m) where a larger negative value means more settlement from a baseline elevation. This settlement is a result of consolidation, where the soil improves under a surcharge load.
* The settlement plate has a standard naming format like 'F3-R03a-SM-01', where 'R03a' denotes a region within the project, 'SM' means Settlement plate and '01 denotes which Settlement Plate is being referred to. The last two digits are an index number. Do not comment on these names as they are fixed.
* Settlement is expected to vary from Settlement Plate to Settlement Plate as the soil layering under each Settlement Plate is unique, and may behave differently even under the same ground level.
* Surcharge load is a certain thickness of sand which weighs the ground down, thereby causing settlement and improvement of the underlying soil's properties.
* The '7day_rate' is the amount of settlement that has occurred over the last 7 days. 
* The "Latest_GL" is the last reported ground elevation in units 'mCD'. A larger number indicates the particular plate is loaded more, and hence should record more settlement.
* Each Settlement Plate is surcharged on a particular date known as the "Surcharge_Complete_Date" which indicates the date from when the major of the settlement occurs.
* The Holding_period is the period of time in days, between the "Surcharge_Complete_Date" and "Latest_Date" when the settlement was last reported. A longer 'Holding_Period' usually means that the settlement has had time to taper off. Shorter periods may result in more ongoing settlement.
* The "Asaoka_DOC" denotes the Degree of Consolidation (DOC) based on the Asaoka Assessment method. It is a measure in units %, of by how much the ground has consolidated. A DOC of 100 % means no more settlement is expected, while a DOC between 90 % and 100 % means the settlement is tapering and a DOC less than 90% indicate on-going settlement. DOC less than 90 % non-compliant to the requirements.
* The 'Latest_GL' should be a minimum 16.9mCD to be compliant with Port specifications.
* The '7day_rate' has to be less than or equal to 4 to be compliant. If this value is greater, indicate that the plate is non compliant.
* When asked for a summary or overview, make sure to provide the Settlement Plate ID, along with the respective "latest_Settlement", "Latest_GL", "Latest_Date", "Asaoka_DOC", "Holding_period" and "7day_rate" which is what users are interested in. Make sure to show the raw values and not interpreted values of these key parameters.
* You do not need to comment on the format of the Settlement Plate ID as this is just a reference identifier. 
* When asked for a summary or overview of the Settlement Plate data, provide a table at the end of your response.
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
def build_messages(system, memory, user_input, classifier_data=None, func_guidelines=None, func_output=None):
    messages = []
    # Always add the common instructions
    messages.append(system[0])
    # Add function-specific context/guidelines
    if func_guidelines:
        messages.append({"role": "system", "content": func_guidelines})
    else:
        messages.append(system[1])  # fallback to default context/guidelines

    # Add function execution/classifier info if present
    if classifier_data or func_output:
        block = ""
        if classifier_data:
            block += f"Function: {classifier_data.get('Function')}\nParams: {classifier_data.get('Params')}\n"
        if func_output:
            block += f"Output: {func_output}\n"
        messages.append({"role": "system", "content": block.strip()})

    # Add recap if requested
    if "@recap" in user_input and memory:
        recap = "=== PREVIOUS CONVERSATION ===\n"
        for i, (user_msg, assistant_msg) in enumerate(memory, 1):
            recap += f"User ({i}): {user_msg.strip()}\n"
            recap += f"Assistant ({i}): {assistant_msg.strip()}\n"
        messages.append({"role": "system", "content": recap.strip()})

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
                * Answer user prompt strictly based on the context and the information given by the user. 
                * A background of the project is given below however only answer the User query. The Background is only for light referencing and basic guidelines. The main guidelines to help to answer the user query will be given in the "functional instructions" section.
                * NOTE: Your purpose is to answer questions based on the given prompt and/or context ONLY. Do not make anything up as all information is provided in the given information. ONLY ask questions if really necessary (i.e. if some terminology is unclear).
                
                == BACKGROUND (ONLY FOR LIGHT REFERENCING) ==
                Consider the following as an overview of the Tuas Terminal Phase 2 Project in Singapore and use of Settlement Plates:
                * A large land-reclamation project in Singapore which comprises the construction of wharf line for the future Tuas megaport.
                * 365 hectares of land reclamation, constructing a 9km wharf line.
                * Settlement Plates are instruments installed every 1600 square metres to measure the change in ground level under sand surcharge. 
                * Sand surcharge is used to weigh the reclaimed land, comprising mostly clay and sand, downwards which results in settlement and improvement of the reclaimed soil properties.
                * Currently, 1900 Settlement Plates are installed at the project
                * Settlement plates are named in a format similar to 'F3-R03a-SM-04' where 'F3' Indicates it belongs to the project, R03a is a specific area within the project, 'SM' indicates it is a Settlement Plates and the last two digits is the plate's index number. Do not comment on these names as they are fixed and require no interpretation.
                * The Settlement Plates are crucial for completion of Soil Improvement Works, where the settlement measurements have to meet certain criteria prior to being approved for removal.
                * The criteria is Asaoka DOC greater than 90%, a ground level above 16.9mCD and rate of settlement less than 4mm.'''
            )},
            {"role": "system", "content": (
                '''=== FUNCTIONAL INSTRUCTIONS ===:
                You are a helpful chatbot tasked to continue the conversation based on the User query and conversation history ONLY.
                * Main goal is to continue to conversation logically. (for example: if the user says "Thank you", say "Your welcome").
                * If the user says something that refers to the previous conversations from the conversation history provided (if any), then respond accordingly. (for example. if the user says "can you tell me about point 2 again?", please look at the conversation history for clues and answer the question correctly).
                * REFER ONLY to the instructions, background, or conversation history. 
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
            summary = get_summary(response)
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
