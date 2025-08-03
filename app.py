import streamlit as st
from dispatcher import Dispatcher
from llm_main import LLMRouter
from main import Classifier
import sys

# --- Streamlit Chat UI for NL2Func Pipeline ---

st.set_page_config(page_title="NL2Func Chat", layout="wide")
st.title("NL2Func Chat Assistant")

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # [(role, message)]
if "slot_state" not in st.session_state:
    st.session_state.slot_state = None  # dict: {func_name, aux_ctx, slots_needed, collected}
if "dispatcher" not in st.session_state:
    st.session_state.classifier = Classifier()
    st.session_state.llm_router = LLMRouter(max_turns=5)
    st.session_state.dispatcher = Dispatcher(st.session_state.classifier, st.session_state.llm_router)

# --- Helper: Display Chat ---
def display_chat():
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

def add_message(role, msg):
    st.session_state.chat_history.append((role, msg))
    display_chat()

# --- Main Chat Logic ---
def process_user_message(user_msg):
    disp = st.session_state.dispatcher
    # If in slot-filling mode
    if st.session_state.slot_state:
        slot_state = st.session_state.slot_state
        slot = slot_state["slots_needed"][0]
        # User's reply is the slot value
        answer = user_msg.strip()
        if answer.lower() in ["skip", "never mind"]:
            add_message("assistant", "Okay, skipping function. Sending your query to the LLM.")
            st.session_state.slot_state = None
            disp.llm_router.handle_user(slot_state["orig_query"])
            add_message("assistant", "[LLM response sent]")
            return
        # Append slot answer to aux_ctx
        slot_state["aux_ctx"] += f"\n{slot}: {answer}"
        # Try to parse again
        try:
            params = disp.pure_parse(slot_state["aux_ctx"], slot_state["func_name"])
            # All slots filled!
            add_message("assistant", f"All parameters collected: {params}")
            out = disp.run_function(slot_state["func_name"], params)
            add_message("assistant", f"Function output: {out}")
            disp.build_and_send(slot_state["orig_query"], slot_state["func_name"], params, out)
            add_message("assistant", "[LLM response sent]")
            st.session_state.slot_state = None
        except Exception as e:
            # Still missing slots
            if hasattr(e, "slot"):
                st.session_state.slot_state["slots_needed"] = [e.slot]
                add_message("assistant", f"What's your {e.slot}?")
            else:
                add_message("assistant", f"Error: {e}")
                st.session_state.slot_state = None
        return
    # Not in slot-filling mode: new query
    add_message("user", user_msg)
    func_name, func = disp.classify(user_msg)
    if func_name:
        # Try to gather params (simulate dispatcher loop)
        try:
            params = disp.pure_parse(user_msg, func_name)
            # All slots filled (rare, unless user pasted slot answers)
            add_message("assistant", f"All parameters collected: {params}")
            out = disp.run_function(func_name, params)
            add_message("assistant", f"Function output: {out}")
            disp.build_and_send(user_msg, func_name, params, out)
            add_message("assistant", "[LLM response sent]")
        except Exception as e:
            # Missing slot: start slot-filling mode
            if hasattr(e, "slot"):
                st.session_state.slot_state = {
                    "func_name": func_name,
                    "aux_ctx": user_msg,
                    "slots_needed": [e.slot],
                    "collected": {},
                    "orig_query": user_msg
                }
                add_message("assistant", f"What's your {e.slot}?")
            else:
                add_message("assistant", f"Error: {e}")
    else:
        # No function matched, send to LLM
        disp.llm_router.handle_user(user_msg)
        add_message("assistant", "[LLM response sent]")

# --- UI ---
display_chat()
user_input = st.chat_input("Type your message and press Enter...")
if user_input:
    process_user_message(user_input)
