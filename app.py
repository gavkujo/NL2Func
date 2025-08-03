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

# --- Helper: Display Chat ---
def display_chat():
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            if isinstance(msg, str):
                st.markdown(msg)
            else:
                for chunk in msg:
                    st.markdown(chunk, unsafe_allow_html=True)

def add_message(role, msg):
    st.session_state.chat_history.append((role, msg))
    # Do not call display_chat() here; only call it once at the end of the script

# --- Main Chat Logic ---

def process_user_message(user_msg):
    add_message("user", user_msg)
    st.session_state["_pending_user_input"] = user_msg
    st.rerun()


# --- LLM Streaming Helper ---
def stream_llm_response(llm_router, user_input, func_name=None, params=None, func_output=None):
    """
    Streams the LLM response to the chat UI. If the backend supports streaming, display tokens as they arrive.
    """
    # Session loss warning
    if "chat_history" not in st.session_state or "dispatcher" not in st.session_state:
        st.warning("Session lost. Please refresh or restart the app.")
        st.stop()
    # Prevent duplicate chat history on rerun
    if st.session_state.get("_suppress_display", False):
        return
    st.session_state["_suppress_display"] = True
    try:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            with st.spinner("Assistant is typing..."):
                try:
                    stream = llm_router.handle_user(user_input, func_name=func_name, classifier_data=None if not func_name else {"Function": func_name, "Params": params, "Output": func_output}, func_output=func_output, stream=True)
                    for token in stream:
                        full_response += token
                        response_placeholder.markdown(full_response)
                except TypeError:
                    resp = llm_router.handle_user(user_input, func_name=func_name, classifier_data=None if not func_name else {"Function": func_name, "Params": params, "Output": func_output}, func_output=func_output)
                    full_response = resp if isinstance(resp, str) else str(resp)
                    response_placeholder.markdown(full_response)
                except Exception as e:
                    response_placeholder.markdown(f"**[Error streaming LLM response: {e}]**")
                    full_response = f"[Error: {e}]"
            st.session_state.chat_history.append(("assistant", full_response))
    finally:
        st.session_state["_suppress_display"] = False
# --- UI ---

# Clear chat button
col1, col2 = st.columns([1, 8])
with col1:
    if st.button("🗑️ Clear Chat", help="Clear all chat history and reset session"):
        st.session_state.chat_history = []
        st.session_state.slot_state = None
        if "_pending_user_input" in st.session_state:
            del st.session_state["_pending_user_input"]
        st.experimental_rerun()

# Handle pending user input (slot-filling, LLM, etc)
pending_input = st.session_state.pop("_pending_user_input", None)
if pending_input:
    disp = st.session_state.dispatcher
    # Slot-filling mode
    if st.session_state.slot_state:
        slot_state = st.session_state.slot_state
        slot = slot_state["slots_needed"][0]
        answer = pending_input.strip()
        add_message("user", answer)
        with st.spinner(f"Filling slot: {slot}..."):
            if answer.lower() in ["skip", "never mind"]:
                add_message("assistant", "Okay, skipping function. Sending your query to the LLM.")
                st.session_state.slot_state = None
                stream_llm_response(disp.llm_router, slot_state["orig_query"])
            else:
                # Append slot answer to aux_ctx
                slot_state["aux_ctx"] += f"\n{slot}: {answer}"
                try:
                    params = disp.pure_parse(slot_state["aux_ctx"], slot_state["func_name"])
                    out = disp.run_function(slot_state["func_name"], params)
                    stream_llm_response(disp.llm_router, slot_state["orig_query"], slot_state["func_name"], params, out)
                    st.session_state.slot_state = None
                except Exception as e:
                    if hasattr(e, "slot"):
                        st.session_state.slot_state["slots_needed"] = [e.slot]
                        add_message("assistant", f"What's your {e.slot}?")
                    else:
                        add_message("assistant", f"Error: {e}")
                        st.session_state.slot_state = None
    else:
        func_name, func = disp.classify(pending_input)
        if func_name:
            with st.spinner(f"Running function: {func_name}..."):
                try:
                    params = disp.pure_parse(pending_input, func_name)
                    out = disp.run_function(func_name, params)
                    stream_llm_response(disp.llm_router, pending_input, func_name, params, out)
                except Exception as e:
                    if hasattr(e, "slot"):
                        st.session_state.slot_state = {
                            "func_name": func_name,
                            "aux_ctx": pending_input,
                            "slots_needed": [e.slot],
                            "collected": {},
                            "orig_query": pending_input
                        }
                        add_message("assistant", f"What's your {e.slot}?")
                    else:
                        add_message("assistant", f"Error: {e}")
        else:
            with st.spinner("Thinking..."):
                stream_llm_response(disp.llm_router, pending_input)

user_input = st.chat_input("Type your message and press Enter...")
if user_input:
    process_user_message(user_input)
display_chat()
