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
    # Only update UI if not streaming (streaming handled separately)
    if not callable(msg):
        # Only display chat if not in the middle of a Streamlit rerun (prevents duplicate rendering)
        if st.session_state.get("_suppress_display", False):
            return
        display_chat()

# --- Main Chat Logic ---

def process_user_message(user_msg):
    disp = st.session_state.dispatcher
    # Prevent duplicate chat history on rerun
    if st.session_state.get("_processing", False):
        return
    st.session_state["_processing"] = True

    add_message("user", user_msg)

    # If in slot-filling mode
    if st.session_state.slot_state:
        slot_state = st.session_state.slot_state
        slot = slot_state["slots_needed"][0]
        answer = user_msg.strip()
        if answer.lower() in ["skip", "never mind"]:
            add_message("assistant", "Okay, skipping function. Sending your query to the LLM.")
            st.session_state.slot_state = None
            stream_llm_response(disp.llm_router, slot_state["orig_query"])
            st.session_state["_processing"] = False
            return
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
        st.session_state["_processing"] = False
        return

    func_name, func = disp.classify(user_msg)
    if func_name:
        try:
            params = disp.pure_parse(user_msg, func_name)
            out = disp.run_function(func_name, params)
            stream_llm_response(disp.llm_router, user_msg, func_name, params, out)
        except Exception as e:
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
        st.session_state["_processing"] = False
    else:
        stream_llm_response(disp.llm_router, user_msg)
        st.session_state["_processing"] = False


# --- LLM Streaming Helper ---
def stream_llm_response(llm_router, user_input, func_name=None, params=None, func_output=None):
    """
    Streams the LLM response to the chat UI. If the backend supports streaming, display tokens as they arrive.
    """
    # Prevent duplicate chat history on rerun
    if st.session_state.get("_suppress_display", False):
        return
    st.session_state["_suppress_display"] = True
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
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
    st.session_state["_suppress_display"] = False

# --- UI ---

display_chat()
user_input = st.chat_input("Type your message and press Enter...")
if user_input:
    process_user_message(user_input)
