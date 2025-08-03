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
                # For streaming, msg can be a generator or list
                for chunk in msg:
                    st.markdown(chunk, unsafe_allow_html=True)

def add_message(role, msg):
    st.session_state.chat_history.append((role, msg))
    # Only update UI if not streaming (streaming handled separately)
    if not callable(msg):
        display_chat()

# --- Main Chat Logic ---

def process_user_message(user_msg):
    disp = st.session_state.dispatcher
    # Always show user message as a chat bubble
    add_message("user", user_msg)

    # If in slot-filling mode
    if st.session_state.slot_state:
        slot_state = st.session_state.slot_state
        slot = slot_state["slots_needed"][0]
        answer = user_msg.strip()
        if answer.lower() in ["skip", "never mind"]:
            add_message("assistant", "Okay, skipping function. Sending your query to the LLM.")
            st.session_state.slot_state = None
            # Stream LLM response
            stream_llm_response(disp.llm_router, slot_state["orig_query"])
            return
        # Append slot answer to aux_ctx
        slot_state["aux_ctx"] += f"\n{slot}: {answer}"
        try:
            params = disp.pure_parse(slot_state["aux_ctx"], slot_state["func_name"])
            # All slots filled!
            add_message("assistant", f"All parameters collected: {params}")
            out = disp.run_function(slot_state["func_name"], params)
            add_message("assistant", f"Function output: {out}")
            # Stream LLM response
            stream_llm_response(disp.llm_router, slot_state["orig_query"], slot_state["func_name"], params, out)
            st.session_state.slot_state = None
        except Exception as e:
            if hasattr(e, "slot"):
                st.session_state.slot_state["slots_needed"] = [e.slot]
                add_message("assistant", f"What's your {e.slot}?")
            else:
                add_message("assistant", f"Error: {e}")
                st.session_state.slot_state = None
        return

    # Not in slot-filling mode: new query
    func_name, func = disp.classify(user_msg)
    if func_name:
        try:
            params = disp.pure_parse(user_msg, func_name)
            add_message("assistant", f"All parameters collected: {params}")
            out = disp.run_function(func_name, params)
            add_message("assistant", f"Function output: {out}")
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
    else:
        # No function matched, send to LLM
        stream_llm_response(disp.llm_router, user_msg)


# --- LLM Streaming Helper ---
def stream_llm_response(llm_router, user_input, func_name=None, params=None, func_output=None):
    """
    Streams the LLM response to the chat UI. If the backend supports streaming, display tokens as they arrive.
    """
    # Add a placeholder for the assistant message
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        # Try to use streaming if available
        try:
            # LLMRouter.handle_user should yield tokens if streaming is supported
            stream = llm_router.handle_user(user_input, func_name=func_name, classifier_data=None if not func_name else {"Function": func_name, "Params": params, "Output": func_output}, func_output=func_output, stream=True)
            for token in stream:
                full_response += token
                response_placeholder.markdown(full_response)
        except TypeError:
            # Fallback: handle_user returns full string
            resp = llm_router.handle_user(user_input, func_name=func_name, classifier_data=None if not func_name else {"Function": func_name, "Params": params, "Output": func_output}, func_output=func_output)
            full_response = resp if isinstance(resp, str) else str(resp)
            response_placeholder.markdown(full_response)
        except Exception as e:
            response_placeholder.markdown(f"**[Error streaming LLM response: {e}]**")
            full_response = f"[Error: {e}]"
        # Add the full response to chat history for future display
        st.session_state.chat_history.append(("assistant", full_response))

# --- UI ---

display_chat()
user_input = st.chat_input("Type your message and press Enter...")
if user_input:
    process_user_message(user_input)
