import streamlit as st
from dispatcher import Dispatcher
from llm_main import LLMRouter
from main import Classifier

# --- Streamlit Chat UI for NL2Func Pipeline ---
st.set_page_config(page_title="NL2Func Chat", layout="wide")
st.title("NL2Func Chat Assistant")

# --- Sidebar Controls ---
st.sidebar.markdown("### Controls")
if st.sidebar.button("🗑️ Clear Chat"):
    # Clear chat but keep dispatcher
    disp = st.session_state.get("dispatcher")
    st.session_state.clear()
    if disp:
        st.session_state.dispatcher = disp
    st.rerun()

if "recap_mode" not in st.session_state:
    st.session_state.recap_mode = False
if "think_mode" not in st.session_state:
    st.session_state.think_mode = False
if "deep_mode" not in st.session_state:
    st.session_state.deep_mode = False

st.session_state.recap_mode = st.sidebar.toggle("Recap Mode (@recap)", value=st.session_state.recap_mode)
st.sidebar.caption("Include a summary of previous conversation in the prompt.")

def set_think():
    st.session_state.think_mode = True
    st.session_state.deep_mode = False

def set_deep():
    st.session_state.deep_mode = True
    st.session_state.think_mode = False

st.session_state.think_mode = st.sidebar.toggle("Think Mode (@think)", value=st.session_state.think_mode, on_change=set_think)
st.sidebar.caption("Use the 'think' model for more reasoning.")

st.session_state.deep_mode = st.sidebar.toggle("Deep Mode (@deep)", value=st.session_state.deep_mode, on_change=set_deep)
st.sidebar.caption("Use the 'deep' model for more detailed answers.")

# Ensure mutual exclusion
if st.session_state.think_mode and st.session_state.deep_mode:
    # If both got set somehow, default to think
    st.session_state.deep_mode = False



# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "slot_state" not in st.session_state:
    st.session_state.slot_state = None
if "dispatcher" not in st.session_state:
    st.session_state.classifier = Classifier()
    st.session_state.llm_router = LLMRouter(max_turns=5)
    st.session_state.dispatcher = Dispatcher(
        st.session_state.classifier,
        st.session_state.llm_router
    )

# --- Helper Functions ---
def add_message(role, message):
    st.session_state.chat_history.append((role, message))
    if len(st.session_state.chat_history) > 200:
        st.session_state.chat_history.pop(0)


def display_chat():
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            if isinstance(msg, str):
                st.markdown(msg)
            else:
                for chunk in msg:
                    st.markdown(chunk, unsafe_allow_html=True)

# --- LLM Streaming Response ---
def stream_response(user_input, func_name=None, params=None, func_output=None):
    # echo user
    add_message("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    placeholder = st.empty()
    full_response = ""
    with st.chat_message("assistant"), st.spinner("Assistant is typing..."):
        try:
            stream = st.session_state.llm_router.handle_user(
                user_input,
                func_name=func_name,
                classifier_data=(None if not func_name else {"Function": func_name, "Params": params, "Output": func_output}),
                func_output=func_output,
                stream=True,
            )
            for token in stream:
                full_response += token
                placeholder.markdown(full_response)
        except TypeError:
            resp = st.session_state.llm_router.handle_user(
                user_input,
                func_name=func_name,
                classifier_data=(None if not func_name else {"Function": func_name, "Params": params, "Output": func_output}),
                func_output=func_output,
            )
            full_response = resp if isinstance(resp, str) else str(resp)
            placeholder.markdown(full_response)
        except Exception as e:
            placeholder.markdown(f"**[Error: {e}]**")
            full_response = f"[Error: {e}]"

    add_message("assistant", full_response)

# --- Main Flow ---
# Render existing chat
display_chat()

# Handle new input
given_input = st.chat_input("Type your message...", key="input")
if given_input:
    # Top-level echo before any logic
    input_text = given_input.strip()
    tags = []
    if st.session_state.recap_mode and "@recap" not in input_text:
        tags.append("@recap")
    if st.session_state.think_mode and "@think" not in input_text:
        tags.append("@think")
    if st.session_state.deep_mode and "@deep" not in input_text:
        tags.append("@deep")
    # Remove mutually exclusive tags if both present
    if st.session_state.think_mode and st.session_state.deep_mode:
        tags = [t for t in tags if t != "@deep"]  # Prefer think
    # Prepend tags to input
    if tags:
        input_text = " ".join(tags) + " | " + input_text
    # dispatch
    disp = st.session_state.dispatcher

    # Slot-filling active
    if st.session_state.slot_state:
        slot_info = st.session_state.slot_state
        slot = slot_info["slots_needed"][0]
        answer = input_text
        print(f"[DEBUG] Slot-filling context before parse: {slot_info['aux_ctx']}")
        # echo answer
        add_message("user", answer)
        with st.chat_message("user"):
            st.markdown(answer)

        if answer.lower() in {"skip", "never mind", "nvm", "nah"}:
            skip_msg = "Skipping function; sending to LLM."
            add_message("assistant", skip_msg)
            with st.chat_message("assistant"):
                st.markdown(skip_msg)
            st.session_state.slot_state = None
            stream_response(slot_info["orig_query"])
        else:
            # build aux_ctx correctly
            slot_info["aux_ctx"] += f"\n{slot}: {answer}"
            # Remove the filled slot from slots_needed
            if slot_info["slots_needed"]:
                slot_info["slots_needed"].pop(0)
            try:
                print(f"[DEBUG] Calling pure_parse with aux_ctx: {slot_info['aux_ctx']}")
                params = disp.pure_parse(slot_info["aux_ctx"], slot_info["func_name"])
                print(f"[DEBUG] pure_parse returned params: {params}")
                out = disp.run_function(slot_info["func_name"], params)
                print(f"[DEBUG] run_function output: {out}")
                st.session_state.slot_state = None
                stream_response(slot_info["orig_query"], slot_info["func_name"], params, out)
            except Exception as e:
                print(f"[DEBUG] Exception in slot-filling: {e}")
                if hasattr(e, "slot"):
                    print(f"[DEBUG] MissingSlot: {e.slot}")
                    # Update slots_needed to ask for the next missing slot
                    st.session_state.slot_state["slots_needed"] = [e.slot]
                    prompt = f"What's your {e.slot}?"
                    add_message("assistant", prompt)
                    with st.chat_message("assistant"):
                        st.markdown(prompt)
                    # slot_state remains
                else:
                    err_msg = f"Error: {e}"
                    add_message("assistant", err_msg)
                    with st.chat_message("assistant"):
                        st.markdown(err_msg)
                    st.session_state.slot_state = None
    else:
        # initial classify
        func_name, _ = disp.classify(input_text)
        if func_name:
            try:
                params = disp.pure_parse(input_text, func_name)
                out = disp.run_function(func_name, params)
                stream_response(input_text, func_name, params, out)
            except Exception as e:
                if hasattr(e, "slot"):
                    # set up slot_state and prompt
                    st.session_state.slot_state = {
                        "func_name": func_name,
                        "aux_ctx": input_text,
                        "slots_needed": [e.slot],
                        "orig_query": input_text
                    }
                    prompt = f"What's your {e.slot}?"
                    add_message("assistant", prompt)
                    with st.chat_message("assistant"):
                        st.markdown(prompt)
                else:
                    err_msg = f"Error: {e}"
                    add_message("assistant", err_msg)
                    with st.chat_message("assistant"):
                        st.markdown(err_msg)
        else:
            # direct LLM
            stream_response(input_text)
