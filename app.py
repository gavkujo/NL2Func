import streamlit as st
from dispatcher import Dispatcher
from llm_main import LLMRouter
from main import Classifier
import re
import os 

def render_assistant_message(msg, func_name=None):
    """
    Render assistant message with <think>...</think> blocks shown first, then the main answer.
    If there are no <think> blocks, just show the message as normal.
    """
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    thinks = think_pattern.findall(msg)
    main = think_pattern.sub("", msg).strip()

    # Show each <think> block first (if any)
    for i, think_content in enumerate(thinks, 1):
        st.markdown(
            f"<div style='margin-bottom:0.5em; padding:0.5em; background:#f6f6f6; border-radius:6px;'>"
            f"<b>💡 Thought process {i}:</b><br>"
            f"<span style='font-size:0.92em; color:#888; font-style:italic;'>{think_content.strip()}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
    # Then show the main answer if any
    if main:
        st.markdown(main, unsafe_allow_html=True)
    if "==PDF ALERT==" in msg:
        links = []
        #if os.path.exists("static/asaoka_report.pdf"):
        if "reporter_Asaoka" in func_name:
            links.append("[Download Asaoka Report PDF](static/asaoka_report.pdf)")
        #if os.path.exists("static/Combined_settlement_plot.pdf"):
        if "plot_combi_S" in func_name:
            links.append("[Download Combined Settlement Plot PDF](static/Combined_settlement_plot.pdf)")
        if links:
            st.markdown("<br>".join(links), unsafe_allow_html=True)

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
            if role == "assistant" and isinstance(msg, str) and "<think>" in msg:
                render_assistant_message(msg)
            elif isinstance(msg, str):
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
                # Live preview: render <think> blocks as plain text while streaming
                # (final rendering will be styled in display_chat)
                placeholder.markdown(full_response.replace("<think>", "\n---\n💡 **Thought process:**\n").replace("</think>", "\n---\n"), unsafe_allow_html=True)
        except TypeError:
            resp = st.session_state.llm_router.handle_user(
                user_input,
                func_name=func_name,
                classifier_data=(None if not func_name else {"Function": func_name, "Params": params, "Output": func_output}),
                func_output=func_output,
            )
            full_response = resp if isinstance(resp, str) else str(resp)
            placeholder.markdown(full_response.replace("<think>", "\n---\n💡 **Thought process:**\n").replace("</think>", "\n---\n"), unsafe_allow_html=True)
        except Exception as e:
            placeholder.markdown(f"**[Error: {e}]**")
            full_response = f"[Error: {e}]"

    add_message("assistant", full_response)

# --- Main Flow ---
display_chat()

given_input = st.chat_input("Type your message...", key="input")
if given_input:
    input_text = given_input.strip()
    disp = st.session_state.dispatcher

    # --- Slot-filling active ---
    if st.session_state.slot_state:
        slot_info = st.session_state.slot_state
        slot = slot_info["slots_needed"][0]
        answer = input_text  # DO NOT inject tags here!
        print(f"[DEBUG] Slot-filling context before parse: {slot_info['aux_ctx']}")
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
            slot_info["aux_ctx"] += f"\n{slot}: {answer}"
            if slot_info["slots_needed"]:
                slot_info["slots_needed"].pop(0)
            try:
                params = disp.pure_parse(slot_info["aux_ctx"], slot_info["func_name"])
                out = disp.run_function(slot_info["func_name"], params)
                st.session_state.slot_state = None
                
                # Add this check to skip LLM for PDF functions in slot-filling path
                if slot_info["func_name"] in ["reporter_Asaoka", "plot_combi_S"]:
                    with st.chat_message("assistant"), st.spinner("Generating PDF..."):
                        pdf_msg = "==PDF ALERT==\nA PDF has been generated and is ready for download below."
                        add_message("assistant", pdf_msg)
                        render_assistant_message(pdf_msg, func_name=slot_info["func_name"])
                else:
                    stream_response(slot_info["orig_query"], slot_info["func_name"], params, out)
            except Exception as e:
                print(f"[DEBUG] Exception in slot-filling: {e}")
                if hasattr(e, "slot"):
                    print(f"[DEBUG] MissingSlot: {e.slot}")
                    st.session_state.slot_state["slots_needed"] = [e.slot]
                    prompt = f"What's your {e.slot}?"
                    add_message("assistant", prompt)
                    with st.chat_message("assistant"):
                        st.markdown(prompt)
                else:
                    err_msg = f"Error: {e}"
                    add_message("assistant", err_msg)
                    with st.chat_message("assistant"):
                        st.markdown(err_msg)
                    st.session_state.slot_state = None

    # --- Initial classify (inject tags here only) ---
    else:
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
        # Prepend tags to input (space-separated)
        if tags:
            input_text = " ".join(tags) + " " + input_text

        func_name, _ = disp.classify(input_text)
        if func_name:
            try:
                params = disp.pure_parse(input_text, func_name)
                out = disp.run_function(func_name, params)
                # Show spinner while generating PDF
                if func_name in ["reporter_Asaoka", "plot_combi_S"]:
                    with st.chat_message("assistant"), st.spinner("Generating PDF..."):
                        pdf_msg = "==PDF ALERT==\nA PDF has been generated and is ready for download below."
                        add_message("assistant", pdf_msg)
                        render_assistant_message(pdf_msg, func_name=func_name)
                else:
                    stream_response(input_text, func_name, params, out)
            except Exception as e:
                if hasattr(e, "slot"):
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
                    print(f"[DEBUG] Exception caught: {e}")
                    err_msg = f"Error: {e}"
                    add_message("assistant", err_msg)
                    with st.chat_message("assistant"):
                        st.markdown(err_msg)
        else:
            stream_response(input_text)