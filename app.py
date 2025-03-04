import streamlit as st
from LegalAIBot.LegalAIBot import LegalAIBot
from dotenv import load_dotenv

load_dotenv()

bot = LegalAIBot()

# Initialize session state for messages and references
if "messages" not in st.session_state:
    st.session_state.messages = []
if "references" not in st.session_state:
    st.session_state.references = []
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False


# Function to display chat messages
def display_messages():
    for i, (message, is_user) in enumerate(st.session_state.messages):
        with st.chat_message("user" if is_user else "assistant"):
            st.markdown(message)
            if not is_user:
                st.markdown("**References:**")
                for j, ref in enumerate(st.session_state.references[i]):
                    if f"show_ref_{i}_{j}" not in st.session_state:
                        st.session_state[f"show_ref_{i}_{j}"] = False

                    if st.button(
                        f"{'Hide' if st.session_state[f'show_ref_{i}_{j}'] else 'Show'} Reference",
                        key=f"button_{i}_{j}",
                    ):
                        st.session_state[f"show_ref_{i}_{j}"] = not st.session_state[
                            f"show_ref_{i}_{j}"
                        ]
                        st.rerun()

                    if st.session_state[f"show_ref_{i}_{j}"]:
                        st.markdown(
                            f'<p style="color:grey; font-size: 15px; text-align:justify;">{ref}</p>',
                            unsafe_allow_html=True,
                        )


# Main chat interface
st.title("Legal AI Chatbot")

# User input
if st.session_state.is_processing:
    pass
else:
    user_input = st.chat_input("Ask a legal doubt...")
    if user_input:
        # Display user message
        st.session_state.messages.append((user_input, True))
        st.session_state.is_processing = True
        st.session_state.references.append(
            []
        )  # Initialize references for the new message
        st.rerun()

# Display existing chat messages
display_messages()

# Generate assistant response
if st.session_state.is_processing:
    with st.spinner("Thinking..."):
        user_input = st.session_state.messages[-1][0]
        try:
            response, references = bot.generate_output(user_input)
            st.session_state.messages.append((response, False))
            st.session_state.references.append(references)
        except Exception as e:
            st.session_state.messages.append(
                ("Something went wrong, please try again later.", False)
            )
            st.session_state.is_processing = False
        finally:
            st.session_state.is_processing = False
            st.rerun()
