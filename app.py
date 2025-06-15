#!/usr/bin/env python

# app.py

import streamlit as st
from pathlib import Path
import sys

# --- Add src to path for imports ---
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

# --- Import Core Services and Models ---
try:
    from services.rag_service import RAGService
    from services.tts import TTSService
    from core.models import RAGResponse
except ImportError as e:
    st.error(f"Failed to import a required service or model. Please ensure your project structure is correct. Error: {e}")
    st.stop()


# --- Page Configuration ---
st.set_page_config(
    page_title="Youtubot 🤖",
    page_icon="🤖",
    layout="wide"
)

# --- Custom CSS for Enhanced Styling ---
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #1a1a1a;
    }
    /* Main title styling */
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        color: #64b5f6; /* A nice light blue */
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }
    .stButton>button {
        transition: transform 0.1s ease-in-out;
    }
    .stButton>button:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)


# --- Service Initialization ---
@st.cache_resource
def load_services():
    """Load and cache RAG and TTS services for performance."""
    try:
        rag_service = RAGService()
        tts_service = TTSService()
        return rag_service, tts_service
    except Exception as e:
        st.error(f"Fatal error during service initialization: {e}. Check model configs.")
        st.stop()

rag_service, tts_service = load_services()

# --- Session State Management ---
if "page" not in st.session_state:
    st.session_state.page = "setup"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_name" not in st.session_state:
    st.session_state.session_name = ""


# --- UI Rendering Functions ---

def render_setup_page():
    """Renders the main page for creating or loading a session."""
    st.markdown('<h1 class="main-title">Youtubot Session Manager 🤖</h1>', unsafe_allow_html=True)
    
    mode = st.radio("Choose your action:", ("Start a New Chat", "Load Saved Session"), horizontal=True, key="main_mode")
    st.markdown("---")

    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        with st.container(border=True):
            if mode == "Start a New Chat":
                st.subheader("Start a New Chat Session")
                content_type = st.radio("Content Type", ("Single Video URL", "Playlist URL"), key="content_type_radio", horizontal=True)
                url_placeholder = "https://www.youtube.com/playlist?list=..." if content_type == "Playlist URL" else "https://www.youtube.com/watch?v=..."
                content_url = st.text_input(f"🔗 YouTube {content_type}", placeholder=url_placeholder)
                lang_options = {"English": "en", "Spanish": "es", "French": "fr", "German": "de", "Turkish": "tr"}
                selected_lang_name = st.selectbox("🌐 Select Video Language", options=list(lang_options.keys()))
                lang_code = lang_options[selected_lang_name]

                st.markdown("---")
                # Whisper Transcription Option ---
                use_whisper_checkbox = st.checkbox(
                    "Use local Whisper transcription (slower, for any video)", 
                    value=False,
                    help="Check this to download the video's audio and transcribe it on your PC. This is much slower but works even for videos without subtitles."
                )
                
                save_session_checkbox = st.checkbox("Save this session for later?", value=False)
                session_name = ""
                if save_session_checkbox:
                    session_name = st.text_input("📝 Session Name (required for saving)", placeholder="e.g., 'ai_lectures_mit'")
                
                if st.button("🚀 Start Chatting", use_container_width=True, type="primary"):
                    if not content_url:
                        st.warning("Please provide a valid URL.", icon="⚠️")
                    elif save_session_checkbox and not session_name:
                        st.warning("Please provide a session name if you want to save.", icon="⚠️")
                    else:
                        with st.spinner(f"Processing content... This may take a while, especially with Whisper."):
                            success = rag_service.process_content(
                                content_url, 
                                lang_code, 
                                'playlist' if content_type == 'Playlist URL' else 'video',
                                use_whisper=use_whisper_checkbox
                            )
                            if success:
                                if save_session_checkbox:
                                    rag_service.save_index_to_disk(session_name)
                                    st.session_state.session_name = session_name
                                else:
                                    st.session_state.session_name = "Temporary Session"
                                st.session_state.messages = []
                                st.session_state.page = "chat"
                                st.rerun()
                            else:
                                st.error("Failed to process content. Even Whisper might fail on some videos.", icon="🚨")

            elif mode == "Load Saved Session":
                st.subheader("Load or Delete a Saved Knowledge Base")
                saved_sessions = rag_service.list_saved_sessions()
                if not saved_sessions:
                    st.info("No saved sessions found. Create one first!")
                else:
                    session_to_load = st.selectbox("Select a session:", saved_sessions)
                    st.markdown("---")
                    load_col, delete_col = st.columns([2, 1])
                    with load_col:
                        if st.button("🚀 Load Session", use_container_width=True, type="primary"):
                            with st.spinner(f"Loading session '{session_to_load}'..."):
                                success = rag_service.load_index_from_disk(session_to_load)
                                if success:
                                    st.session_state.session_name = session_to_load
                                    st.session_state.messages = []
                                    st.session_state.page = "chat"
                                    st.rerun()
                                else:
                                    st.error("Failed to load the selected session.", icon="🚨")
                    with delete_col:
                        confirm_delete = st.checkbox(f"Confirm deletion", key=f"delete_confirm_{session_to_load}")
                        if st.button("🗑️ Delete Session", use_container_width=True, disabled=not confirm_delete):
                            success = rag_service.delete_session(session_to_load)
                            if success:
                                st.success(f"Session '{session_to_load}' has been deleted.")
                                st.rerun()
                            else:
                                st.error("An error occurred while deleting the session.")

def render_chat_page():
    """Renders the main chat interface with a persistent sidebar."""
    
    # --- Sidebar for Session Controls ---
    with st.sidebar:
        st.header(f"📚 Session: {st.session_state.session_name}")
        if rag_service.processed_videos_metadata:
            st.info(f"{len(rag_service.processed_videos_metadata)} videos loaded.")
        
        st.divider()
        st.subheader("⚙️ Controls")
        if st.button("↩️ New/Load Session", use_container_width=True):
            st.session_state.page = "setup"
            st.rerun()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        st.subheader("🌐 Response Language")
        video_language_code = rag_service.processed_videos_metadata[0].get('language', 'en') if rag_service.processed_videos_metadata else 'en'
        lang_map = {"English": "en", "Spanish": "es", "French": "fr", "German": "de", "Turkish": "tr"}
        video_language_name = next((name for name, code in lang_map.items() if code == video_language_code), "English")
        lang_options = {f"Automatic ({video_language_name})": "auto", **lang_map}
        
        selected_lang_key = st.selectbox("Force response language:", options=list(lang_options.keys()), key="override_language_select_key")
        st.session_state.override_language_select = lang_options[selected_lang_key]

        st.divider()
        st.subheader("🔊 Text-to-Speech")
        if tts_service.is_available():
            st.checkbox("Enable Online TTS", value=True, key="use_tts_enabled")
        else:
            st.warning("⚠️ TTS service is not available.")

    # --- Main Chat Area ---
    st.title("💬 Chat with Youtubot")

    # Display existing messages with custom avatars
    for message in st.session_state.messages:
        role = message["role"]
        avatar = "🧑‍💻" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(message["content"])
            if role == "assistant" and "raw_response" in message:
                display_assistant_extras(message)

    # User chat input widget
    if prompt := st.chat_input(f"Ask something about '{st.session_state.session_name}'..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)
            
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Youtubot is thinking..."):
                override_lang = st.session_state.get("override_language_select")
                final_override_lang = override_lang if override_lang and override_lang != "auto" else None
                response: RAGResponse = rag_service.generate_response(prompt, override_language=final_override_lang)
                
                assistant_message = {"role": "assistant", "content": response.answer, "raw_response": response}
                st.markdown(response.answer)
                display_assistant_extras(assistant_message)
                st.session_state.messages.append(assistant_message)

def display_assistant_extras(message):
    """Displays the TTS button and source expander for an assistant message."""
    raw_response = message["raw_response"]
    if st.session_state.get("use_tts_enabled", False) and tts_service.is_available():
        lang_code = raw_response.language
        button_key = f"tts_{hash(message['content'])}"
        if st.button("🔊 Play", key=button_key, help="Listen to the response"):
            with st.spinner("Generating speech..."):
                audio_data = tts_service.generate_speech(message['content'], lang_code)
                if audio_data:
                    st.audio(audio_data, format="audio/mp3")
                else:
                    st.error("Failed to generate speech.")

    if raw_response.sources:
        with st.expander("View Sources & Confidence"):
            for i, source in enumerate(raw_response.sources):
                st.markdown(f"**Source {i+1}:** [{source.video_title}]({source.video_url})")
                st.info(f"> {source.text_content[:250]}...")
            st.markdown(f"**Overall Confidence:** `{raw_response.confidence_score:.2f}`")


def main():
    if st.session_state.page == "setup":
        render_setup_page()
    else:
        render_chat_page()

if __name__ == "__main__":
    main()