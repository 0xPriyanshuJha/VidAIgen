import asyncio
import tempfile
import streamlit as st
from embedchain import App
from mtranslate import translate


# Function to translate text using mtranslate
def translate_text(text, target_lang):
    return translate(text, target_lang)

# Async function to chat with video
async def chat_with_video(prompt, app):
    return await asyncio.to_thread(app.chat, prompt)

# Function to create an Embedchain bot instance
def embedchain_bot(db_path):
    return App.from_config(
        config={
            "llm": {"provider": "ollama", "config": {"model": "llama2", "base_url": "http://localhost:11434"}},
            "vectordb": {"provider": "chroma", "config": {"dir": db_path}},
            "embedder": {"provider": "ollama", "config": {"model": "llama2", "base_url": "http://localhost:11434"}},
        }
    )

# Create Streamlit app
st.title("Chat with YouTube Video 📺")
st.caption("This app allows you to chat with a YouTube video using Ollama's Llama2 model")

# Create a temporary directory to store the database
db_path = tempfile.mkdtemp()

# Create an instance of Embedchain App using Llama2
app = embedchain_bot(db_path)

# Get the YouTube video URL from the user
video_url = st.text_input("Enter YouTube Video URL", type="default")

# Language selection
languages = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh-cn",
    "Hindi": "hi",
    "Arabic": "ar",
    "Portuguese": "pt",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Urdu": "ur"
}

selected_language = st.selectbox("Select output language", options=list(languages.keys()), index=0)

# Add the video to the knowledge base
if video_url:
    app.add(video_url, data_type="youtube_video")
    st.success(f"Added {video_url} to knowledge base!")

    # Ask a question about the video
    prompt = st.text_input("Ask any question about the YouTube Video")

    # Chat with the video asynchronously
    if prompt:
        answer = asyncio.run(chat_with_video(prompt, app))
        
        # Translate the answer if necessary
        if selected_language != "English":
            translated_answer = translate_text(answer, target_lang=languages[selected_language])
        else:
            translated_answer = answer

        st.write(translated_answer)
