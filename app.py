import asyncio
import tempfile
import streamlit as st
from embedchain import App
from mtranslate import translate
from yt_dlp import YoutubeDL
import os


def translate_text(text, target_lang):
    """
    Translates the given text into the target language using mtranslate.

    Args:
        text (str): The text to translate.
        target_lang (str): The target language code (e.g., 'en', 'es').

    Returns:
        str: The translated text.
    """
    if not text:
        raise ValueError("Input text cannot be empty.")
    if not isinstance(target_lang, str) or len(target_lang) != 2:
        raise ValueError("Invalid target language code. Must be a two-letter ISO 639-1 code.")
    try:
        return translate(text, target_lang)
    except Exception as e:
        raise Exception(f"Translation failed: {str(e)}")


async def chat_with_video(prompt, app):
    """
    Chats asynchronously with the video knowledge base.

    Args:
        prompt (str): The user's question.
        app (App): The Embedchain App instance.

    Returns:
        str: The response from the knowledge base.
    """
    return await asyncio.to_thread(app.chat, prompt)


def embedchain_bot(db_path):
    """
    Creates an Embedchain bot instance.

    Args:
        db_path (str): The path to the database directory.

    Returns:
        App: An Embedchain App instance configured with the desired models and vector database.
    """
    return App.from_config(
        config={
            "llm": {"provider": "ollama", "config": {"model": "llama3.1", "base_url": "http://localhost:11434"}},
            "vectordb": {"provider": "chroma", "config": {"dir": db_path}},
            "embedder": {"provider": "ollama", "config": {"model": "llama3.1", "base_url": "http://localhost:11434"}},
        }
    )


def download_youtube_subtitles(url, temp_dir):
    """
    Downloads YouTube subtitles using yt_dlp.

    Args:
        url (str): The YouTube video URL.
        temp_dir (str): The directory to save subtitles.

    Returns:
        str: The title of the video.

    Raises:
        ValueError: If subtitles cannot be processed.
    """
    try:
        ydl_opts = {
            "skip_download": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en"],
            "outtmpl": os.path.join(temp_dir, "%(title)s.%(ext)s"),
        }
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info["title"]
    except Exception as e:
        raise ValueError(f"Failed to process video subtitles: {str(e)}")


# Streamlit App Configuration
st.title("Chat with YouTube Video 📺")
st.caption("This app allows you to chat with a YouTube video using Ollama's Llama2 model")

# Database for Embedchain
db_path = tempfile.mkdtemp()
app = embedchain_bot(db_path)

# YouTube URL Input
video_url = st.text_input("Enter YouTube Video URL", type="default")

if st.button("Process Video"):
    if video_url:
        try:
            temp_dir = tempfile.mkdtemp()
            video_title = download_youtube_subtitles(video_url, temp_dir)
            subtitle_path = os.path.join(temp_dir, f"{video_title}.en.vtt")

            if os.path.exists(subtitle_path):
                with open(subtitle_path, "r") as file:
                    subtitles = file.read()
                    app.add(subtitles, data_type="text")
                st.success(f"Added subtitles of {video_title} to knowledge base!")
            else:
                st.error("Subtitles not found. Ensure the video has subtitles available.")

        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
    else:
        st.error("Please enter a valid YouTube URL.")

# Language Selection
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

# Prompt Input and Response Generation
prompt = st.text_input("Ask any question about the YouTube Video")
if st.button("Ask Question"):
    if prompt:
        try:
            answer = asyncio.run(chat_with_video(prompt, app))

            # Translate the answer if necessary
            if selected_language != "English":
                translated_answer = translate_text(answer, target_lang=languages[selected_language])
            else:
                translated_answer = answer

            st.write(translated_answer)
        except Exception as e:
            st.error(f"An error occurred while chatting with the video: {str(e)}")
    else:
        st.error("Please enter a question.")
