import streamlit as st
import whisper
import tempfile
import os
import re
import subprocess

def sanitize_filename(filename):
    """Sanitize file name to remove special characters."""
    return re.sub(r'[^\w\-_\.]', '_', filename)

def validate_audio(file_path):
    """Validate audio file using FFmpeg."""
    try:
        # Run FFmpeg with a dummy output to ensure proper validation
        dummy_output = "test.wav"
        result = subprocess.run(
            ["ffmpeg", "-i", file_path, "-t", "1", dummy_output, "-y"],
            check=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        # Remove the dummy file if successfully created
        if os.path.exists(dummy_output):
            os.remove(dummy_output)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg validation failed: {e.stderr.decode()}")
        return False

# Initialize session state
if "transcriptions" not in st.session_state:
    st.session_state["transcriptions"] = {}

st.title("Multi-lingual Transcription using Whisper")

# Allow multiple file uploads
audio_files = st.file_uploader("Upload your audio files", type=["wav", "mp3", "m4a"], accept_multiple_files=True)
model = whisper.load_model("base")

if st.sidebar.button("Transcribe Audio"):
    if audio_files:
        for audio_file in audio_files:
            try:
                # Sanitize file name
                safe_filename = sanitize_filename(audio_file.name)

                # Save audio file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_audio:
                    temp_audio.write(audio_file.read())
                    temp_audio_path = temp_audio.name

                # Validate audio file
                if not validate_audio(temp_audio_path):
                    st.sidebar.error(f"Invalid or unsupported audio file: {audio_file.name}")
                    continue

                # Perform transcription
                transcription = model.transcribe(temp_audio_path)
                transcription_text = transcription["text"]

                # Store transcription in session state
                st.session_state["transcriptions"][audio_file.name] = transcription_text

                # Remove temporary audio file
                os.remove(temp_audio_path)

            except Exception as e:
                st.sidebar.error(f"An error occurred with {audio_file.name}: {e}")
    else:
        st.sidebar.error("Please upload at least one audio file.")

# Display stored transcriptions
if st.session_state["transcriptions"]:
    for file_name, transcription_text in st.session_state["transcriptions"].items():
        st.markdown(f"### Transcription for {file_name}")
        st.text_area(f"Transcription ({file_name})", transcription_text, height=200)
        st.download_button(
            label=f"Download Transcription ({file_name})",
            data=transcription_text.encode("utf-8"),
            file_name=f"{sanitize_filename(file_name).split('.')[0]}_transcription.txt",
            mime="text/plain",
        )
