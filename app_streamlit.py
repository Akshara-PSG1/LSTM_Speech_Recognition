import os
import joblib
import webbrowser
import librosa
import numpy as np
import soundfile as sf
import streamlit as st
import ctypes
from ctypes import cast, POINTER
import subprocess
from tensorflow.keras.models import load_model
import screen_brightness_control as sbc
import wave
import io
from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
import pyaudio

def record_audio_pyaudio(duration):
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1  # Mono
    fs = 44100  # Record at 44100 samples per second
    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    # Start recording
    st.write("Recording...")
    stream = p.open(format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True)
    frames = []  # Initialize array to store frames

    for i in range(0, int(fs / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded data into a byte stream
    audio_stream = io.BytesIO()
    wf = wave.open(audio_stream, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    audio_stream.seek(0)  # Reset stream position to the start

    st.write("Recording finished.")
    return audio_stream
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    model = load_model(model_path)
    return model

@st.cache(allow_output_mutation=True)
def load_label_encoder(label_encoder_path):
    label_encoder = joblib.load(label_encoder_path)  # Load your saved label encoder here
    return label_encoder


# Function to extract MFCC from an audio file
def extract_mfcc(audio_data, max_pad_len=54, n_mfcc=13):
    try:
        mfcc = librosa.feature.mfcc(y=audio_data, sr=44100, n_mfcc=n_mfcc)
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc
    except Exception as e:
        st.error(f"Error extracting MFCCs: {e}")
        return None

# Function to predict the command from audio data
def predict_command(audio_data, model, label_encoder):
    mfcc = extract_mfcc(audio_data)
    if mfcc is not None:
        mfcc_reshaped = mfcc.reshape(1, mfcc.shape[1], mfcc.shape[0])
        prediction_prob = model.predict(mfcc_reshaped)
        predicted_class = np.argmax(prediction_prob, axis=1)[0]
        command_name = label_encoder.inverse_transform([predicted_class])
        return command_name[0]
    else:
        return None

def set_brightness(level):
    sbc.set_brightness(level)

def mute_sound():
    sessions = AudioUtilities.GetAllSessions()
    for session in sessions:
        volume = session._ctl.QueryInterface(ISimpleAudioVolume)
        volume.SetMasterVolume(0, None)

def start_screen_recording():
    subprocess.Popen(['ffmpeg', '-video_size', '1920x1080', '-framerate', '25', '-f', 'x11grab', '-i', ':0.0', 'output.mp4'])

def handle_shutdown():
    st.write("Are You Sure You want to Shutdown the system?")
    if st.button("Yes"):
        st.write("Shutting down the system...")
        os.system("shutdown /s /t 5")  

    if st.button("No"):
        st.write("OK, not shutting down. Ready for the next command.")
        
def handle_restart():
    st.write("Are You Sure You want to Restart the system?")
    if st.button("Yes"):
        st.write("Restarting the system...")
        os.system("shutdown /r /t 5") 

    if st.button("No"):
        st.write("OK, not restarting. Ready for the next command.")

def open_application(app_name):
    if app_name == "chrome":
        subprocess.Popen(["C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"])
    if app_name == "notepad":
        subprocess.Popen('notepad.exe') 
        st.write("Notepad started.")
    if app_name == "Calculator":
        subprocess.Popen('calc.exe') 
        st.write("Calculator started.")

def execute_command(command):
    if command == "Dim":
        st.write("Dimming brightness...")
        set_brightness(50) 
        
    elif command == "Play":
        st.write("Opening YouTube...")
        webbrowser.open('https://www.youtube.com')
        
    elif command == "Mute":
        st.write("Muting sound...")
        mute_sound()
        
    elif command == "Record":
        st.write("Starting screen recording...")
        start_screen_recording()
        
    elif command == "Search":
        st.write("Opening Google Chrome and searching...")
        search_query = st.text_input("What do you want to search for?")
        webbrowser.open(f'https://www.google.com/search?q={search_query}')
        
    elif command == "Shutdown":
        handle_shutdown()
        
    elif command == "Start":
       app_choice = st.selectbox("Which application would you like to start?", 
                                  ("Chrome", "Calculator", "Notepad"))
       if st.button("Start Application"):
           open_application(app_choice)
    
    elif command == "Restart":
        handle_restart()
        
    else:
        st.write("Command not recognized.")
        
        
# Streamlit App Layout
st.title("Speech Command Recognition System")

label_encoder_path = 'label_encoder.pkl' 
label_encoder = load_label_encoder(label_encoder_path)

model_path = "Speech_commands_LSTM.h5"
model = load_model(model_path)


if st.button("Record Audio"):
    audio_stream = record_audio_pyaudio(duration=3)
    audio_data, _ = librosa.load(audio_stream, sr=44100)  # Load audio using librosa for processing

if audio_data is not None:
    audio_data = audio_data.flatten()
    audio_data = audio_data.astype(np.float32)

    command = predict_command(audio_data, model, label_encoder)

    if command:
        st.write(f"Predicted Command: {command}")
        execute_command(command)
    else:
        st.write("Could not predict the command from the audio input.")
