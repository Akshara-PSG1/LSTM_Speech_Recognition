from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import soundfile as sf
import librosa
import os
import logging
from pydub import AudioSegment
from tensorflow.keras.models import load_model
import screen_brightness_control as sbc
import pyautogui
import webbrowser
import threading
import subprocess

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the model and label encoder
model = load_model("Speech_commands_LSTM.h5")
label_encoder = joblib.load("label_encoder.pkl")

# Variable to track if a recording is in progress
is_recording = False
output_file = 'C:\\Users\\lenovo\\Downloads\\Speech_recog\\output.mp4'

# Define the audio processing function
def extract_mfcc(audio_data, sample_rate=16000, max_pad_len=54, n_mfcc=13):
    """Extract MFCC features from audio data."""
    try:
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        logging.info("MFCC returned")
        return mfcc
    except Exception as e:
        logging.error(f"MFCC extraction failed: {e}")
        return None

@app.route("/predict", methods=["POST"])
def predict_command():
    logging.info("Received audio file for prediction.")
    
    if "file" not in request.files:
        logging.error("No audio file provided in request.")
        return jsonify({"error": "No file provided"}), 400

    audio_data = request.files["file"].read()
    temp_webm_path = "temp_audio.webm"  
    with open(temp_webm_path, "wb") as f:
        f.write(audio_data)
    logging.debug("Temporary WEBM audio file created.")

    try:
        audio = AudioSegment.from_file(temp_webm_path, format="webm")
        temp_wav_path = "temp_audio.wav"
        audio.export(temp_wav_path, format="wav")
        logging.debug("Converted WEBM to WAV.")
    except Exception as e:
        logging.error(f"Error converting audio file to WAV: {e}")
        return jsonify({"error": "Error converting audio file"}), 500
    finally:
        print("Webm Created")
        #os.remove(temp_webm_path)

    try:
        y, sr = librosa.load(temp_wav_path, sr=16000)
        logging.debug(f"Loaded audio file with shape {y.shape} and sample rate {sr}.")
    except Exception as e:
        logging.error(f"Error loading audio file: {e}")
        return jsonify({"error": "Error loading audio file"}), 500
    finally:
        os.remove(temp_wav_path)

    # Extract MFCC features
    mfcc = extract_mfcc(y)
    if mfcc is None:
        logging.error("MFCC extraction failed.")
        return jsonify({"error": "Error in audio processing"}), 500

    mfcc_reshaped = mfcc.reshape(1, mfcc.shape[1], mfcc.shape[0])
    try:
        prediction_prob = model.predict(mfcc_reshaped)
        predicted_class = np.argmax(prediction_prob, axis=1)[0]
        max_confidence = prediction_prob[0][predicted_class]
        command_name = label_encoder.inverse_transform([predicted_class])[0]

        confidence_threshold = 0.4 

        if max_confidence < confidence_threshold:
            logging.info(f"Prediction uncertainty: max confidence {max_confidence:.2f} below threshold")
            return jsonify({"command": "uncertain", "message": "Confidence too low to determine command"}), 200
        else:
            logging.info(f"Predicted command: {command_name} with confidence {max_confidence:.2f}")
            if command_name.lower() == "stop":
                response_message = handle_stop_command()
                return jsonify({"command": command_name, "message": response_message})
            else:
                response_message = execute_command(command_name)
                return jsonify({"command": command_name, "message": response_message})

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500

def handle_stop_command():
    global is_recording

    if is_recording:
        stop_recording()
        return "Recording stopped."
    else:
        return jsonify({
            "command": "stop",
            "message": "Are you sure you want to stop? (y/n)"
        })

def stop_recording():
    global is_recording
    try:
        command = f"taskkill /IM ffmpeg.exe /F"
        subprocess.run(command, shell=True)
        is_recording = False
        logging.info("Stopped screen recording.")
        return "Recording stopped."
    except Exception as e:
        logging.error(f"Error stopping recording: {e}")
        return "Error stopping recording."

def execute_command(command_name):
    if command_name.lower() == "dim":
        try:
            current_brightness = sbc.get_brightness()
            if isinstance(current_brightness, list):
                current_brightness = current_brightness[0]
            current_brightness = int(current_brightness)
            new_brightness = max(current_brightness - 20, 0)  
            sbc.set_brightness(new_brightness)
            return f"Screen brightness dimmed to {new_brightness}%"
        except Exception as e:
            logging.error(f"Error adjusting brightness: {e}")
            return "Failed to adjust screen brightness"
    elif command_name.lower() == "play":
        webbrowser.open("https://www.youtube.com/watch?v=KK2smasHg6w&list=RDQMMrKfe2Uzd-k&start_radio=1") 
        return "Playing video on YouTube"
    elif command_name.lower() == 'mute':
        return "Feature has to be added"
    elif command_name.strip().lower() == "search":
        webbrowser.open("https://www.google.com") 
        return "Opened Google search"
    elif command_name.lower() == "record":
        start_recording()
        return "Screen recording started"
    elif command_name.lower() == 'start':
        webbrowser.open("https://leetcode.com/problemset/")
        return "Leetcode Opened"
    elif command_name.lower() == "shutdown":
        return "Feature will be added soon"
    else:
        logging.warning(f"Command '{command_name}' did not match any defined cases.")
        return "Unknown command"

def start_recording():
    global is_recording
    try:
        recorder_path = 'C:\\Users\\lenovo\\Downloads\\Speech_recog\\ffmpeg-master-latest-win64-gpl\\bin\\ffmpeg.exe'
        command = f'"{recorder_path}" -f gdigrab -i desktop -framerate 30 -video_size 1920x1080 "{output_file}"'
        subprocess.Popen(command, shell=True)
        is_recording = True
        logging.info("Started screen recording.")
    except Exception as e:
        logging.error(f"Error starting recording: {e}")

@app.route("/confirm_stop", methods=["POST"])
def confirm_stop():
    user_input = request.json.get("confirmation")
    if user_input.lower() == 'y':
        pyautogui.hotkey("ctrl", "w")  # Close current window
        return jsonify({"message": "Stopped"})
    else:
        return jsonify({"message": "Stop Cancelled"})

@app.route("/")
def home():
    logging.info("Serving the home page.")
    return send_from_directory('', 'index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico')

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False) 
