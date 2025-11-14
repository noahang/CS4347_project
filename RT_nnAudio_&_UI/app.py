import threading
import queue
import sounddevice as sd
import numpy as np
import librosa
import time
import statistics
import random

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from nnAudio.Spectrogram import CQT

from flask import Flask, render_template
from flask_socketio import SocketIO
import secrets

from Model2.src.train import Classifier
from Model2.src.hparams import Hparams
from Model2.src.config.mode_map import NUM_TO_TONAL_CENTER_MAP, NUM_TO_MUSICAL_MODE_MAP

device = "cpu" 
classifier = Classifier(
    device=device, 
    model_path="./Model/src/results/best_model3.pth"
)
classifier.model.eval()


app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
socketio = SocketIO(app)

#Audio and Queue Parameters:
SAMPLE_RATE = 44100
CHANNELS = 1
CALLBACK_BLOCKSIZE = 1024
audio_queue = queue.Queue(maxsize=100)
STOP_SIGNAL = None 

#Analysis Parameters:
PROCESSING_INTERVAL_SECONDS = 6
PROCESSING_INTERVAL_SAMPLES = int(SAMPLE_RATE * PROCESSING_INTERVAL_SECONDS)

#frame length:
FRAME_MS = 20.0
FRAME_SAMPLES = int(SAMPLE_RATE * (FRAME_MS / 1000.0))

#hop-length:
HOP_MS = 10.0
HOP_SAMPLES = int(SAMPLE_RATE * (HOP_MS / 1000.0))

#CQT Parameters:
N_FFT = 1024
CQT_OCTAVES = 7
CQT_BINS_PER_OCTAVE = 12
CQT_N_BINS = CQT_OCTAVES * CQT_BINS_PER_OCTAVE

#Sliding Window Parameters:
PROCESSING_HOP_SECONDS = 1
PROCESSING_HOP_SAMPLES = int(SAMPLE_RATE * PROCESSING_HOP_SECONDS)

lstm_state = None

def classify_mode_1_head(feature_data: torch.tensor):
    # 1. Add a batch dimension (B, C, H, W) -> (1, 1, 84, 601)
    #Our model expects a batch, but we are processing one chunk at a time.
    x = feature_data.unsqueeze(0).to(device)

    #Calculate with no_grad for faster computation
    with torch.no_grad():
        out = classifier.model(x)
        probs = torch.softmax(out, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return 

def classify_mode_2_heads(feature_data: torch.tensor):
    global lstm_state
    # 1. Add a batch dimension (B, C, H, W) -> (1, 1, 84, 601)
    #Our model expects a batch, but we are processing one chunk at a time.
    x = feature_data.unsqueeze(0).to(device)

    #Calculate with no_grad for faster computation
    with torch.no_grad():
        #center, mode = classifier.model(x)
        (center, mode), lstm_state = classifier.model(x, lstm_state)
        probs_center = torch.softmax(center, dim=1)
        probs_mode = torch.softmax(mode, dim=1)
        pred_center = torch.argmax(probs_center, dim=1).item()
        pred_mode = torch.argmax(probs_mode, dim=1).item()

        print(probs_center)

    return NUM_TO_TONAL_CENTER_MAP[pred_center]+ " " + NUM_TO_MUSICAL_MODE_MAP[pred_mode]

#Thread 1: Audio Callback (High-Priority):
def audio_callback(indata, frames, time, status):
    """
    High priority audio callback function.
    """
    if status:
        print(f"Audio status: {status}")
        
    try:
        audio_queue.put_nowait(indata.copy())
    except queue.Full:
        print("Warning: Dropping audio frame, the processing-thread task takes too long.")

#Thread 2: Processing Thread (Normal-Priority)
def processing_thread_task():
    """
    Thread runs analysis on overlapping sliding windows.
    the size and hop-length of the overlapping windows are decided by 
    PROCESSING_INTERVAL_SECONDS and PROCESSING_HOP_SECONDS
    """
    print("Processing thread started, waiting for audio...")
    
    #Main audio-buffer used by the processing thread
    audio_buffer = np.array([], dtype=np.float32)

    print("Initializing nnAudio CQT layer...")
    FMIN = librosa.note_to_hz('C1') 
    
    #initalizing the 1D-convolutional layer to calculate CQTs.
    cqt_layer = CQT(
        sr=SAMPLE_RATE,
        hop_length=HOP_SAMPLES,
        fmin=FMIN,
        n_bins=CQT_N_BINS,
        bins_per_octave=CQT_BINS_PER_OCTAVE
    )
    
    while True:
        try:
            # Get next audio block from producer thread
            data = audio_queue.get(timeout=0.5)
            
            if data is STOP_SIGNAL:
                print("Processing thread received stop signal.")
                break

            audio_buffer = np.concatenate((audio_buffer, data.flatten()))

            #Fast Processing Loop:
            while len(audio_buffer) >= PROCESSING_INTERVAL_SAMPLES:
                 # Take the next "processing-interval"- seconds of samples:
                audio_chunk = audio_buffer[:PROCESSING_INTERVAL_SAMPLES]
                

                # nnAudio CQT computation:
                # Convert to tensor and add batch dimension: [1, num_samples]
                # (nnAudio expects input shape [batch, time])
                chunk_tensor = torch.tensor(audio_chunk, dtype=torch.float32).unsqueeze(0)
                cqt_features_tensor = cqt_layer(chunk_tensor)
                # Convert from complex to magnitude features, and remove batch dimension
                magnitude_features_tensor = torch.abs(cqt_features_tensor).squeeze(0)
            

                #Classification
                # Run the neural network classifier on the CQT magnitude features
                estimated_mode = classify_mode_2_heads(magnitude_features_tensor)
                print(f"Mode-estimate: {estimated_mode}")

                # Send the estimated mode to the frontend via WebSocket
                socketio.emit('update_mode', {'data': estimated_mode})

                # Slide the buffer forward by the hop-size:
                audio_buffer = audio_buffer[PROCESSING_HOP_SAMPLES:]

        except queue.Empty:
            # No audio arrived within timeout. Loop continues without blocking
            continue
            
    print("Processing thread finished.")


def audio_processing_task():
    print("Main: Starting the processing thread...")
    processor = threading.Thread(target=processing_thread_task)
    processor.daemon = True

    
    # Start the processing-thread that consumes audio and runs the neural network
    processor.start()

    
    print("Main: Starting audio stream...")
    try:
        # Open the microphone stream; audio_callback() is invoked for each block
        with sd.InputStream(callback=audio_callback,
                             channels=CHANNELS,
                             samplerate=SAMPLE_RATE, 
                             blocksize=CALLBACK_BLOCKSIZE,
                             dtype='float32'):
            
            print("\n" + "_"*40)
            print("   Audio is being streamed and processed.")
            print("   Web server running. Open http://127.0.0.1:5000")
            print("_"*40 + "\n")
            
            while True:
                # Keeps the stream alive
                time.sleep(10)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        print("\nMain: Stopping threads...")
        audio_queue.put(STOP_SIGNAL)
        print("Main: Audio processing done.")

@app.route('/')
def mode_viewer():
    """Serve the main HTML page."""
    return render_template('mode_viewer.html')


if __name__ == "__main__":
    print("Main: Starting audio processing...")
    audio_thread = threading.Thread(target=audio_processing_task)
    audio_thread.daemon = True
    audio_thread.start()
    
    print("Main: Starting Flask-SocketIO server...")
    # Launch the web server (SocketIO enables real-time push updates to the browser)
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)