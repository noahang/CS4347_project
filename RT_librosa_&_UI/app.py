import threading
import queue
import sounddevice as sd
import numpy as np
import librosa
import time
import statistics
import random

from flask import Flask, render_template
from flask_socketio import SocketIO
import secrets

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from Model.src.train import Classifier
from Model.src.hparams import Hparams
from Model.src.config.mode_map import NUM_TO_MODE_MAP


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

#Interval between each output of mode-estimate
FINAL_OUTPUT_INTERVAL_SECONDS = 2.0
RESULTS_PER_FINAL_OUTPUT = int(FINAL_OUTPUT_INTERVAL_SECONDS / PROCESSING_HOP_SECONDS)

#Classifier
device = "cpu"

classifier = Classifier(
    device=device,
    model_path="./Model/src/results/best_model2.pth"
)
classifier.model.eval()



def classify_mode_old(feature_data):
    """
    Placeholder function for neural network classification.
    """
    MODES = ["Ionian", "Dorian", "Phrygian", "Lydian", "Mixolydian", "Aeolian", "Locrian"]
    print("Classifier Received features for classification...")
    return random.choice(MODES)

def classify_mode(feature_data: np.ndarray):
    """
    feature_data: numpy array of shape (84, T)
    """

    # Convert numpy → torch
    x = torch.tensor(feature_data, dtype=torch.float32)

    # Model expects (batch, F, T)
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        out = classifier.model(x)           # shape (1, 84)
        probs = torch.softmax(out, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    return NUM_TO_MODE_MAP[pred_class]


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
    It then aggregates these results and finds the statistical mode every X seconds.
    """
    print("Processing thread started, waiting for audio...")
    
    audio_buffer = np.array([], dtype=np.float32)
    mode_results_list = []
    
    while True:
        try:
            data = audio_queue.get(timeout=0.1)
            
            if data is STOP_SIGNAL:
                print("Processing thread received stop signal.")
                break

            audio_buffer = np.concatenate((audio_buffer, data.flatten()))

            #Fast Processing Loop:
            while len(audio_buffer) >= PROCESSING_INTERVAL_SAMPLES:
                
                chunk_to_process = audio_buffer[:PROCESSING_INTERVAL_SAMPLES]

                cqt_features = librosa.cqt(
                    y=chunk_to_process,
                    sr=SAMPLE_RATE,
                    hop_length=HOP_SAMPLES,
                    bins_per_octave= CQT_BINS_PER_OCTAVE,
                    n_bins= CQT_N_BINS, 
                )
                
                print("CQT shape real-time:", np.abs(cqt_features).shape)
                estimated_mode = classify_mode(np.abs(cqt_features))
                mode_results_list.append(estimated_mode)
                print(f"   > Intermediate mode: {estimated_mode}")
                
                audio_buffer = audio_buffer[PROCESSING_HOP_SAMPLES:]

            #Aggregation loop:
            if len(mode_results_list) >= RESULTS_PER_FINAL_OUTPUT:
                
                try:
                    final_mode = statistics.mode(mode_results_list)
                    print("\n" + "-"*30)
                    print(f"   FINAL MODE (last {FINAL_OUTPUT_INTERVAL_SECONDS}s): {final_mode}")
                    print("-"*30 + "\n")

                    socketio.emit('update_mode', {'data': final_mode})
                    
                except statistics.StatisticsError:
                    print(f"No unique mode found, defaulting to '{mode_results_list[0]}'")
                
                mode_results_list.clear()

        except queue.Empty:
            continue
            
    print("Processing thread finished.")


def audio_processing_task():
    print("Main: Starting the processing thread...")
    processor = threading.Thread(target=processing_thread_task)
    processor.daemon = True
    processor.start()

   #Warm-up analysis functions:
    print("Main: Warming up analysis functions...")
    try:
        dummy_audio = np.zeros(N_FFT * 4, dtype=np.float32) 
        _ =  librosa.cqt(
                y=dummy_audio,
                sr=SAMPLE_RATE,
                hop_length=HOP_SAMPLES,
                bins_per_octave= CQT_BINS_PER_OCTAVE,
                n_bins= CQT_N_BINS, 
            )
        print("Main: Warm-up complete.")
    except Exception as e:
        print(f"Error during warm-up: {e}")
   


    print("Main: Starting audio stream...")
    try:
        with sd.InputStream(callback=audio_callback,
                             channels=CHANNELS,
                             samplerate=SAMPLE_RATE, 
                             blocksize=CALLBACK_BLOCKSIZE,
                             dtype='float32'):
            
            print("\n" + "-"*30)
            print("   Audio is being streamed and processed.")
            print("   Web server is running. Open http://127.0.0.1:5000")
            print("-"*30 + "\n")
            
            while True:
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

    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)