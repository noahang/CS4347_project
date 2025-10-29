import threading
import queue
import sounddevice as sd
import numpy as np
import librosa
import time
import statistics
import random

#Audio and Queue Parameters:
SAMPLE_RATE = 44100
CHANNELS = 1
CALLBACK_BLOCKSIZE = 1024
audio_queue = queue.Queue(maxsize=10)
STOP_SIGNAL = None 

#Analysis Parameters:
PROCESSING_INTERVAL_SECONDS = 2
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
PROCESSING_HOP_SECONDS = 0.5
PROCESSING_HOP_SAMPLES = int(SAMPLE_RATE * PROCESSING_HOP_SECONDS)

#Interval between each output of mode-estimate
FINAL_OUTPUT_INTERVAL_SECONDS = 2.0
RESULTS_PER_FINAL_OUTPUT = int(FINAL_OUTPUT_INTERVAL_SECONDS / PROCESSING_HOP_SECONDS)



def classify_mode(feature_data):
    """
    Placeholder function for neural network classification.
    
    """

    # Classifier Parameters:
    MODES = ["Ionian", "Dorian", "Phrygian", "Lydian", "Mixolydian", "Aeolian", "Locrian"]
    print("  > Classifier Received features for classification...")
       
    # Return a random value for demonstration purposes
    return random.choice(MODES)


#Thread 1: Audio Callback (High-Priority):

def audio_callback(indata, frames, time, status):
    """
    High priority audio callback function.
    """
    if status:
        print(f"Audio status: {status}")
        
    try:
        # Put a copy of the data into the queue without waiting
        audio_queue.put_nowait(indata.copy())
    except queue.Full:
        # If the queue is full, the frame will be dropped
        print("Warning: Dropping audio frame, the processing-thread task takes too long.")


#Thread 2: Processing Thread (Normal-Priority)

def processing_thread_task():
    """
    Thread runs analysis on an overlapping sliding window.
    
    It then aggregates these results and finds the statistical mode every 2 seconds.
    """
    print("Processing thread started, waiting for audio...")
    
    audio_buffer = np.array([], dtype=np.float32)
    mode_results_list = []
    
    while True:
        try:
            # Get audio data from the queue
            data = audio_queue.get(timeout=1)
            
            if data is STOP_SIGNAL:
                print("Processing thread received stop signal.")
                break

            audio_buffer = np.concatenate((audio_buffer, data.flatten()))

            # --- 1. Fast Analysis Loop (Sliding Window) ---
            while len(audio_buffer) >= PROCESSING_INTERVAL_SAMPLES:
                
                chunk_to_process = audio_buffer[:PROCESSING_INTERVAL_SAMPLES]
                
                # ...
                # Start timing the feature calculation
                start_time = time.perf_counter()
                
                # 1. Calculate features
                cqt_features = librosa.cqt(
                    y=chunk_to_process,
                    sr=SAMPLE_RATE,
                    hop_length=HOP_SAMPLES,
                    bins_per_octave= CQT_BINS_PER_OCTAVE,
                    n_bins= CQT_N_BINS, 
                )
                end_time = time.perf_counter()
                print(f"  > Features calculated in {end_time - start_time:.3f} seconds.")
                
            
                # 2. Call the classifier
                estimated_mode = classify_mode(cqt_features)
                
                # 3. Add the result
                mode_results_list.append(estimated_mode)
                print(f"  > Intermediate mode: {estimated_mode}")
                
                # ...
                
                # Discard only the processing hop
                audio_buffer = audio_buffer[PROCESSING_HOP_SAMPLES:]

            #Slow Aggregation Loop:
            
            if len(mode_results_list) >= RESULTS_PER_FINAL_OUTPUT:
                
                try:
                    final_mode = statistics.mode(mode_results_list)
                    print("\n" + "-"*30)
                    print(f"  FINAL MODE (last {FINAL_OUTPUT_INTERVAL_SECONDS}s): {final_mode}")
                    print("-"*30 + "\n")
                    
                except statistics.StatisticsError:
                    print(f"No unique mode found, defaulting to '{mode_results_list[0]}'")
                
                mode_results_list.clear()

        except queue.Empty:
            continue
            
    print("Processing thread finished.")


#Thread 3: Main Thread (Setup & UI):

def main():
    print("Main: Starting the processing thread...")
    processor = threading.Thread(target=processing_thread_task)
    processor.start()
    #Warm-up Section:
    """Warm up analysis functions to avoid initial latency."""

    print("Main: Warming up analysis functions (this may take a moment)...")
    try:
        # 1. Warm up Librosa (CQT):
        #dummy buffer:
        dummy_audio = np.zeros(N_FFT * 4, dtype=np.float32) 
        _ =  librosa.cqt(
                    y=dummy_audio,
                    sr=SAMPLE_RATE,
                    hop_length=HOP_SAMPLES,
                    bins_per_octave= CQT_BINS_PER_OCTAVE,
                    n_bins= CQT_N_BINS, 
                )
        
        # 2. Warm up the Classifier:

        #Insert classifier warm-up code here if needed:

        print("Main: Warm-up complete.")
        
    except Exception as e:
        print(f"Error during warm-up: {e}")
    # --- END WARM-UP SECTION ---

    print("Main: Starting audio stream...")
    try:
        with sd.InputStream(callback=audio_callback,
                            channels=CHANNELS,
                            samplerate=SAMPLE_RATE, 
                            blocksize=CALLBACK_BLOCKSIZE,
                            dtype='float32'):
            
            print("\n" + "-"*30)
            print("  Audio is being streamed and processed.")
            print("  Press Enter to stop...")
            print("-"*30 + "\n")
            input() #Waits for user to press Enter


    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        print("\nMain: Stopping threads...")
        audio_queue.put(STOP_SIGNAL)
        processor.join()
        
        print("Main: All done.")

if __name__ == "__main__":
    main()