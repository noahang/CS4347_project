import os
import librosa
import numpy as np

#Audio Parameters:
SAMPLE_RATE = 44100
CHANNELS = 1

#Analysis Parameters:
PROCESSING_INTERVAL_SECONDS = 6
PROCESSING_INTERVAL_SAMPLES = int(SAMPLE_RATE * PROCESSING_INTERVAL_SECONDS)

#frame length (not used by CQT):
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


def feature_extraction_cqt(AUDIO_DIR : str, OUTPUT_DIR : str) -> None:
    """Extract CQT features from all audio files in AUDIO_DIR and save to OUTPUT_DIR."""
    for filename in os.listdir(AUDIO_DIR):
        
        if not filename.lower().endswith(".mp3"):
            continue
        
        filepath = os.path.join(AUDIO_DIR, filename)
        print(f"Processing {filename}...")

        # Load audio file 
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE)
        total_duration = librosa.get_duration(y=y, sr=sr)

        if sr!=SAMPLE_RATE:
            print(f"  [Warning] Sample rate mismatch in {filename}. Expected {SAMPLE_RATE}, got {sr}. Skipping.")
            continue

        start_samples = np.arange(0, len(y) - PROCESSING_INTERVAL_SAMPLES + 1, PROCESSING_HOP_SAMPLES)


        cqts = []
        for start in start_samples:
            end = start + PROCESSING_INTERVAL_SAMPLES
            chunk_to_process = y[start:end]

            # Compute CQT
            cqt_features_for_one_interval = librosa.cqt(
                        y=chunk_to_process,
                        sr=SAMPLE_RATE,
                        hop_length=HOP_SAMPLES,
                        bins_per_octave= CQT_BINS_PER_OCTAVE,
                        n_bins= CQT_N_BINS, 
                    )

            cqts.append(np.abs(cqt_features_for_one_interval))

        # Stack all windows’ CQTs into one array
        cqts = np.stack(cqts, axis=0)

        # Save resulting CQT array
        out_name = os.path.splitext(filename)[0] + "_cqt.npy"

        #Directory of the folder where the label files are located:
        label_in_dir = os.path.join(AUDIO_DIR, "..", "labels")
        label_path = os.path.join(label_in_dir, os.path.splitext(filename)[0] + ".txt")

        if not os.path.exists(label_path):
            print(f"  [Warning] Label file missing for {filename}, skipping.")
            continue

        with open(label_path, "r") as f:
            label = f.readline().strip()
        os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)
        label_out_dir = os.path.join(OUTPUT_DIR, label)
        np.save(os.path.join(label_out_dir, out_name), cqts)



# Process training data
if __name__ == "__main__":
    AUDIO_TRAIN_DIR = "DATA_audio/Data/train/audio"
    AUDIO_TEST_DIR = "DATA_audio/Data/test/audio"


    interval_str = str(PROCESSING_INTERVAL_SECONDS).replace('.', 'p')
    hop_str = str(PROCESSING_HOP_SECONDS).replace('.', 'p')
    OUTPUT_DIR = "processed_data/"+f"{interval_str}s_interval_{hop_str}s_hop/"
    OUTPUT_TRAIN_DIR = OUTPUT_DIR + "train/"
    OUTPUT_TEST_DIR = OUTPUT_DIR + "test/"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_TRAIN_DIR, exist_ok=True)
    os.makedirs(OUTPUT_TEST_DIR, exist_ok=True)

    print("Extracting CQT features from training data...")
    feature_extraction_cqt(AUDIO_TRAIN_DIR,OUTPUT_TRAIN_DIR)

    print("Extracting CQT features from test data...")
    feature_extraction_cqt(AUDIO_TEST_DIR,OUTPUT_TEST_DIR)  

    print("Done computing CQTs for all files!")