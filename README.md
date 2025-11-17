# Real-Time Classification of Musical Mode
Classify 84 tonic-mode classes with sounddevice, nnAudio, CQT, CNN + LSTM Model.

Music Information Retrieval (MIR) for tonal analysis has historically focused on offline, major/minor key estimation. 
This project is an attempt to address these limitations by designing and implementing a system for the real-time 
classification of all seven diatonic modes, resulting in 84 distinct tonic-mode classes. (C major, D dorian, F# mixolydian, etc.).

## Setup
To correctly run the real-time implementation of the code, 
Navigate to the CS4347_project folder in the terminal, then to 
create the right environment run:

```bash
conda env create -f SMC_project_environment.yml
conda activate SMC_project_environment
```
then, still from the CS4347_project folder run:

```bash
python "RT_nnAudio_&_UI\app.py"
```
Following this, the output can be seen in the terminal, and using the following link:
http://127.0.0.1:5000/

##  Requirements (can be found in SMC_project_environment.yml)
- python
- pip
- numpy
- scipy
- scikit-learn
- flask
- flask-socketio
- pytorch
- torchvision
- tqdm
- matplotlib
- seaborn
- librosa
- llvmlite
- numba
- audioread
- cython
- soxr
- sounddevice
- soundfile
- nnAudio
