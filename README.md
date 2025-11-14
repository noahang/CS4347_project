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
