# Baseline code
- `main.py`: Main file
- `modules`
  - `audio`: Audio module (parsing)
  - `data.py`: Data loader
  - `inference.py`: Inference
  - `metrics.py` : Metrics related to evaluation (CER)
  - `model.py`: Model building (DeepSpeech2)
  - `preprocess.py`: Preprocessing (label/transcripts creation)
  - `trainer.py`: Training related
  - `utils.py` : Miscellaneous settings and necessary functions
  - `vocab.py` : Vocabulary Class file


# Driver & CUDA version
NVIDIA-SMI 535.129.03, Driver Version: 535.129.03, CUDA Version: 12.2


# Install requires
```
pip install astropy==5.1
```
```
pip install torch==2.1.0
```
```
pip install torchaudio==2.1.0
```
```
pip install librosa==0.10.1
```
```
pip install numpy==1.24.3
```
```
pip install pandas==2.0.3
```
```
pip install tqdm==4.65.0
```
```
pip install matplotlib==3.7.2
```
```
pip install scikit-learn==1.3.0
```
```
pip install pydub==0.25.1
```
```
pip install glob2==0.7
```


# How to train with main.py
1. Prepare the dataset.
  -  You can download korean speech datasets from AI-hub (https://www.aihub.or.kr/). After downloading the dataset and unzip the downloaded file.
  - Link to an example dataset: https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=100  
2. Generate the `transcripts.txt` file (run `make-transcripts.py`)
3. Generate the `labels.csv` file based on the data you have prepared (run `make-labels.py`)
4. Generate the `transcipts-final.txt` file in the format provided as an example transcripts-final.txt file (run `make-transcripts-final.py`)
5. Check the sampling rate of the audio files, adjust parameters, and then run `main.py`


# Code reference
This baseline code has been modified based on the open-source project kospeech by Mr. Kim Soo-hwan (https://github.com/sooftware/kospeech).
