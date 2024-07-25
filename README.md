# Phrex

Phrex is a PyTorch model for inferring speaker-independent embeddings and pitch from speech audio spectrograms.


## Features

- Extracts speaker-independent content embeddings from mel spectrograms
- Infers pitch (F0) information from audio
- Efficient inference on GPU and CPU


## Usage

```python
import torch
import torchaudio
from .modules.decoder import Phrex, load_model, get_normalized_spectrogram

model, args = load_model('path/to/phrex-128.pt')

# Load audio and compute mel spectrogram
audio, sr = torchaudio.load(audio_path)
spec = get_normalized_spectrogram(args, audio, sr)

# Extract content embeddings and pitch
embeddings = model(spec)
embeddings, pred_f0 = embeddings[:,:,:-1], embeddings[:,:,-1:]
```


## Pre-trained Models
WIP - Coming soon!


## Training Instruction

0. Clone the repo:
```sh
git clone https://github.com/TylorShine/Phrex
```


1. We recommend first installing the PyTorch from the [official website](https://pytorch.org/). then run:

```sh
pip install -r requirements/main.txt
```

We only test the code using python 3.11.9 + cuda 12.1 + torch 2.3.1.


2. Download pre-trained models
- [WavLM Base+](https://github.com/microsoft/unilm/tree/master/wavlm)
    - puts `WavLM-Base+.pt` in `models/pretrained/wavlm` directory
- [RMVPE](https://github.com/yxlllc/RMVPE/releases/)
    - download `rmvpe.zip` and unzip it into `models/pretrained/rmvpe` directory
  

3. Preprocessing

Put all the dataset (audio clips) in `dataset/audio` diectory.

The audio folders need to be named with **positive integers** to represent speaker ids and friendly name separated with a underscore "_", the directory structure is like below:

```sh
# the 1st speaker
dataset/audio/1_first-speaker/aaa.wav
dataset/audio/1_first-speaker/bbb.wav
...
# the 2nd speaker
dataset/audio/2_second-speaker/aaa.wav
dataset/audio/2_second-speaker/bbb.wav
...
```

then run:

```sh
python sortup.py -c configs/combsub-mnp.yaml
```

to divide your datasets to "train" and "test" automatically. If you wanna adjust some parameters, run `python sortup.py -h` to help you. (for now, the "test" datasets will not used)
After that, then run

```sh
python preprocess.py -c configs/combsub-mnp.yaml
```

This will preprocess the audio files and generate spectrograms and F0 that will be used for training.


4. Training

```sh
python train.py -c configs/combsub-mnp.yaml
```

You can safely interrupt training, then running the same command line will resume training.

Checkpoints will be saved periodically to `datasets/exp/` folder.


## License
MIT