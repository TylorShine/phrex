import torch
from torchaudio.transforms import Resample, Spectrogram

class SpecExtractor:
    def __init__(self, n_fft, hop_length, sample_rate, device = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        
        self.spectrogram = Spectrogram(self.n_fft, win_length=self.hop_length, hop_length=self.hop_length, power=1).to(self.device)
        
        self.resample_kernel = {}
        
    def extract(self, audio, sample_rate=0):
        # resample
        if sample_rate == self.sample_rate or sample_rate == 0:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, self.sample_rate, lowpass_filter_width = 128).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)
            
        return self.spectrogram(audio_res).transpose(1, 2)
    