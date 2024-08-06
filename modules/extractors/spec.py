import torch
from torchaudio.transforms import Resample, Spectrogram, MelSpectrogram

class SpecExtractor:
    def __init__(self, n_fft, n_mels, hop_length, sample_rate, device = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        
        # self.spectrogram = Spectrogram(self.n_fft, win_length=self.hop_length, hop_length=self.hop_length, center=False, power=1).to(self.device)
        self.spectrogram = MelSpectrogram(sample_rate, self.n_fft, n_mels=n_mels, win_length=self.hop_length, hop_length=self.hop_length, center=False, power=1, mel_scale='slaney').to(self.device)
        
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
    