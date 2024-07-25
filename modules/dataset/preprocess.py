import os
import torch
import numpy as np

import librosa
from tqdm import tqdm

from modules.extractors import F0Extractor, UnitsEncoder, SpecExtractor


class PreprocessorParameters:
    def __init__(self,
                 data_dir: str,
                 sample_rate: int = 16000,
                 block_size: int = 320,
                 use_f0: bool = True,
                 f0_extractor: str = 'rmvpe',
                 f0_min: float = 35.,
                 f0_max: float = 1600.,
                 units_encoder: str = 'wavlmbase',
                 units_encoder_path: str = 'models/pretrained/wavlm/WavLM-Base+.pt',
                 units_encoder_sample_rate: int = 16000,
                 units_encoder_hop_size: int = 320,
                 units_encoder_skip_frames: int = 0,
                 units_encoder_extract_layers: list[list[int]] = 0,
                 units_encoder_no_alignment: bool = False,
                 spec_n_fft: int = 512,
                 spec_out_channels: int = 128,
                 spec_hop_length: int = 256,
                 device: str | torch.device = 'cpu',
                 ):
        
        self.common = {
            'data_dir': data_dir,
            'sample_rate': sample_rate,
            'block_size': block_size,
            'device': device,
        }
        
        self.units_encoder = {
            'encoder': units_encoder,
            'encoder_ckpt': units_encoder_path,
            'encoder_sample_rate': units_encoder_sample_rate,
            'encoder_hop_size': units_encoder_hop_size,
            'skip_frames': units_encoder_skip_frames,
            'extract_layers': units_encoder_extract_layers,
            'no_alignment': units_encoder_no_alignment,
            'device': device,
        }
        
        self.f0_extractor = None
        
        if use_f0:
            self.f0_extractor = {
                'f0_extractor': f0_extractor,
                'sample_rate': sample_rate,
                'hop_size': block_size,
                'f0_min': f0_min,
                'f0_max': f0_max,
            }
            
        self.spec_extractor = {
            'n_fft': spec_n_fft,
            'hop_length': spec_hop_length,
            'sample_rate': sample_rate,
            'device': device,
        }
        self.spec_out_channels = spec_out_channels
            

PREPROCESSOR_PARAMS: PreprocessorParameters = None

def preprocess_main(root_path: str, dataset: dict[str, dict[str, str]], params: PreprocessorParameters = PREPROCESSOR_PARAMS):
    data_dir = os.path.join(params.common['data_dir'], 'data')
    
    # units
    units_encoder = UnitsEncoder(**params.units_encoder)
    units_dir = os.path.join(params.common['data_dir'], 'units')
    for path in tqdm(dataset.keys(), desc='Extract units'):
        # audio, sr = librosa.load(os.path.join(root_path, path), sr=None)
        audio, sr = librosa.load(os.path.join(root_path, path), sr=params.common['sample_rate'])
        units = units_encoder.encode(
            torch.from_numpy(audio).to(params.common['device'], dtype=torch.float32).unsqueeze(0), sr, params.common['block_size'])
        units_path = os.path.join(units_dir, os.path.relpath(path, start='data'))
        os.makedirs(os.path.dirname(units_path), exist_ok=True)
        np.savez_compressed(f'{units_path}.npz', units=units.squeeze().cpu().numpy())
    del units_encoder
    units_encoder = None
    
    # f0, normalized spectrogram
    if params.f0_extractor is not None:
        f0_extractor = F0Extractor(**params.f0_extractor)
        f0_dir = os.path.join(params.common['data_dir'], 'f0')
        spec_extractor = SpecExtractor(**params.spec_extractor)
        spec_dir = os.path.join(params.common['data_dir'], 'norm_spec')
        
        for path in tqdm(dataset.keys(), desc='Extract f0 and spectrogram'):
            # audio, sr = librosa.load(os.path.join(root_path, path), sr=None)
            audio, sr = librosa.load(os.path.join(root_path, path), sr=params.common['sample_rate'])
            
            f0 = f0_extractor.extract(audio, device=params.common['device'])
            f0_uv = f0 == 0
            spec = spec_extractor.extract(torch.from_numpy(audio).float().to(spec_extractor.device).unsqueeze(0), sample_rate=sr)
            spec = spec.squeeze().cpu().numpy()
            # normalize spec by its max value
            spec = spec / np.max(spec)
            
            f0_norm_spec = spec[:, :params.spec_out_channels]
            
            # f0[f0_uv] = np.random.rand(*f0[f0_uv].shape)*float(params.common['sample_rate']/params.common['block_size']) + float(params.common['sample_rate']/params.common['block_size'])
            f0_path = os.path.join(f0_dir, os.path.relpath(path, start='data'))
            os.makedirs(os.path.dirname(f0_path), exist_ok=True)
            np.savez_compressed(f'{f0_path}.npz', f0=f0)
            
            spec_path = os.path.join(spec_dir, os.path.relpath(path, start='data'))
            os.makedirs(os.path.dirname(spec_path), exist_ok=True)
            np.savez_compressed(f'{spec_path}.npz', norm_spec=f0_norm_spec)
        del f0_extractor
        f0_extractor = None
    
    
if __name__ == '__main__':
    import sys
    from modules.dataset import loader
    
    PREPROCESSOR_PARAMS = PreprocessorParameters(
        data_dir=sys.argv[1],
        units_encoder_extract_layers=[[10, 11]],
        device='cuda')
    
    splits = map(os.path.basename, os.listdir(os.path.join(sys.argv[1], 'data')))
    
    ds_train = loader.get_datasets(os.path.join(sys.argv[1], 'train.csv'))
    ds_test = loader.get_datasets(os.path.join(sys.argv[1], 'test.csv'))
    
    preprocess_main(sys.argv[1], ds_train)
    preprocess_main(sys.argv[1], ds_test)
    
    
    