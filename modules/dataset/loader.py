import os
import random
import torch
import numpy as np

import csv

import librosa

from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset


def get_datasets(data_csv: str):
    if not os.path.isfile(data_csv):
        raise FileNotFoundError(f'metadata csv not found: {data_csv}')
    
    with open(data_csv, encoding='utf-8') as f:
        reader = csv.reader(f)
        lines = [row for row in reader]
        headers = lines[0]
        datas = {}
        for line in lines[1:]:
            datas[line[0]] = {}
            for h, v in enumerate(line[1:], start=1):
                datas[line[0]][headers[h]] = v
    return datas


class AudioDataset(TorchDataset):
    def __init__(
        self,
        root_path,
        metadatas: dict,
        crop_duration,
        hop_size,
        sampling_rate,
        whole_audio=False,
        cache_all_data=True,
        device='cpu',
        fp16=False,
        use_aug=False,
        use_spk_embed=False,
        per_file_spk_embed=False,
        use_mel=False,
        units_only=False,
    ):
        super().__init__()
        
        self.root_path = root_path
        self.crop_duration = crop_duration
        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.whole_audio = whole_audio
        self.use_aug = use_aug
        self.use_mel = use_mel,
        self.use_spk_embed = use_spk_embed
        self.per_file_spk_embed = per_file_spk_embed
        
        self.units_only = units_only
        
        self.paths = list(metadatas.keys())
        
        self.data_buffer = {}
        self.spk_embeds = {}
        
        skip_index = []
        # if units_only:
        data_dir = os.path.join(root_path, 'data')
        for file in tqdm(metadatas.keys(), desc='loading data'):
            audio_path = os.path.join(root_path, file)
            duration = librosa.get_duration(path=audio_path, sr=sampling_rate)
            file_dir, file_name = os.path.split(file)
            # file_rel = os.path.relpath(file_dir, start=data_dir)
            file_rel = os.path.relpath(file_dir, start='data')
            
            if cache_all_data:
                # load units
                units_dir = os.path.join(self.root_path, 'units', file_rel, file_name) + '.npz'
                units = np.load(units_dir)['units']
                units = torch.from_numpy(units).to(device)
                
                # load norm_spec
                spec_dir = os.path.join(self.root_path, 'norm_spec', file_rel, file_name) + '.npz'
                spec = np.load(spec_dir)['norm_spec']
                spec = torch.from_numpy(spec).to(device)
            
            # # load augumented norm_spec
            # spec_aug_dir = os.path.join(self.root_path, 'norm_spec_aug', file_rel, file_name) + '.npz'
            # spec_aug = np.load(spec_aug_dir)['norm_spec_aug']
            # spec_aug = torch.from_numpy(spec_aug).to(device)
            
            # load f0
            f0_path = os.path.join(self.root_path, 'f0', file_rel, file_name) + '.npz'
            f0 = np.load(f0_path)['f0']
            f0 = torch.from_numpy(f0).float().unsqueeze(-1).to(device)
            
            # # load augumented f0 ratio
            # f0_aug_path = os.path.join(self.root_path, 'f0_aug', file_rel, file_name) + '.npz'
            # f0_aug = np.load(f0_aug_path)['f0_aug']
            # f0_aug = torch.from_numpy(f0_aug).float().to(device)
            
            self.data_buffer[file] = {
                'duration': duration,
                'f0': f0,
                'spk_id': torch.LongTensor(np.array([int(metadatas[file]['spk_id'])])).to(device),
            }
            
            if cache_all_data:
                if fp16:
                    units = units.half()
                    spec = spec.half()
                self.data_buffer[file]['units'] = units
                self.data_buffer[file]['norm_spec'] = spec
                    
        if len(skip_index) > 0:
            print(f"skip {len(skip_index)} files.")
            self.paths = [v for i, v in enumerate(self.paths) if i not in skip_index]
                    
    def __getitem__(self, file_idx):
        file = self.paths[file_idx]
        data_buffer = self.data_buffer[file]
        
        # # check duration. if too short, then skip
        if data_buffer['duration'] < (self.crop_duration + 0.1):
            return self.__getitem__( (file_idx + 1) % len(self.paths))
            
        # get item
        return self.get_data(file, data_buffer)
    
    def __len__(self):
        return len(self.paths)
    
    def get_data(self, file, data_buffer):
        name = os.path.splitext(file)[0]
        frame_resolution = self.hop_size / self.sampling_rate
        duration = data_buffer['duration']
        crop_duration = duration if self.whole_audio else self.crop_duration
        
        idx_from = 0 if self.whole_audio else random.uniform(0, duration - crop_duration - 0.1)
        start_frame = int(idx_from / frame_resolution)
        units_frame_len = int(crop_duration / frame_resolution)
        
        aug_flag = random.choice([True, False]) and self.use_aug
        file_dir, file_name = os.path.split(file)
        file_rel = os.path.relpath(file_dir, start='data')
        
        # load units
        if 'units' in data_buffer.keys():
            units = data_buffer['units'][start_frame : start_frame + units_frame_len]
        else:
            units_dir = os.path.join(self.root_path, 'units', file_rel, file_name) + '.npz'
            units = np.load(units_dir)['units'][start_frame : start_frame + units_frame_len]
            units = torch.from_numpy(units).to(self.device)
            
        # load spk_id
        spk_id = data_buffer['spk_id']
        
        # load f0
        f0 = data_buffer['f0'][start_frame : start_frame + units_frame_len]
        
        norm_spec_key = "norm_spec"
        
        # load spec
        if norm_spec_key in data_buffer.keys():
            spec = data_buffer[norm_spec_key][start_frame : start_frame + units_frame_len]
        else:
            spec_dir = os.path.join(self.root_path, norm_spec_key, file_rel, file_name) + '.npz'
            spec = np.load(spec_dir)[norm_spec_key][start_frame : start_frame + units_frame_len]
            spec = torch.from_numpy(spec).to(self.device)
            
        if aug_flag:
            # calc bin shift amount from random ratio
            bin_shift_ratio = np.random.uniform(0.8, 1.5)
            bin_shift_ratio_int = int(int(bin_shift_ratio - 1) + 1*np.sign(bin_shift_ratio))
            bin_shift_ratio_resi = np.modf(bin_shift_ratio)[0]
            
            # shift spec bins
            # rolls = int(spec.shape[-1] * bin_shift_ratio)
            spec_roll = torch.roll(spec, bin_shift_ratio_int, dims=-1)
            spec_roll[:, 0] = spec[:, 0]
            spec_roll[:, -1] = spec[:, -1]
            
            # lerp bins
            spec = spec * (1 - bin_shift_ratio_resi) + spec_roll * bin_shift_ratio_resi
            
            # recalc f0
            f0 = f0 * bin_shift_ratio
        
        return dict(spk_id=spk_id, units=units, norm_spec=spec, f0=f0)    
      
        
class AudioCrop:
    def __init__(self, block_size, sampling_rate, crop_duration):
        self.block_size = block_size
        self.sampling_rate = sampling_rate
        self.crop_duration = crop_duration
        
    def crop_audio(self, batch):
        frame_resolution = self.block_size / self.sampling_rate
        units_frame_len = int(self.crop_duration / frame_resolution)
        # print(batch['units'].shape, batch['audio'].shape)
        # print(batch['units'][0][0].shape, len(batch['units']), len(batch['units'][0]))
        # print(len(batch['units']), len(batch['units'][0]), len(batch['units'][0][0]))
        for b in range(len(batch['audio'])):
            duration = len(batch['audio'][b]) / self.sampling_rate
            idx_from = random.uniform(0, duration - self.crop_duration - 0.1)
            start_frame = int(idx_from / frame_resolution)
        
            batch['units'][b] = batch['units'][b][0][start_frame:start_frame+units_frame_len]
            batch['f0'][b] = batch['f0'][b][start_frame:start_frame+units_frame_len]
            batch['volume'][b] = batch['volume'][b][start_frame:start_frame+units_frame_len]
            
            batch['audio'][b] = batch['audio'][b][start_frame*self.block_size:(start_frame + units_frame_len)*self.block_size]
            
        for b in range(len(batch['audio'])):
            batch['units'] = torch.tensor(batch['units'])
            batch['f0'] = torch.tensor(batch['f0'])
            batch['volume'] = torch.tensor(batch['volume'])
            batch['audio'] = torch.tensor(batch['audio'])
            batch['spk_embed'] = torch.tensor(batch['spk_embed'])
            batch['spk_id'] = torch.tensor(batch['spk_id'])
        
        return batch


def get_data_loaders(args):
    loaders = {}
    
    ds_train = get_datasets(os.path.join(args.data.dataset_path, 'train.csv'))
    
    loaders['train'] = DataLoader(
        AudioDataset(
            root_path=args.data.dataset_path,
            metadatas=ds_train,
            crop_duration=args.data.duration,
            hop_size=args.data.block_size,
            sampling_rate=args.data.sampling_rate,
            whole_audio=False,
            device=args.train.cache_device,
            fp16=args.train.cache_fp16,
            use_aug=True,
            use_spk_embed=args.model.use_speaker_embed,
            use_mel="Diffusion" in args.model.type or "Reflow" in args.model.type,
            units_only=args.train.only_u2c_stack),
        batch_size=args.train.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.train.num_workers if args.train.cache_device=='cpu' else 0,
        persistent_workers=(args.train.num_workers > 0) if args.train.cache_device=='cpu' else False,
        pin_memory=True if args.train.cache_device=='cpu' else False
    )
    
    test_csv = os.path.join(args.data.dataset_path, 'test.csv')
    if os.path.isfile(test_csv):
        ds_test = get_datasets(test_csv)
        
        loaders['test'] = DataLoader(
            AudioDataset(
                root_path=args.data.dataset_path,
                metadatas=ds_test,
                crop_duration=args.data.duration,
                hop_size=args.data.block_size,
                sampling_rate=args.data.sampling_rate,
                whole_audio=True,
                device=args.train.cache_device,
                use_aug=False,
                use_spk_embed=args.model.use_speaker_embed,
                use_mel="Diffusion" in args.model.type or "Reflow" in args.model.type,
                units_only=args.train.only_u2c_stack),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True if args.train.cache_device=='cpu' else False
        )
    else:
        loaders['test'] = None
    
    return loaders


if __name__ == '__main__':
    import os, sys
    import numpy as np
    
    ds = get_datasets(sys.argv[1], name=os.path.basename(sys.argv[2]), data_dir=sys.argv[2], cache_dir=sys.argv[3])
        
    print(ds)
    print(ds['train'][0].keys())
    