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
        f0_out_channels,
        f0_min,
        f0_max,
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
        self.f0_out_channels = f0_out_channels
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.f0_step = f0_max/f0_out_channels
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
        # cnt = 0
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
            f0_np = np.load(f0_path)['f0']
            f0 = torch.from_numpy(f0_np).float().unsqueeze(-1).to(device)
            
            # make linear f0 prob.
            # f0_hot_ch = np.clip(f0_np, 0.0, f0_max) / self.f0_step
            # f0_hot_ch_resi, f0_hot_ch_int = np.modf(f0_hot_ch)
            # f0_hot_ch_int = f0_hot_ch_int.astype(np.int32)
            f0_hot_ch = torch.clamp(f0.squeeze(-1), max=self.f0_max - 1e-3) / self.f0_step
            f0_hot_ch_resi = torch.frac(f0_hot_ch)
            f0_hot_ch_int = f0_hot_ch - f0_hot_ch_resi
            f0_hot_ch_int = f0_hot_ch_int.long()
            # f0_prob = torch.zeros(f0.shape[0], f0_out_channels, device=device)
            f0_prob = torch.nn.functional.one_hot(f0_hot_ch_int, num_classes=f0_out_channels).to(f0)
            # f0_prob[:, 0] = f0_hot_ch_resi
            # f0_prob = torch.ones(f0.shape[0], f0_out_channels, device=device)*(torch.arange(f0_out_channels)[None, :].repeat(f0.shape[0], 1) < f0_hot_ch[None, :]).to(device)
            
            # f0_prob = torch.ones(f0.shape[0], f0_out_channels, device=device)
            # f0_prob = f0_prob * (torch.arange(f0_out_channels, 0, -1, device=device)[None, :].long() < f0_hot_ch[:, None]).to(device)
            # f0_prob[f0_hot_ch_int] = f0_hot_ch_resi
            
            # # f0_prob = torch.clamp(torch.arange(f0_out_channels, 0, -1, device=device)[None, :] - (f0.squeeze(-1)/(self.f0_max/f0_out_channels))[:, None], min=0., max=1.)
            # f0_prob = torch.where((torch.arange(f0_out_channels, 0, -1, device=device)[None, :] - (f0.squeeze(-1)/(self.f0_max/f0_out_channels))[:, None]) < 1., 0., 1.)
            
            # f0_prob[f0_hot_ch_int] = f0_hot_ch_resi
            # f0_prob[:, 0] = torch.from_numpy(f0_hot_ch_resi).float().to(device)
            f0_prob[:, -1] = f0_hot_ch_resi
            # print(f0_hot_ch_int, f0_hot_ch_resi, f0_prob[45:55], f0_hot_ch_int.shape)
            
            # # load augumented f0 ratio
            # f0_aug_path = os.path.join(self.root_path, 'f0_aug', file_rel, file_name) + '.npz'
            # f0_aug = np.load(f0_aug_path)['f0_aug']
            # f0_aug = torch.from_numpy(f0_aug).float().to(device)
            
            self.data_buffer[file] = {
                'duration': duration,
                'f0': f0,
                'f0_prob': f0_prob,
                'spk_id': torch.LongTensor(np.array([int(metadatas[file]['spk_id'])])).to(device),
            }
            
            if cache_all_data:
                if fp16:
                    units = units.half()
                    spec = spec.half()
                self.data_buffer[file]['units'] = units
                self.data_buffer[file]['norm_spec'] = spec
                
            # import sys
            # sys.exit()
            # cnt += 1
            # if cnt > 48:
            #     self.paths = self.paths[:cnt]
            #     break
                    
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
        # aug_flag = True
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
        
        # load f0 prob.
        f0_prob = data_buffer['f0_prob'][start_frame : start_frame + units_frame_len]
        
        norm_spec_key = "norm_spec"
        
        # load spec
        if norm_spec_key in data_buffer.keys():
            spec = data_buffer[norm_spec_key][start_frame : start_frame + units_frame_len]
        else:
            spec_dir = os.path.join(self.root_path, norm_spec_key, file_rel, file_name) + '.npz'
            spec = np.load(spec_dir)[norm_spec_key][start_frame : start_frame + units_frame_len]
            spec = torch.from_numpy(spec).to(self.device)
            
        # if aug_flag == True:
        #     # calc bin shift amount from random ratio
        #     bin_shift_ratio = np.random.uniform(-0.01, 0.03)
            
        #     # shift spec bins
        #     rolls = spec.shape[-1] * bin_shift_ratio
        #     rolls_int = int(rolls)
        #     # rolls_resi = abs(rolls - rolls_int)
        #     spec_roll = torch.roll(spec, rolls_int, dims=-1)
        #     spec_roll[:, 0] = spec[:, 0]
        #     spec_roll[:, -1] = spec[:, -1]
            
        #     # spec_roll_resi = torch.roll(spec, int(rolls_int+np.sign(bin_shift_ratio)), dims=-1)
        #     # spec_roll_resi[:, 0] = spec[:, 0]
        #     # spec_roll_resi[:, -1] = spec[:, -1]
            
        #     # # lerp bins
        #     # spec = spec_roll * (1 - rolls_resi) + spec_roll_resi * rolls_resi
            
        #     # recalc actual f0 shift from roll amount
        #     f0_uv = f0 == 0
        #     # f0[~f0_uv] = torch.clamp(f0[~f0_uv] + 16000.*0.5*bin_shift_ratio, min=0.)
        #     f0[~f0_uv] = torch.clamp(f0[~f0_uv] + 16000./spec.shape[-1]*0.5*rolls_int, min=0.)
        
        # if aug_flag == True:
        #     # calc bin shift amount from random ratio
        #     bin_shift_ratio = np.random.uniform(-0.02, 0.02)
            
        #     # shift spec bins
        #     rolls = spec.shape[-1] * bin_shift_ratio
        #     rolls_int = int(rolls)
        #     spec_roll = torch.roll(spec, rolls_int, dims=-1)
        #     spec_roll[:, 0] = spec[:, 0]
        #     spec_roll[:, -1] = spec[:, -1]
            
        #     # Apply smoothing to reduce artifacts
        #     kernel_size = 3
        #     smoothing_kernel = torch.ones(1, 1, kernel_size) / kernel_size
        #     spec_roll = torch.nn.functional.conv1d(spec_roll.unsqueeze(1), smoothing_kernel.to(spec_roll), padding=kernel_size//2).squeeze(1)
            
        #     # recalc actual f0 shift from roll amount
        #     f0_uv = f0 == 0
        #     f0_shift = 16000. / spec.shape[-1] * 0.5 * rolls_int
        #     f0[~f0_uv] = torch.clamp(f0[~f0_uv] + f0_shift, min=0.)
            
        #     # Apply f0 smoothing
        #     f0_smoothed = torch.nn.functional.conv1d(f0.unsqueeze(1), smoothing_kernel.to(f0), padding=kernel_size//2).squeeze(1)
        #     f0[~f0_uv] = f0_smoothed[~f0_uv]
            
        #     spec = spec_roll            
        
        if aug_flag == True:
        # if False:
            # get f0 center per batch
            with torch.no_grad():
                # f0_center = torch.median(f0).nan_to_num().unsqueeze(0)
                # f0_center = torch.median(f0).unsqueeze(0).cpu().detach().numpy()
                # print(f0.shape)
                # f0_center = torch.median(f0).cpu().numpy()
                f0_clamp = torch.clamp(f0, min=35.)
                f0_min, f0_max = torch.min(f0_clamp).cpu().numpy(), torch.max(f0_clamp).cpu().numpy()
                # print(f0_center)
                # f0_center = f0_center.unsqueeze(0).cpu().detach().numpy()
                # f0_scaled_min, f0_scaled_max = 35., 1600.
                f0_scaled_min, f0_scaled_max = self.f0_min, self.f0_max
                # freq width of a bin
                bin_width = self.sampling_rate * 0.5 / spec.shape[-1]
                
                # f0_center_num = f0_center / bin_width
                f0_min_width = f0_min - f0_scaled_min
                # f0_min_width = 300. # force
                f0_max_width = f0_scaled_max - f0_max
                
                f0_width = f0_max_width + f0_min_width
                
                # print(bin_width)
                
                # print(f0_center_num, spec.shape)

                f0_uv = f0 < self.f0_min
                
                # calc min/max shift factor from f0 center and its min/max
                # f0_shift_min, f0_shift_max = (f0_scaled_min / bin_width), (f0_scaled_max / bin_width)
                # calc random shift factor from min/max
                # shift_amount_freq = torch.rand_like(f0_center) * (f0_shift_max - f0_shift_min) - f0_center + f0_scaled_min
                # shift_random_offset = torch.rand_like(f0_center) * (f0_shift_max - f0_shift_min) + f0_shift_min
                # shift_random_offset = np.random.rand() ** 2. * f0_width - f0_min_width
                shift_random_offset = np.random.rand() * f0_width - f0_min_width
                # shift_amount_freq = np.random.rand() * (f0_shift_max - f0_shift_min) - f0_center + f0_scaled_min
                # calc shift amount from shift_amount_freq and bin_width
                shift_amount_num = shift_random_offset / bin_width
                
                # # calc min/max shift factor from f0 center and its min/max
                # f0_shift_min, f0_shift_max = (f0_scaled_min - f0_center), (f0_scaled_max - f0_center)
                # # calc random shift factor from min/max
                # shift_amount_freq = torch.rand_like(f0_center) * (f0_shift_max - f0_shift_min) - f0_center + f0_scaled_min
                # # shift_amount_freq = np.random.rand() * (f0_shift_max - f0_shift_min) - f0_center + f0_scaled_min
                # # calc shift amount from shift_amount_freq and bin_width
                # shift_amount_bins = shift_amount_freq / bin_width
                
                # print(f0.shape)
                
                # # calc bin shift amount from random ratio
                # bin_shift_ratio = np.random.uniform(-0.15, 0.15)
                
                # # shift spec bins using linear interpolation
                # rolls = spec.shape[-1] * bin_shift_ratio
                
                # # spec_shifted = torch.zeros_like(spec)
                # spec_shifted = torch.rand_like(spec) * spec.min(dim=1, keepdim=True).values
                # for i in range(spec.shape[-1]):
                #     pos = i - shift_amount_bins
                #     # pos_floor = int(torch.floor(pos))
                #     # pos_ceil = int(torch.ceil(pos))
                #     pos_floor = torch.floor(pos)
                #     pos_ceil = torch.ceil(pos)
                #     weight_ceil = pos - pos_floor
                #     weight_floor = 1 - weight_ceil
                    
                #     pos_floor_bounded = 0 <= pos_floor < spec.shape[-1]
                #     pos_ceil_bounded = 0 <= pos_ceil < spec.shape[-1]
                    
                #     pos_floor = int(pos_floor.clamp(min=0., max=spec.shape[-1]-1).cpu().numpy())
                #     pos_ceil = int(pos_ceil.clamp(min=0., max=spec.shape[-1]-1).cpu().numpy())
                    
                #     spec_shifted[:, i] += weight_floor * spec[:, pos_floor] * pos_floor_bounded
                #     spec_shifted[:, i] += weight_ceil * spec[:, pos_ceil] * pos_ceil_bounded
                    
                #     # if 0 <= pos_floor < spec.shape[-1]:
                #     #     spec_shifted[:, i] += weight_floor * spec[:, pos_floor]
                #     # if 0 <= pos_ceil < spec.shape[-1]:
                #     #     spec_shifted[:, i] += weight_ceil * spec[:, pos_ceil]
                
                # shift_amount_bins_int = int(torch.floor(shift_amount_bins).cpu().numpy())
                # shift_amount_bins_int = int(torch.floor(shift_amount_num).cpu().numpy())
                # shift_amount_bins_int = torch.floor(shift_amount_num).to(torch.int64)
                shift_amount_bins_int = int(np.floor(shift_amount_num).astype(np.int64))
                spec_shifted = spec.roll(shifts=shift_amount_bins_int, dims=-1)
                # spec_shifted = torch.roll(spec, shift_amount_bins_int, dims=-1)
                spec_shifted[f0_uv[:, 0]] = spec[f0_uv[:, 0]]
                
                if shift_amount_bins_int > 0:
                    spec_shifted[:, :shift_amount_bins_int] = torch.rand_like(spec_shifted[:, :shift_amount_bins_int]) * spec.abs().min(dim=1, keepdim=True).values
                if shift_amount_bins_int < 0:
                    spec_shifted[:, spec_shifted.shape[1] + shift_amount_bins_int:] = torch.rand_like(spec_shifted[:, spec_shifted.shape[1] + shift_amount_bins_int:]) * spec.abs().min(dim=1, keepdim=True).values
                # spec_shifted[:, 0] = spec[:, 0]
                
                # # Apply smoothing to reduce artifacts
                # kernel_size = 3
                # smoothing_kernel = torch.ones(1, 1, kernel_size) / kernel_size
                # spec_shifted = torch.nn.functional.conv1d(spec_shifted.unsqueeze(1), smoothing_kernel.to(spec_shifted), padding=kernel_size//2).squeeze(1)
                
                # recalc actual f0 shift from roll amount
                # f0_uv = f0 == 0
                # f0_uv = f0 < self.f0_min
                # f0_shift = 16000. / spec.shape[-1] * 0.5 * rolls
                # f0[~f0_uv] = torch.clamp(f0[~f0_uv] + f0_shift, min=0.)
                # f0[~f0_uv] = torch.clamp(f0[~f0_uv] + shift_amount_freq, min=0.)
                # f0[~f0_uv] = torch.clamp(f0[~f0_uv] + shift_amount_bins_int*bin_width/self.sampling_rate, min=0.)
                f0[~f0_uv] = torch.clamp(f0[~f0_uv] + shift_amount_bins_int*bin_width, min=0.)
                
                # # Apply f0 smoothing
                # f0_smoothed = torch.nn.functional.conv1d(f0.unsqueeze(1), smoothing_kernel.to(f0), padding=kernel_size//2).squeeze(1)
                # f0[~f0_uv] = f0_smoothed[~f0_uv]
                
                # recalc linear f0 prob.
                f0_hot_ch = torch.clamp(f0.squeeze(-1), max=self.f0_max - 1e-3) / self.f0_step
                f0_hot_ch_resi = torch.frac(f0_hot_ch)
                f0_hot_ch_int = f0_hot_ch - f0_hot_ch_resi
                f0_hot_ch_int = f0_hot_ch_int.to(torch.int64)
                f0_prob = torch.zeros(f0.shape[0], self.f0_out_channels).to(f0)
                # # f0_prob[f0_hot_ch_int] = 1.0
                f0_prob = torch.nn.functional.one_hot(f0_hot_ch_int, num_classes=self.f0_out_channels).to(f0)
                # f0_prob[:, 0] = f0_hot_ch_resi
                # f0_prob = torch.ones(f0.shape[0], self.f0_out_channels).to(f0) * (torch.arange(self.f0_out_channels) < f0_hot_ch).to(f0)
                # f0_prob[:, f0_hot_ch] = f0_hot_ch_resi
                
                # f0_prob = torch.ones(f0.shape[0], self.f0_out_channels).to(f0)
                # f0_prob = f0_prob * (torch.arange(self.f0_out_channels).to(f0)[None, :].repeat(f0.shape[0], 1) < f0_hot_ch[:, None]).to(f0)
                # f0_prob[:, f0_hot_ch.to(torch.int64)] = f0_hot_ch_resi
                
                # f0_prob = torch.clamp(torch.arange(self.f0_out_channels, 0, -1).to(f0)[None, :] - (f0.squeeze(-1)/(self.f0_max/self.f0_out_channels))[:, None], min=0., max=1.)
                
                # f0_prob = torch.where((torch.arange(self.f0_out_channels, 0, -1).to(f0)[None, :] - (f0.squeeze(-1)/(self.f0_max/self.f0_out_channels))[:, None]) < 1., 0., 1.)
                f0_prob[:, -1] = f0_hot_ch_resi
                
                # print(f0[~f0_uv], f0_hot_ch_int, f0_hot_ch_resi)
                
                spec = spec_shifted
        
        return dict(spk_id=spk_id, units=units, norm_spec=spec, f0=f0, f0_prob=f0_prob)
        # return dict(spk_id=spk_id, units=units, norm_spec=spec, f0=f0)
      
        
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
            f0_out_channels=args.model.f0_out_channels,
            f0_min=args.data.f0_min,
            f0_max=args.data.f0_max,
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
                f0_out_channels=args.model.f0_out_channels,
                f0_min=args.data.f0_min,
                f0_max=args.data.f0_max,
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
    