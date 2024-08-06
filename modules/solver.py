import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import autocast
from torch.amp import GradScaler
from tqdm import tqdm

from modules.logger.saver import Saver
from modules.logger import utils


def correlation_loss(pred, target, eps=1e-7):
    pred = pred.view(-1)
    target = target.view(-1)

    # Calculate the mean of prediction and target
    pred_mean = torch.mean(pred)
    target_mean = torch.mean(target)

    # Center the data by subtracting the mean
    pred_centered = pred - pred_mean
    target_centered = target - target_mean

    # Calculate the numerator (covariance)
    numerator = torch.sum(pred_centered * target_centered)

    # Calculate the denominator (product of standard deviations)
    denominator = torch.sqrt(torch.sum(pred_centered**2) * torch.sum(target_centered**2))

    # Compute the correlation coefficient
    # Add a small epsilon to avoid division by zero
    correlation = numerator / (denominator + eps)

    # Return the loss as 1 minus the correlation
    # This ensures that minimizing the loss maximizes the correlation
    return 1 - correlation


def correlation_loss_b(pred, target, eps=1e-7):
    B, _, _ = pred.shape
    pred = pred.view(B, -1)
    target = target.view(B, -1)

    # Calculate the mean of prediction and target
    pred_mean = pred.mean(dim=1, keepdim=True)
    target_mean = target.mean(dim=1, keepdim=True)

    # Center the data by subtracting the mean
    pred_centered = pred - pred_mean
    target_centered = target - target_mean

    # Calculate the numerator (covariance)
    # numerator = torch.sum(pred_centered * target_centered)
    numerator = (pred_centered * target_centered).sum(dim=1, keepdim=True)

    # Calculate the denominator (product of standard deviations)
    # denominator = torch.sqrt(torch.sum(pred_centered**2) * torch.sum(target_centered**2))
    denominator = torch.sqrt((pred_centered**2).sum(dim=1, keepdim=True) * (target_centered**2).sum(dim=1, keepdim=True))

    # Compute the correlation coefficient
    # Add a small epsilon to avoid division by zero
    correlation = numerator / (denominator + eps)

    # Return the loss as 1 minus the correlation
    # This ensures that minimizing the loss maximizes the correlation
    return (1 - correlation).mean()


# def contrastive_loss(signal, units, spk_ids, low_similar_loss_variation):
#     B, T, D = signal.shape
    
#     # batching the cosine similarity
#     signal_norm = F.normalize(signal.reshape(B*T, D), dim=1)
#     units_norm = F.normalize(units.reshape(B*T, D), dim=1)
    
#     sim_matrix = torch.mm(signal_norm, units_norm.t()).view(B, T, B, T)
    
#     # make mask based on spk_ids
#     spk_mask = (spk_ids.unsqueeze(1) != spk_ids.unsqueeze(0)).float()
    
#     # choose pos/neg samples
#     pos_sim, _ = sim_matrix.max(dim=3)
#     neg_sim, _ = (sim_matrix + spk_mask.unsqueeze(2).unsqueeze(3) * -1e9).max(dim=1)
    
#     # calculate losses
#     loss_pos = F.l1_loss(1 - pos_sim, torch.zeros_like(pos_sim))
#     loss_neg = F.l1_loss(neg_sim, torch.zeros_like(neg_sim))
    
#     return loss_pos + low_similar_loss_variation * loss_neg

# def contrastive_loss(signal, units, spk_ids, loss_variation, low_similar_loss_variation):
#     B, T, D = signal.shape
    
#     signal_norm = F.normalize(signal.reshape(B*T, D), dim=1)
#     units_norm = F.normalize(units.reshape(B*T, D), dim=1)
    
#     sim_matrix = torch.mm(signal_norm, units_norm.t()).view(B, T, B, T)
    
#     spk_mask = (spk_ids.unsqueeze(1) != spk_ids.unsqueeze(0)).float()
    
#     # Calculate similarity between different speakers
#     diff_spk_sim = sim_matrix * spk_mask.unsqueeze(2).unsqueeze(3)
    
#     # Select the most similar units from different speakers
#     top_k = 3
#     top_sim, _ = diff_spk_sim.reshape(B*T, -1).topk(top_k, dim=1)
    
#     # Select the least similar units from different speakers
#     bottom_k = 3
#     bottom_sim, _ = diff_spk_sim.reshape(B*T, -1).topk(bottom_k, dim=1, largest=False)
    
#     # Calculate weighted center vectors based on similarity
#     weights_top = F.softmax(top_sim, dim=1)
#     target_units_top = torch.sum(weights_top.unsqueeze(2) * units_norm.unsqueeze(0).repeat(B*T, 1, 1), dim=1)
    
#     weights_bottom = F.softmax(-bottom_sim, dim=1)
#     target_units_bottom = torch.sum(weights_bottom.unsqueeze(2) * units_norm.unsqueeze(0).repeat(B*T, 1, 1), dim=1)
    
#     # Maximize similarity with highly similar units
#     loss_high = 1 - F.cosine_similarity(signal_norm, target_units_top, dim=1).mean()
    
#     # Minimize similarity with less similar units
#     loss_low = F.cosine_similarity(signal_norm, target_units_bottom, dim=1).mean()
    
#     total_loss = loss_high + loss_low * low_similar_loss_variation
    
#     return total_loss * loss_variation


def contrastive_loss(signal, units, spk_ids, loss_variation, low_similar_loss_variation):
    B, T, D_signal = signal.shape
    _, T_units, D_units = units.shape
    
    signal_norm = F.normalize(signal, dim=2)
    units_norm = F.normalize(units, dim=2)
    
    loss_high = 0
    loss_low = 0
    
    # choose_num = max(1, int(T//(B*8)))
    # choose_num = min(T, 8)
    
    # indice = random.sample(list(range(T)), choose_num)
    
    # for t in indice:
    for t in range(T):
        current_units = units_norm[:, t, :].unsqueeze(1)
        sim_matrix = torch.bmm(current_units, units_norm.transpose(1, 2))
        
        spk_mask = (spk_ids.unsqueeze(1) != spk_ids.unsqueeze(0)).float().unsqueeze(2)
        
        diff_spk_sim = sim_matrix * spk_mask
        
        top_k = min(3, diff_spk_sim.shape[2] - 1)
        top_sim, top_idx = diff_spk_sim.reshape(B, -1).topk(top_k, dim=1)
        
        bottom_k = min(3, diff_spk_sim.shape[2] - 1)
        bottom_sim, bottom_idx = diff_spk_sim.reshape(B, -1).topk(bottom_k, dim=1, largest=False)
        
        # Compute cosine similarity for signal
        signal_sim_top = F.cosine_similarity(signal_norm[:, t].unsqueeze(1), signal_norm[:, top_idx], dim=2)
        signal_sim_bottom = F.cosine_similarity(signal_norm[:, t].unsqueeze(1), signal_norm[:, bottom_idx], dim=2)
        
        # Compute cosine similarity for units
        units_sim_top = F.cosine_similarity(units_norm[:, t].unsqueeze(1), units_norm[:, top_idx], dim=2)
        units_sim_bottom = F.cosine_similarity(units_norm[:, t].unsqueeze(1), units_norm[:, bottom_idx], dim=2)
        
        # Compute loss
        loss_high += F.l1_loss(signal_sim_top.mean(dim=2), units_sim_top.mean(dim=2))
        loss_low += F.l1_loss(signal_sim_bottom.mean(dim=2), units_sim_bottom.mean(dim=2))
    
    loss_high /= T
    loss_low /= T
    
    total_loss = loss_high + loss_low * low_similar_loss_variation
    
    return total_loss * loss_variation


def contrastive_loss_at_once(signal, units, spk_ids, loss_variation, low_similar_loss_variation):
    B, T, D_signal = signal.shape
    _, T_units, D_units = units.shape

    signal_norm = F.normalize(signal, dim=2)
    units_norm = F.normalize(units, dim=2)

    sim_matrix = torch.bmm(units_norm.reshape(B*T_units, 1, D_units), units_norm.reshape(B*T_units, D_units).transpose(0, 1)).reshape(B, T_units, B, T_units)

    spk_mask = (spk_ids.unsqueeze(1) != spk_ids.unsqueeze(0)).float().unsqueeze(2).unsqueeze(3)

    diff_spk_sim = sim_matrix * spk_mask

    top_k = min(3, diff_spk_sim.shape[-1] - 1)
    top_sim, top_idx = diff_spk_sim.reshape(B, T, -1).topk(top_k, dim=2)

    bottom_k = min(3, diff_spk_sim.shape[-1] - 1)
    bottom_sim, bottom_idx = diff_spk_sim.reshape(B, T, -1).topk(bottom_k, dim=2, largest=False)

    signal_sim = torch.bmm(signal_norm.reshape(B*T, 1, D_signal), signal_norm.reshape(B*T, D_signal).transpose(0, 1)).reshape(B, T, B, T)
    units_sim = torch.bmm(units_norm.reshape(B*T_units, 1, D_units), units_norm.reshape(B*T_units, D_units).transpose(0, 1)).reshape(B, T_units, B, T_units)

    signal_sim_top = signal_sim.reshape(B, T, -1).gather(2, top_idx)
    signal_sim_bottom = signal_sim.reshape(B, T, -1).gather(2, bottom_idx)
    units_sim_top = units_sim.reshape(B, T_units, -1).gather(2, top_idx)
    units_sim_bottom = units_sim.reshape(B, T_units, -1).gather(2, bottom_idx)

    loss_high = F.l1_loss(signal_sim_top, units_sim_top)
    loss_low = F.l1_loss(signal_sim_bottom, units_sim_bottom)

    total_loss = loss_high + loss_low * low_similar_loss_variation

    return total_loss * loss_variation



# def contrastive_loss_at_once(signal, units, spk_ids, loss_variation, low_similar_loss_variation):
#     B, T, D_signal = signal.shape
#     _, _, D_units = units.shape
    
#     signal_norm = F.normalize(signal, dim=2)
#     units_norm = F.normalize(units, dim=2)
    
#     sim_matrix = torch.bmm(units_norm, units_norm.transpose(1, 2))
    
#     spk_mask = (spk_ids != spk_ids.transpose(1, 0)).float().unsqueeze(2).expand(-1, -1, T)
    
#     print(sim_matrix.shape, spk_mask.shape)
    
#     diff_spk_sim = sim_matrix * spk_mask
    
#     top_k = min(3, diff_spk_sim.shape[2] - 1)
#     top_sim, top_idx = diff_spk_sim.reshape(B, T, -1).topk(top_k, dim=2)
    
#     bottom_k = min(3, diff_spk_sim.shape[2] - 1)
#     bottom_sim, bottom_idx = diff_spk_sim.reshape(B, T, -1).topk(bottom_k, dim=2, largest=False)
    
#     # Compute cosine similarity for signal
#     signal_sim_top = F.cosine_similarity(signal_norm.unsqueeze(2), signal_norm.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, -1, D_signal)), dim=3)
#     signal_sim_bottom = F.cosine_similarity(signal_norm.unsqueeze(2), signal_norm.gather(1, bottom_idx.unsqueeze(-1).expand(-1, -1, -1, D_signal)), dim=3)
    
#     # Compute cosine similarity for units
#     units_sim_top = F.cosine_similarity(units_norm.unsqueeze(2), units_norm.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, -1, D_units)), dim=3)
#     units_sim_bottom = F.cosine_similarity(units_norm.unsqueeze(2), units_norm.gather(1, bottom_idx.unsqueeze(-1).expand(-1, -1, -1, D_units)), dim=3)
    
#     # Compute loss
#     loss_high = F.l1_loss(signal_sim_top.mean(dim=2), units_sim_top.mean(dim=2))
#     loss_low = F.l1_loss(signal_sim_bottom.mean(dim=2), units_sim_bottom.mean(dim=2))
    
#     total_loss = loss_high + loss_low * low_similar_loss_variation
    
#     return total_loss * loss_variation


# def test(args, model, loss_func, loader_test, saver, vocoder=None):
#     model.eval()

#     # losses
#     test_loss = 0.
    
#     # intialization
#     num_batches = len(loader_test)
#     rtf_all = []
    
#     spk_id_key = 'spk_id'
#     if args.model.use_speaker_embed:
#         spk_id_key = 'spk_embed'
    
#     # run
#     with torch.no_grad():
#         with tqdm(loader_test, desc="test") as pbar:
#             for data in pbar:
#                 fn = data['name'][0].lstrip("data/test/")

#                 # unpack data
#                 for k in data.keys():
#                     if k != 'name':
#                         data[k] = data[k].to(args.device)
                
#                 units = data['units']
                
#                 # forward
#                 st_time = time.time()
#                 signal = model(units, data['f0'], data['volume'], data[spk_id_key])
#                 ed_time = time.time()

#                 # crop
#                 min_len = np.min([signal.shape[1], data['audio'].shape[1]])
#                 signal        = signal[:,:min_len]
#                 data['audio'] = data['audio'][:,:min_len]

#                 # RTF
#                 run_time = ed_time - st_time
#                 song_time = data['audio'].shape[-1] / args.data.sampling_rate
#                 rtf = run_time / song_time
#                 rtf_all.append(rtf)
            
#                 # loss
#                 loss = loss_func(data['audio'], signal)

#                 test_loss += loss.item()

#                 # log
#                 saver.log_audio({fn+'/gt.wav': data['audio'], fn+'/pred.wav': signal})
                
#                 pbar.set_description(fn)
#                 pbar.set_postfix({'loss': loss.item(), 'RTF': rtf})
            
#     # report
#     test_loss /= num_batches
    
#     return test_loss


def train(args, initial_global_step, nets_g, loader_train, loader_test):
    model, optimizer, scheduler = nets_g
    # saver
    saver = Saver(args, initial_global_step=initial_global_step)
    
    last_model_save_step = saver.global_step
    
    expdir_dirname = os.path.split(args.env.expdir)[-1]
    
    # run
    num_batches = len(loader_train)
    model.train()
    
    scaler = GradScaler('cuda')
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)
    
    
    if args.train.only_f0 == True:
        # freeze out_e layer
        for p in model.out_e.parameters():
            p.requires_grad = False
    elif args.train.only_content_emb == True:
        # freeze except out_e layer
        freeze_layers = [model.conv_in, model.decoder, model.norm, model.out_f]
        for l in freeze_layers:
            for p in l.parameters():
                p.requires_grad = False
    
    
    # model size
    params_count = utils.get_network_params_amount({'model': model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)
        
    # TODO: too slow iteration, could be much faster
    saver.log_info('======= start training =======')
    for epoch in range(args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer.zero_grad()
            
            # unpack data
            for k in data.keys():
                if k != 'name':
                    data[k] = data[k].to(args.device)
                    
            units = data['units']
            norm_spec = data['norm_spec']
            f0 = data['f0']
            f0_prob = data['f0_prob']
                    
            # forward
            if dtype == torch.float32:
                all_signal = model(norm_spec.float())
            else:
                with autocast(device_type=args.device, dtype=dtype):
                    all_signal = model(norm_spec.to(dtype))
                    
            signal, pred_f0 = all_signal[:,:,:args.model.out_channels], all_signal[:,:,args.model.out_channels:]
            
            # optimizer.zero_grad()
            
            if args.train.only_f0 != True:
                
                losses = []
                
                # minibatch contrastive learning
                # Bring per-frame per-speaker's feature to centroid of top two of different speakers' similarities and itself,
                # keep away per-frame per-speaker's feature to centroid of worst two of different speakers' similarities and itself.
                # These are expecting to learn that different speakers' same utterance mapping to nearly vectors.
                B, frames, T = signal.shape
                for b in range(B):
                    # cosine similarity
                    opps = [o for o in range(B)
                            if data['spk_id'][b] != data['spk_id'][o]]
                    if len(opps) <= 0:
                        # TODO: should be quit backpropagation of epoch itself?
                        continue
                    
                    last_hop = 1
                    # f = 0
                    f = torch.randint(0, args.train.frame_hop_random_min-1, (1,))[0]
                    while f < frames - 1:
                        ## brute-force
                        opps_sims = F.cosine_similarity(
                            units.float()[b, f:f+1].unsqueeze(0).repeat(len(opps), units.float().shape[1], 1),
                            units.float()[opps, 0:],
                            dim=2)
                        opps_sort_sim_frame = torch.argsort(opps_sims, dim=1)
                        
                        # sim_mini_opp_frames = [units.float()[b, f]]
                        sim_mini_opp_frames = []
                        sim_maxi_opp_frames = [units.float()[b, f]]
                        
                        for i in range(min(len(opps), 2)):  # TODO: parametrize?
                            opps_large_sims = opps_sims[:, opps_sort_sim_frame[-1-i]].diagonal()
                            opps_small_sims = opps_sims[:, opps_sort_sim_frame[i]].diagonal()
                            
                            sim_mini_opp = torch.argmin(opps_small_sims)
                            sim_mini_opp_frame = opps_sort_sim_frame[i][sim_mini_opp]
                            sim_maxi_opp = torch.argmax(opps_large_sims)
                            sim_maxi_opp_frame = opps_sort_sim_frame[-1-i][sim_maxi_opp]
                            
                            sim_maxi_opp_frames.append(units.float()[opps[sim_maxi_opp], sim_maxi_opp_frame])
                            sim_mini_opp_frames.append(units.float()[opps[sim_mini_opp], sim_mini_opp_frame])
                            
                        sim_mini_opps_centroid = torch.mean(torch.stack(sim_mini_opp_frames), dim=0)
                        sim_maxi_opps_centroid = torch.mean(torch.stack(sim_maxi_opp_frames), dim=0)
                        
                        # ## random pick
                        # rand_frame = torch.randint(0, frames, (len(opps),))
                        # opps_sims = F.cosine_similarity(
                        #     units.float()[b, f].repeat(len(opps), 1),
                        #     # units.float()[opps, 0:],
                        #     torch.stack([units[o, i] for o, i in zip(opps, rand_frame)]).float(),
                        #     dim=1)
                        # opps_sort_sim_batch = torch.argsort(opps_sims, dim=0)
                        
                        # # sim_mini_opp_frames = [units.float()[b, f]]
                        # sim_mini_opp_frames = []
                        # sim_maxi_opp_frames = [units.float()[b, f]]
                        
                        # for i in range(min(len(opps), 2)):  # TODO: parametrize?
                        #     sim_maxi_opp = opps_sort_sim_batch[-1-i]
                        #     sim_maxi_opp_frame = rand_frame[sim_maxi_opp]
                        #     sim_mini_opp = opps_sort_sim_batch[i]
                        #     sim_mini_opp_frame = rand_frame[sim_mini_opp]
                            
                        #     sim_maxi_opp_frames.append(units.float()[opps[sim_maxi_opp], sim_maxi_opp_frame])
                        #     sim_mini_opp_frames.append(units.float()[opps[sim_mini_opp], sim_mini_opp_frame])
                            
                        # sim_mini_opps_centroid = torch.mean(torch.stack(sim_mini_opp_frames), dim=0)
                        # sim_maxi_opps_centroid = torch.mean(torch.stack(sim_maxi_opp_frames), dim=0)
                        
                        if dtype == torch.float32:
                            losses.append(
                                F.l1_loss(
                                    1. - (F.cosine_similarity(signal[b, f], signal[opps[sim_maxi_opp], sim_maxi_opp_frame], dim=0)*0.5 + 0.5),
                                    (1. - (F.cosine_similarity(units.float()[b, f], sim_maxi_opps_centroid, dim=0)*0.5 + 0.5))*args.train.loss_variation)
                            )
                            
                            losses.append(
                                F.l1_loss(
                                    F.cosine_similarity(signal[b, f], signal[opps[sim_mini_opp], sim_mini_opp_frame], dim=0)*0.5 + 0.5,
                                    (F.cosine_similarity(units.float()[b, f], sim_mini_opps_centroid, dim=0)*0.5 + 0.5)*args.train.low_similar_loss_variation)
                            )
                        else:
                            with autocast(device_type=args.device, dtype=dtype):
                                losses.append(
                                    F.l1_loss(
                                        1. - (F.cosine_similarity(signal[b, f], signal[opps[sim_maxi_opp], sim_maxi_opp_frame], dim=0)*0.5 + 0.5),
                                        (1. - (F.cosine_similarity(units[b, f], sim_maxi_opps_centroid, dim=0)*0.5 + 0.5))*args.train.loss_variation)
                                )
                                
                                losses.append(
                                    F.l1_loss(
                                        F.cosine_similarity(signal[b, f], signal[opps[sim_mini_opp], sim_mini_opp_frame], dim=0)*0.5 + 0.5,
                                        (F.cosine_similarity(units[b, f], sim_mini_opps_centroid, dim=0)*0.5 + 0.5)*args.train.low_similar_loss_variation)
                                )
                                
                        last_hop = torch.randint(args.train.frame_hop_random_min, args.train.frame_hop_random_max, (1,))[0]
                        # last_hop = 1
                        f += last_hop
                        
                if len(losses) <= 0:
                    # TODO: should be quit backpropagation of epoch itself?
                    continue
                
                loss = torch.stack([l/(len(losses)/2) for l in losses]).sum()
                
                # # calc for the signal should be convergence to normal distribution
                # signal_std, signal_mean = signal.std(dim=-1), signal.mean(dim=-1)
                # loss = loss + (F.l1_loss(signal_std, torch.ones_like(signal_std)) + signal_mean.abs().mean())*0.5
                
                # # calc loss for pred_f0
                # # loss = loss*0.5 + F.mse_loss(torch.log2(pred_f0 + 1e-3), torch.log2(f0 + 1e-3))
                # # loss = loss*0.5 + F.l1_loss(torch.log2(pred_f0 + 1e-3), torch.log2(f0 + 1e-3))
                # loss = loss*0.5 + (F.huber_loss(torch.log2(pred_f0 + 1e-3), torch.log2(f0 + 1e-3)) + corr_loss(pred_f0, f0))*0.5
                
                # loss_contrastive = contrastive_loss(signal, units.float(), data['spk_id'], args.train.low_similar_loss_variation)
                # loss_contrastive = contrastive_loss(signal, units.float(), data['spk_id'], args.train.loss_variation, args.train.low_similar_loss_variation)
                # loss_contrastive = contrastive_loss_at_once(signal, units.float(), data['spk_id'], args.train.loss_variation, args.train.low_similar_loss_variation)
                
                # calc for the signal should be convergence to normal distribution
                signal_std, signal_mean = signal.std(dim=1), signal.mean(dim=1)
                loss_dist = (F.l1_loss(signal_std, torch.ones_like(signal_std)) + signal_mean.abs().mean())*0.5
            
            # calc loss for pred_f0
            # loss_f0 = (F.huber_loss(torch.log2(pred_f0 + 1e-3), torch.log2(f0 + 1e-3)) + correlation_loss(pred_f0, f0))*0.5
            # loss_f0 = F.huber_loss(torch.log2(pred_f0 + 1e-3), torch.log2(f0 + 1e-3))
            # loss_f0 = F.huber_loss(pred_f0, f0)
            # loss_f0 = (F.l1_loss(torch.log2(pred_f0 + 1.), torch.log2(f0 + 1.)) + correlation_loss(pred_f0, f0))*0.5
            # loss_f0 = F.l1_loss(torch.log2(pred_f0 + 1.), torch.log2(f0 + 1.))
            # loss_f0_resi = F.l1_loss(pred_f0[:, :, 0], f0_prob[:, :, 0])
            
            # loss_f0_resi = F.huber_loss(pred_f0[:, :, 0], f0_prob[:, :, 0])
            loss_f0_resi = F.huber_loss(pred_f0[:, :, -1], f0_prob[:, :, -1])
            # print(pred_f0[:, :, 1:], pred_f0.shape, f0_prob[:, :, 1:], f0_prob.shape)
            # loss_f0_prob = F.binary_cross_entropy(pred_f0[:, :, 1:], f0_prob[:, :, 1:])
            loss_f0_prob = F.binary_cross_entropy(pred_f0[:, :, :-1], f0_prob[:, :, :-1])
            # loss_f0 = F.huber_loss(pred_f0, f0_prob)
            # loss_f0 = F.l1_loss(pred_f0, f0)
            # loss_f0 = F.huber_loss(pred_f0, f0)
            f0_proc = model.freq_table[torch.argmax(pred_f0[:, :, :-1], dim=-1, keepdim=True) + 1] + pred_f0[:, :, -1:]*model.freq_table[-1]

            # loss_f0 = F.l1_loss(torch.log2(f0_proc + 1.), torch.log2(f0 + 1.))
            f0_subh = f0_proc / torch.clamp(f0, min=1.)
            f0_subh = torch.clamp(f0_subh - 0.5, min=0.0)
            f0_subh_resi = torch.fmod(f0_subh, 1.)
            f0_subl = f0 / torch.clamp(f0_proc, min=1.)
            f0_subl = torch.clamp(f0_subl - 1.5, min=0.0)
            f0_subl_resi = torch.fmod(f0_subl, 1.)
            loss_f0_harm = (f0_subh_resi + f0_subl_resi).mean()
            # loss_f0_harm = f0_subh_resi.abs().mean()
            
            if args.train.only_f0 == True:
                loss = (loss_f0_resi + loss_f0_prob)*16. + loss_f0_harm
                # loss = (loss_f0_resi + loss_f0_prob)*16 + loss_f0
                # loss = loss_f0
                # loss = loss_f0 * 10.
            elif args.train.only_content_emb == True:
                # pass
                loss = loss + loss_dist
            else:
                # loss = loss + loss_dist*0.5 + (loss_f0_resi + loss_f0_prob)*16.
                loss = loss + loss_dist + (loss_f0_resi + loss_f0_prob)*16. + loss_f0_harm
                # loss = loss + loss_dist + (loss_f0_resi + loss_f0_prob)*16. + loss_f0
                # loss = loss + loss_dist + loss_f0 * 10.
            
            # handle nan loss
            if torch.isnan(loss):
                raise ValueError(' [x] nan loss ')
            
            # backpropagate
            if dtype == torch.float32:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            # log loss
            if saver.global_step % args.train.interval_log == 0:
                saver.log_info(
                    '\repoch: {} | {:3d}/{:3d} | {} | batch/s: {:2.2f} | loss: {:.7f} | lr: {:.6f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        expdir_dirname,
                        args.train.interval_log/saver.get_interval_time(),
                        loss.item(),
                        scheduler.get_last_lr()[0],
                        saver.get_total_time(),
                        saver.global_step
                    ),
                    end="",
                )
                
                saver.log_value({
                    'train/loss': loss.item(),
                    'train/lr': scheduler.get_last_lr()[0],
                })
                
            # validation
            if saver.global_step % args.train.interval_val == 0:
                optimizer_save = optimizer if args.train.save_opt else None
                
                states = {
                    'scheduler': scheduler.state_dict(),
                    'last_lr': scheduler.get_last_lr(),
                }
                    
                # save latest
                saver.save_model(model, optimizer_save, postfix=f'_{saver.global_step}', states=states)
                
        scheduler.step(loss.item())
                # scheduler.step()
    return

                          
