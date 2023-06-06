from tensorboardX import SummaryWriter
from collections import OrderedDict
# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os
import sys
import time
from os.path import dirname, abspath

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc
import subprocess
from tensorboardX import SummaryWriter
from collections import OrderedDict
from mmpdvc import MMPDVC
from visual_Encoders import BaseEncoder
from text_encoders import *
from post_processes import *
from model_utils.GPT_utils import *



param_dict = {}
#run info
param_dict["id"] = "trial"

#data source
param_dict["train_caption_file"] = "drive/MyDrive/Dataset/ActivityNet_processed/captiondata/train_modified.json"
param_dict["visual_feature_type"] = "i3d_flow"
param_dict["val_caption_file"] = "drive/MyDrive/Dataset/ActivityNet_processed/captiondata/val_1.json"
param_dict["visual_feature_folder"] = "drive/MyDrive/Dataset/ActivityNet_processed/i3d_25fps_stack64step64_2stream_npy"
param_dict["audio_feature_folder"] = "drive/MyDrive/Dataset/ActivityNet_processed/vggish_npy"
param_dict["dict_file"] = "drive/MyDrive/Dataset/ActivityNet_processed/vocabulary_activitynet.json"
param_dict["invalid_video_json"] = []


#preprocess 
param_dict["data_rescale"] = 1
param_dict["feature_sample_rate"] = 1
param_dict["vocab_size"] = 5747
param_dict["frame_embedding_num"] = 100
param_dict["num_classes"] = 1
param_dict["data_norm"] = 0
param_dict["max_caption_len"] = 30
param_dict["train_proposal_sample_num"] = 24
param_dict["gt_proposal_sample_num"] = 10
param_dict["num_queries"] = 100
param_dict["batch_size"] = 1
param_dict["batch_size_for_eval"] = 1
param_dict["nthreads"] = 1
param_dict["num_feature_levels"] = 4
param_dict["input_encoding_size"] = 512


#Deformable Transformer
param_dict["visual_feature_dim"] = 1024
param_dict["audio_feature_dim"] = 128
param_dict["visual_hidden_dim"] = 512
param_dict["audio_hidden_dim"] = 128
param_dict["feature_dim"] = 512
param_dict["nhead"] = 8
param_dict["num_encoder_layers"] = 6
param_dict["num_decoder_layers"] = 6
param_dict["dim_feedforward"] = 1024
param_dict["dropout"] = 0.1
param_dict["activation"] = "relu"

#Text Decoder params (Deform LSTM)
param_dict["rnn_size"] = 512
param_dict["num_layers"] = 1
param_dict["drop_prob"] = 0.5
param_dict["att_hid_size"] = 512
param_dict["clip_context_dim"] = param_dict["visual_hidden_dim"]
param_dict["cap_nheads"] = 8
param_dict["hidden_dim"] = param_dict["visual_hidden_dim"]
param_dict["wordRNN_input_feats_type"] = "C"
param_dict["cap_num_feature_levels"] = 4
param_dict["cap_dec_n_points"] = 4
param_dict["num_feature_levels"] = 4
param_dict['vocab_size'] = 5747

#Caption params 
param_dict["max_eseq_length"] = 10
param_dict["share_caption_head"] = 1

#Matcher and Loss hyperparameters
param_dict["set_cost_class"] = 1
param_dict["set_cost_bbox"] = 5
param_dict["set_cost_giou"] = 2
param_dict["cost_alpha"] = 0.25
param_dict["cost_gamma"] = 2
param_dict['lloss_gau_mask'] = 1
param_dict['lloss_beta'] = 1

#Loss weights
param_dict["cls_loss_coef"] = 2
param_dict["bbox_loss_coef"] = 5
param_dict["giou_loss_coef"] = 2
param_dict["count_loss_coef"] = 0
param_dict["caption_loss_coef"] = 0
param_dict["distill_loss_coef"] = 1

#focal param
param_dict["focal_alpha"] = 0.25
param_dict["focal_gamma"] = 2.0

#training_hyperparam
param_dict["epochs"] = 50
param_dict["epoch"] = 30
param_dict["device"] = "cuda"
param_dict["start_from_mode"] = 'last' # 'best'
param_dict["lr"] = 1e-4
param_dict['optimizer_type'] = "adam" # "adamW"
param_dict['weight_decay'] = 0
param_dict["learning_rate_decay_start"] = 8
param_dict["learning_rate_decay_every"] = 3
param_dict["learning_rate_decay_rate"] = 0.5
param_dict['scheduled_sampling_start'] = -1
param_dict['disable_tqdm'] = True
param_dict['debug'] = True
param_dict['transformer_input_type'] = "queries" #["gt_proposals", "learnt_proposals", "queries"]
param_dict['caption_cost_type'] = 'loss'
param_dict['caption_decoder_type'] ="standard"
param_dict["grad_clip"] = 100


def train(param_dict):
    losses = ["labels", "boxes", "cardinality"]
    matcher = HungarianMatcher(cost_class=param_dict["set_cost_class"], cost_bbox=param_dict["set_cost_bbox"], cost_giou=param_dict["set_cost_giou"], cost_alpha=param_dict["cost_alpha"], cost_gamma=param_dict["cost_gamma"])
    saved_info = {"best": {}, "last": {}, "history": {}, "eval_history": {}}


    train_dataset = PropSeqDataset(param_dict['train_caption_file'], param_dict['visual_feature_folder'], param_dict['dict_file'], True, "gt", param_dict)
    val_dataset = PropSeqDataset(param_dict['train_caption_file'], param_dict['visual_feature_folder'], param_dict['dict_file'], False, "gt", param_dict)
    train_loader = DataLoader(
        train_dataset, batch_size=param_dict['batch_size'], shuffle=True, num_workers=param_dict['nthreads'], collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=param_dict['batch_size_for_eval'], shuffle=False, num_workers=param_dict['nthreads'], collate_fn=collate_fn
    )

    epoch = saved_info[param_dict['start_from_mode'][:4]].get("epoch", 0)
    iteration = saved_info[param_dict['start_from_mode'][:4]].get("iter", 0)
    best_val_score = saved_info[param_dict['start_from_mode'][:4]].get("best_val_score", -1e5)
    val_result_history = saved_info["history"].get("val_result_history", {})
    loss_history = saved_info["history"].get("loss_history", {})
    lr_history = saved_info["history"].get("lr_history", {})
    param_dict['current_lr'] = param_dict['lr']

    # Build model
    visual_base_encoder = BaseEncoder(param_dict["num_feature_levels"], 
                                      param_dict["visual_feature_dim"], 
                                      param_dict["visual_hidden_dim"]
                                      )
    deformable_transformer = MultimodalDeformableTransformer(param_dict["visual_hidden_dim"], 
                                                             param_dict["nhead"], 
                                                             param_dict["num_encoder_layers"], 
                                                             param_dict["num_decoder_layers"], 
                                                             param_dict["dim_feedforward"], 
                                                             dropout = 0.1, activation="relu", 
                                                             return_intermediate_dec=True, 
                                                             dec_n_points=4,enc_n_points=4
                                                             )
    caption_embed = LSTMDSACaptioner(param_dict)
    model = MMPDVC(visual_base_encoder, deformable_transformer, 
                            caption_embed, param_dict["num_classes"], 
                            param_dict["num_queries"], 
                            param_dict["num_feature_levels"], 
                            True, False, opt_dict=param_dict, translator=None
                   )
    weight_dict = {'loss_ce': param_dict["cls_loss_coef"], 
                   'loss_bbox': param_dict["bbox_loss_coef"], 
                   'loss_giou': param_dict["giou_loss_coef"], 
                   'loss_counter': param_dict["count_loss_coef"], 
                   'loss_caption': param_dict["caption_loss_coef"]
                   }
    criterion = SetCriterion(
        param_dict["num_classes"],
        matcher,
        weight_dict,
        losses,
        focal_alpha=param_dict["focal_alpha"],
        focal_gamma=param_dict["focal_gamma"],
        opt=param_dict,
        ) 
    post_processor =  PostProcess(param_dict)
    model.translator = train_dataset.translator
    model.train()
    model.to(param_dict['device'])

    # Build teacher
    gpt_state_dict = torch.load("/content/drive/MyDrive/Dataset/ActivityNet_processed/GPT/gpt2-pytorch_model.bin", map_location="cpu")
    teacher_encoder = get_encoder_text()
    teacher = GPT2LMHeadModel(GPT2Config())
    teacher = load_weight_GPT(teacher, gpt_state_dict)
    teacher.eval()
    teacher.to(param_dict['device'])

    if param_dict['optimizer_type'] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=param_dict['lr'], weight_decay=param_dict['weight_decay'])

    elif param_dict['optimizer_type'] == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=param_dict['lr'], weight_decay=param_dict['weight_decay'])

    milestone = [
        param_dict['learning_rate_decay_start'] + param_dict['learning_rate_decay_every'] * _
        for _ in range(int((param_dict['epoch'] - param_dict['learning_rate_decay_start']) / param_dict['learning_rate_decay_every']))
    ]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestone, gamma=param_dict['learning_rate_decay_rate'])

    # Start training
    print("Start training !")

    start = time.time()

    weight_dict = criterion.weight_dict

    # Epoch-level iteration
    for e in range(param_dict['epoch']):
        if True:
            # scheduled sampling rate update
            if epoch > param_dict['scheduled_sampling_start'] >= 0:
                frac = (epoch - param_dict['scheduled_sampling_start']) // param_dict['scheduled_sampling_increase_every']
                param_dict['ss_prob'] = min(
                    param_dict['basic_ss_prob'] + param_dict['scheduled_sampling_increase_prob'] * frac, param_dict['scheduled_sampling_max_prob']
                )
                model.caption_head.ss_prob = param_dict['ss_prob']

            print("lr:{}".format(float(param_dict['current_lr'])))
            pass

        # Batch-level iteration
        iter = 0
        for dt in tqdm(train_loader, disable=param_dict['disable_tqdm']):
            p = subprocess.check_output('nvidia-smi')
            ram_using = re.findall(r'\b\d+MiB+ /', str(p))
            print(f'iter: {iter} / {len(train_loader)}')
            print(ram_using)

            if param_dict['device'] == "cuda":
                torch.cuda.synchronize(param_dict['device'])
  
            optimizer.zero_grad()
            dt = {key: _.to(param_dict['device']) if isinstance(_, torch.Tensor) else _ for key, _ in dt.items()}
            dt["video_target"] = [
                {key: _.to(param_dict['device']) if isinstance(_, torch.Tensor) else _ for key, _ in vid_info.items()}
                for vid_info in dt["video_target"]
            ]
            

            with torch.no_grad():
                context = torch.full([len(dt["cap_tensor"]), 100], 50256, device=param_dict['device'], dtype=torch.long)
                gpt_indices = []
                pdvc_indices = []
                for i, c in enumerate(dt["cap_raw"][0]):
                    gpt_index = [0]
                    pdvc_index = [1]
                    for x in c.split():
                        gpt_index += [len(teacher_encoder.encode(x))]
                        pdvc_index += [pdvc_index[-1] + 2] if "-" in x else [pdvc_index[-1] + 1]
                    gpt_indices += [torch.tensor(gpt_index, device=param_dict['device'], dtype=torch.long).cumsum(0)]
                    pdvc_indices += [torch.tensor(pdvc_index, device=param_dict['device'], dtype=torch.long)]
                    tokens = torch.tensor(teacher_encoder.encode(c), dtype=torch.long, device=param_dict['device'])
                    context[i, : len(tokens)] = tokens
                states, tmp_info = teacher(context, past=None)
                #dt["gpt_state"] = states
  

                dt["gpt_pdvc_mask"] = torch.zeros(
                    dt["cap_tensor"].shape[0], 100, dt["cap_tensor"].shape[1] - 1, device=param_dict['device'], dtype=torch.bool
                )
                for i, (gi, pi) in enumerate(zip(gpt_indices, pdvc_indices)):
                    for j in range(len(gi) - 1):
                        dt["gpt_pdvc_mask"][i, gi[j] : gi[j + 1], pi[j] : pi[j + 1]] = True
            dt = collections.defaultdict(lambda: None, dt)

            output, loss = model(dt, criterion, param_dict['transformer_input_type'])

            final_loss = sum(loss[k] * weight_dict[k] for k in loss.keys() if k in weight_dict)
 
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), param_dict['grad_clip'])

            optimizer.step()


            iter += 1
            del(pdvc_indices)
            del(gpt_indices)
            del(tokens)
            del(states)
            del(context)
            del(tmp_info)
            loss_dict_list = list(loss.keys())
            for k in loss_dict_list:
              del(loss[k])
            loss.clear()
            del(loss)
            del(final_loss)
            dt_k_list = list(dt.keys())
            for k in dt_k_list:
                del(dt[k])
            dt.clear()
            del(dt)
            output_key_list = list(output.keys())
            for k in output_key_list:
                del(output[k])
            output.clear()
            del(output)
            gc.collect()