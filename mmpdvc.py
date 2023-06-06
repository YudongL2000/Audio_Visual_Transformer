import torch
import torch.nn as nn


def decide_two_stage(transformer_input_type, dt, criterion):
    if transformer_input_type == "gt_proposals":
        two_stage = True
        proposals = dt["gt_boxes"]
        proposals_mask = dt["gt_boxes_mask"]
        criterion.matcher.cost_caption = 0
        for q_k in ["loss_length", "loss_ce", "loss_bbox", "loss_giou"]:
            for key in criterion.weight_dict.keys():
                if q_k in key:
                    criterion.weight_dict[key] = 0
        disable_iterative_refine = True
    elif transformer_input_type == "queries":  #
        two_stage = False
        proposals = None
        proposals_mask = None
        disable_iterative_refine = False
    else:
        raise ValueError("Wrong value of transformer_input_type, got {}".format(transformer_input_type))
    return two_stage, disable_iterative_refine, proposals, proposals_mask



class MMPDVC(nn.Module):
    def __init__(self, base_encoder, transformer, captioner, num_classes, num_queries, num_feature_levels, aux_loss=True, with_box_refine=False, opt_dict=None, translator=None):
        """Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            captioner: captioning head for generate a sentence for each event queries
            num_classes: number of foreground classes
            num_queries: number of event queries. This is the maximal number of events
                         PDVC can detect in a single video. For ActivityNet Captions, we recommend 10-30 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            opt: all configs
        """
        super().__init__()
        self.opt_dict = opt_dict
        self.base_encoder = base_encoder
        self.transformer = transformer
        self.caption_head = captioner

        hidden_dim_encoder = transformer.d_visual_encoder
        self.query_embed = nn.Embedding(num_queries, hidden_dim_encoder * 2)
        self.class_head = nn.Linear(hidden_dim_encoder, num_classes)
        self.count_head = nn.Linear(hidden_dim_encoder, opt_dict["max_eseq_length"] + 1)
        self.bbox_head = MLP(hidden_dim_encoder, hidden_dim_encoder, 2, 3)

        self.num_feature_levels = num_feature_levels
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.share_caption_head = opt_dict["share_caption_head"]

        # initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_head.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_head.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_head.layers[-1].bias.data, 0)

        num_pred = transformer.decoder.num_layers
        if self.share_caption_head:
            print("all decoder layers share the same caption head")
            self.caption_head = nn.ModuleList([self.caption_head for _ in range(num_pred)])
        else:
            print("do NOT share the caption head")
            self.caption_head = _get_clones(self.caption_head, num_pred)

        if with_box_refine:
            self.class_head = _get_clones(self.class_head, num_pred)
            self.count_head = _get_clones(self.count_head, num_pred)
            self.bbox_head = _get_clones(self.bbox_head, num_pred)
            nn.init.constant_(self.bbox_head[0].layers[-1].bias.data[1:], -2)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_head = self.bbox_head
        else:
            nn.init.constant_(self.bbox_head.layers[-1].bias.data[1:], -2)
            self.class_head = nn.ModuleList([self.class_head for _ in range(num_pred)])
            self.count_head = nn.ModuleList([self.count_head for _ in range(num_pred)])
            self.bbox_head = nn.ModuleList([self.bbox_head for _ in range(num_pred)])
            self.transformer.decoder.bbox_head = None

        #self.adaptor = nn.Sequential(nn.Linear(hidden_dim_encoder, 768), nn.LayerNorm(768))
        self.adaptor = nn.Sequential(nn.Linear(hidden_dim_encoder, 50257), nn.LayerNorm(50257))
        self.translator = translator

    def forward(self, dt, criterion, transformer_input_type, eval_mode=False):
        vf = dt["video_tensor"]  # (N, L, C)
        mask = ~dt["video_mask"]  # (N, L)
        duration = dt["video_length"][:, 1]
        N, L, C = vf.shape
        # assert N == 1, "batch size must be 1."
        srcs, masks, pos = self.base_encoder(vf, mask, duration)

        (src_flatten, temporal_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten) = self.transformer.prepare_encoder_inputs(srcs, masks, pos)
        memory = self.transformer.forward_encoder(src_flatten, temporal_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        query_embed = self.query_embed.weight
        proposals_mask = torch.ones(N, query_embed.shape[0], device=query_embed.device).bool()
        init_reference, tgt, reference_points, query_embed = self.transformer.prepare_decoder_input_query(memory, query_embed)
        hs, inter_references = self.transformer.forward_decoder(tgt,reference_points, memory, temporal_shapes, level_start_index, valid_ratios, query_embed, mask_flatten, proposals_mask)
        others = {
            "memory": memory,
            "mask_flatten": mask_flatten,
            "spatial_shapes": temporal_shapes,
            "level_start_index": level_start_index,
            "valid_ratios": valid_ratios,
            "proposals_mask": proposals_mask,
        }
        """
        if eval_mode or self.opt_dict["caption_loss_coef"] == 0:
            out, loss = self.parallel_prediction_full(dt, criterion, hs, init_reference, inter_references, others)
        else:
            out, loss = self.parallel_prediction_matched(dt, criterion, hs, init_reference, inter_references, others)
        """
        out, loss = self.parallel_prediction_matched(dt, criterion, hs, init_reference, inter_references, others)
        return out, loss

    def predict_event_num(self, counter, hs_lid):
        hs_lid_pool = torch.max(hs_lid, dim=1, keepdim=False)[0]  # [bs, feat_dim]
        outputs_class0 = counter(hs_lid_pool)
        return outputs_class0
    
    
    def parallel_prediction_matched(self, dt, criterion, hs, init_reference, inter_references, others):
        outputs_classes = []
        outputs_counts = []
        outputs_coords = []
        outputs_cap_costs = []
        outputs_cap_losses = []
        outputs_cap_probs = []
        outputs_cap_seqs = []

        num_pred = hs.shape[0]
        for l_id in range(num_pred):
            hs_lid = hs[l_id]
            reference = (
                init_reference if l_id == 0 else inter_references[l_id - 1]
            )  # [decoder_layer, batch, query_num, ...]
            outputs_class = self.class_head[l_id](hs_lid)  # [bs, num_query, N_class]
            outputs_count = self.predict_event_num(self.count_head[l_id], hs_lid)
            tmp = self.bbox_head[l_id](hs_lid)  # [bs, num_query, 4]
            cost_caption, loss_caption, cap_probs, seq = self.caption_prediction(self.caption_head[l_id], dt, hs_lid, reference, others, None)
            reference = inverse_sigmoid(reference)
            if reference.shape[-1] == 2:
                tmp += reference
            else:
                assert reference.shape[-1] == 1
                tmp[..., :1] += reference
                outputs_coord = tmp.sigmoid()  # [bs, num_query, 4]

            outputs_classes.append(outputs_class)
            outputs_counts.append(outputs_count)
            outputs_coords.append(outputs_coord)
            # outputs_cap_losses.append(cap_loss)
            outputs_cap_probs.append(cap_probs)
            outputs_cap_seqs.append(seq)

        outputs_class = torch.stack(outputs_classes)  # [decoder_layer, bs, num_query, N_class]
        outputs_count = torch.stack(outputs_counts)
        outputs_coord = torch.stack(outputs_coords)  # [decoder_layer, bs, num_query, 4]
        # outputs_cap_loss = torch.stack(outputs_cap_losses)

        all_out = {
            "pred_logits": outputs_class,
            "pred_count": outputs_count,
            "pred_boxes": outputs_coord,
            # 'caption_losses': outputs_cap_loss,
            "caption_probs": outputs_cap_probs,
            "seq": outputs_cap_seqs,
        }
        out = {k: v[-1] for k, v in all_out.items()}

        if self.aux_loss:
            ks, vs = list(zip(*(all_out.items())))
            out["aux_outputs"] = [{ks[i]: vs[i][j] for i in range(len(ks))} for j in range(num_pred - 1)]
            loss, last_indices, aux_indices = criterion(out, dt["video_target"])
            for l_id in range(hs.shape[0]):
                hs_lid = hs[l_id]
                reference = init_reference if l_id == 0 else inter_references[l_id - 1]
                indices = last_indices[0] if l_id == hs.shape[0] - 1 else aux_indices[l_id][0]
                cap_loss, cap_probs, seq = self.caption_prediction(self.caption_head[l_id], dt, hs_lid, reference, others, indices)
                #dist_loss = self.caption_distillation(cap_probs["cap_state_train"], dt["gpt_state"], dt["gpt_pdvc_mask"])
                #l_dict = {"loss_caption": cap_loss, "loss_distill": dist_loss}
                l_dict = {"loss_caption": cap_loss}
                if l_id != hs.shape[0] - 1:
                    l_dict = {k + f"_{l_id}": v for k, v in l_dict.items()}
                loss.update(l_dict)
            out.update({"caption_probs": cap_probs, "seq": seq})
        else:
            loss, last_indices = criterion(out, dt["video_target"])
            l_id = hs.shape[0] - 1
            reference = inter_references[l_id - 1]  # [decoder_layer, batch, query_num, ...]
            hs_lid = hs[l_id]
            indices = last_indices[0]
            cap_loss, cap_probs, seq = self.caption_prediction(self.caption_head[l_id], dt, hs_lid, reference, others, indices)
            #dist_loss = self.caption_distillation(cap_probs["cap_state_train"], dt["gpt_state"], dt["gpt_pdvc_mask"])
            #l_dict = {"loss_caption": cap_loss, "loss_distill": dist_loss}
            l_dict = {"loss_caption": cap_loss}
            loss.update(l_dict)
            out.pop("caption_losses")
            out.pop("caption_costs")
            out.update({"caption_probs": cap_probs, "seq": seq})
        return out, loss

    def caption_distillation(self, cap_state, target_state, mask):
        prediction = F.normalize(self.adaptor(cap_state), dim=-1)
        target = F.normalize(target_state, dim=-1)
        #print(target.shape, prediction.shape)
        loss = (1 - target @ prediction.permute(0, 2, 1))[mask].mean()
        # loss = ((target[:, :, None] - prediction[:, None]) ** 2).sum(-1)[mask].mean()
        return loss

    def caption_prediction_eval(self, cap_head, dt, hs, reference, others):
        N_, N_q, C = hs.shape
        query_mask = others["proposals_mask"]
        gt_mask = dt["gt_boxes_mask"]
        mix_mask = torch.zeros(query_mask.sum().item(), gt_mask.sum().item())
        query_nums, gt_nums = query_mask.sum(1).cpu(), gt_mask.sum(1).cpu()
        hs_r = torch.masked_select(hs, query_mask.unsqueeze(-1)).reshape(-1, C)
        row_idx, col_idx = 0, 0
        for i in range(N_):
            mix_mask[row_idx : (row_idx + query_nums[i]), col_idx : (col_idx + gt_nums[i])] = 1
            row_idx = row_idx + query_nums[i]
            col_idx = col_idx + gt_nums[i]
        cap_probs = {}
        with torch.no_grad():
            out, seq, cap_prob_eval = cap_head.sample(hs, reference, others)
            if len(seq):
                seq = seq.reshape(-1, N_q, seq.shape[-1])
                cap_prob_eval = cap_prob_eval.reshape(-1, N_q, cap_prob_eval.shape[-1])
            cap_probs["cap_prob_eval"] = cap_prob_eval
            cap_probs["gpt_state"] = out
        return cap_probs, seq
    def caption_prediction(self, cap_head, dt, hs, reference, others, indices=None):
        N_, N_q, C = hs.shape
        all_cap_num = len(dt["cap_tensor"])
        query_mask = others["proposals_mask"]
        gt_mask = dt["gt_boxes_mask"]
        mix_mask = torch.zeros(query_mask.sum().item(), gt_mask.sum().item())
        query_nums, gt_nums = query_mask.sum(1).cpu(), gt_mask.sum(1).cpu()

        hs_r = torch.masked_select(hs, query_mask.unsqueeze(-1)).reshape(-1, C)

        if indices == None:
            row_idx, col_idx = 0, 0
            for i in range(N_):
                mix_mask[row_idx : (row_idx + query_nums[i]), col_idx : (col_idx + gt_nums[i])] = 1
                row_idx = row_idx + query_nums[i]
                col_idx = col_idx + gt_nums[i]

            bigids = mix_mask.nonzero(as_tuple=False)
            feat_bigids, cap_bigids = bigids[:, 0], bigids[:, 1]

        else:
            feat_bigids = torch.zeros(sum([len(_[0]) for _ in indices])).long()
            cap_bigids = torch.zeros_like(feat_bigids)
            total_query_ids = 0
            total_cap_ids = 0
            total_ids = 0
            max_pair_num = max([len(_[0]) for _ in indices])

            new_hr_for_dsa = torch.zeros(N_, max_pair_num, C)  # only for lstm-dsa
            cap_seq = dt["cap_tensor"]
            new_seq_for_dsa = torch.zeros(N_, max_pair_num, cap_seq.shape[-1], dtype=cap_seq.dtype)  # only for lstm-dsa
            for i, index in enumerate(indices):
                feat_ids, cap_ids = index
                feat_bigids[total_ids : total_ids + len(feat_ids)] = total_query_ids + feat_ids
                cap_bigids[total_ids : total_ids + len(feat_ids)] = total_cap_ids + cap_ids
                new_hr_for_dsa[i, : len(feat_ids)] = hs[i, feat_ids]
                new_seq_for_dsa[i, : len(feat_ids)] = cap_seq[total_cap_ids + cap_ids]
                total_query_ids += query_nums[i]
                total_cap_ids += gt_nums[i]
                total_ids += len(feat_ids)
        cap_probs = {}
        flag = True

        """
        if captioner_type == "none":
            cost_caption = torch.zeros(
                N_, N_q, all_cap_num, device=hs.device
            )  # batch_size * num_queries * all_caption_num
            loss_caption = torch.zeros(N_, N_q, all_cap_num, device=hs.device)
            cap_probs["cap_prob_train"] = torch.zeros(1, device=hs.device)
            cap_probs["cap_prob_eval"] = torch.zeros(N_, N_q, 3, device=hs.device)
            seq = torch.zeros(N_, N_q, 3, device=hs.device)
            return cost_caption, loss_caption, cap_probs, seq

        elif captioner_type in ["light"]:
            clip = hs_r.unsqueeze(1)
            clip_mask = clip.new_ones(clip.shape[:2])
            event = None
        """
        # assert N_ == 1, 'only support batchsize = 1'
        if self.training:
            seq = dt["cap_tensor"][cap_bigids]
            if self.opt_dict["caption_cost_type"] != "rl":
                cap_state, cap_prob = cap_head(hs[:, feat_bigids], reference[:, feat_bigids], others, seq)
                cap_probs["cap_state_train"] = cap_state
                cap_probs["cap_prob_train"] = cap_prob
        else:
            with torch.no_grad():
                cap_prob = cap_head(hs[:, feat_bigids], reference[:, feat_bigids], others, dt["cap_tensor"][cap_bigids])
                out, seq, cap_prob_eval = cap_head.sample(hs, reference, others)
                if len(seq):
                    seq = seq.reshape(-1, N_q, seq.shape[-1])
                    cap_prob_eval = cap_prob_eval.reshape(-1, N_q, cap_prob_eval.shape[-1])
                cap_probs["cap_prob_eval"] = cap_prob_eval
                cap_probs["gpt_state"] = out
        """
        flag = False
        pass

        if flag:
            clip_ext = clip[feat_bigids]
            clip_mask_ext = clip_mask[feat_bigids]

            if self.training:
                seq = dt["cap_tensor"][cap_bigids]
                if self.opt.caption_cost_type != "rl":
                    cap_state, cap_prob = cap_head(event, clip_ext, clip_mask_ext, seq)
                    cap_probs["cap_state_train"] = cap_state
                    cap_probs["cap_prob_train"] = cap_prob
            else:
                with torch.no_grad():
                    seq_gt = dt["cap_tensor"][cap_bigids]
                    cap_prob = cap_head(event, clip_ext, clip_mask_ext, seq_gt)
                    seq, cap_prob_eval = cap_head.sample(event, clip, clip_mask)

                    if len(seq):
                        # re_seq = torch.zeros(N_, N_q, seq.shape[-1])
                        # re_cap_prob_eval = torch.zeros(N_, N_q, cap_prob_eval.shape[-1])
                        seq = seq.reshape(-1, N_q, seq.shape[-1])
                        cap_prob_eval = cap_prob_eval.reshape(-1, N_q, cap_prob_eval.shape[-1])
                    cap_probs["cap_prob_eval"] = cap_prob_eval
        """
        if self.opt_dict["caption_cost_type"] == "loss":
            cap_prob = cap_prob.reshape(-1, cap_prob.shape[-2], cap_prob.shape[-1])
            caption_tensor = dt["cap_tensor"][:, 1:][cap_bigids]
            caption_mask = dt["cap_mask"][:, 1:][cap_bigids]
            cap_loss = cap_head.build_loss(cap_prob, caption_tensor, caption_mask)
            cap_cost = cap_loss

        else:
            raise AssertionError("caption cost type error")

        if indices:
            return cap_loss.mean(), cap_probs, seq
        cap_id, query_id = cap_bigids, feat_bigids
        cost_caption = hs_r.new_zeros((max(query_id) + 1, max(cap_id) + 1))
        cost_caption[query_id, cap_id] = cap_cost
        loss_caption = hs_r.new_zeros((max(query_id) + 1, max(cap_id) + 1))
        loss_caption[query_id, cap_id] = cap_loss
        cost_caption = cost_caption.reshape(-1, N_q, max(cap_id) + 1)  # batch_size * num_queries * all_caption_num
        loss_caption = loss_caption.reshape(-1, N_q, max(cap_id) + 1)
        return cost_caption, loss_caption, cap_probs, seq