

class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    def __init__(self, param_dict):
        super().__init__()
        self.param_dict = param_dict
    @torch.no_grad()
    def forward(self, outputs, target_sizes, loader):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size] containing the size of each video of the batch
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
        N, N_q, N_class = out_logits.shape
        assert len(out_logits) == len(target_sizes)

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), N_q, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_cl_to_xy(out_bbox)
        raw_boxes = copy.deepcopy(boxes)
        boxes[boxes < 0] = 0
        boxes[boxes > 1] = 1
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 2))

        scale_fct = torch.stack([target_sizes, target_sizes], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        seq = outputs["seq"]  # [batch_size, num_queries, max_Cap_len=30]
        cap_prob = outputs["caption_probs"]["cap_prob_eval"]  # [batch_size, num_queries]
        eseq_lens = outputs["pred_count"].argmax(dim=-1).clamp(min=1)

        if len(seq):
            mask = (seq > 0).float()
            # cap_scores = (mask * cap_prob).sum(2).cpu().numpy().astype('float') / (
            #         1e-5 + mask.sum(2).cpu().numpy().astype('float'))
            cap_scores = (mask * cap_prob).sum(2).cpu().numpy().astype("float")
            seq = seq.detach().cpu().numpy().astype("int")  # (eseq_batch_size, eseq_len, cap_len)
            caps = [[loader.dataset.translator.rtranslate(s) for s in s_vid] for s_vid in seq]
            caps = [[caps[batch][idx] for q_id, idx in enumerate(b)] for batch, b in enumerate(topk_boxes)]
            cap_scores = [[cap_scores[batch, idx] for q_id, idx in enumerate(b)] for batch, b in enumerate(topk_boxes)]
        else:
            bs, num_queries = boxes.shape[:2]
            cap_scores = [[-1e5] * num_queries] * bs
            caps = [[""] * num_queries] * bs

        results = [
            {
                "scores": s,
                "labels": l,
                "boxes": b,
                "raw_boxes": b,
                "captions": c,
                "caption_scores": cs,
                "query_id": qid,
                "vid_duration": ts,
                "pred_seq_len": sl,
            }
            for s, l, b, rb, c, cs, qid, ts, sl in zip(
                scores, labels, boxes, raw_boxes, caps, cap_scores, topk_boxes, target_sizes, eseq_lens
            )
        ]
        return results
    
    
