from collections import OrderedDict

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from clip import clip
from models.blocks import (get_sinusoid_encoding, TransformerBlock, MaskedConv1D, LayerNorm)
from typing import List


class Generate_gt(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
       # self.thresholds = [0.45, 0.5, 0.55,0.6,0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95 ]
    
        
    def forward(self,batch_size,logits,lengths,grouping,filter,cumulative_thresh,flat_ratio1,flat_ratio2,thresholds):
        
       
        results = [{} for _ in range(logits[0].shape[0])]
        label = [{} for _ in range(logits[0].shape[0])]


        for i in range(logits[0].shape[0]): # 배치 수수
            
            all_segments = []
            
            for j in range(5): # 멀티 스케일
                
                log = logits[j][i].squeeze(1)
                log = log[ 0:lengths[i]]
                
                for th in thresholds:
                    mask = log >= th
                    indices = torch.nonzero(mask).squeeze()
                    values = log[mask]
                    segment = self.group_consecutive(indices,values)
                    segment = self.grouping(segment,grouping)
                    segment = self.filter(segment,filter)
                    all_segments.extend(segment) 
            
            results[i] = all_segments  
            pseu = self.new(results[i],lengths[i],cumulative_thresh) 
            label[i] = self.apply_adaptive_gaussian(pseu,flat_ratio1,flat_ratio2)
        
    
     
        return label
    
    def apply_adaptive_gaussian(self,arr,flat_ratio1,flat_ratio2):
    
        
        if isinstance(arr, torch.Tensor):
            arr = arr.clone().detach().float()
        else:
            arr = torch.tensor(arr, dtype=torch.float32)

        result = arr.clone()

        in_sequence = False
        start = 0

        for i in range(len(arr)):
           if arr[i] == 1 and not in_sequence:
              start = i
              in_sequence = True
           elif arr[i] != 1 and in_sequence:
              end = i
              in_sequence = False

           
              length = end - start
              center = (start + end - 1) / 2.0
              sigma = length / 3

              positions = torch.arange(start, end, dtype=torch.float32)
             # gauss = torch.exp(-0.5 * ((positions - center) / sigma) ** 2)
              gauss = self.flat_gaussian(length, flat_ratio1)

              result[start:end] = gauss

   
        if in_sequence:
           end = len(arr)
           length = end - start
           center = (start + end - 1) / 2.0
           sigma = length / 3
           positions = torch.arange(start, end, dtype=torch.float32)
         #  gauss = torch.exp(-0.5 * ((positions - center) / sigma) ** 2)
           gauss = self.flat_gaussian(length, flat_ratio2)  # 0.55, 0.6
           result[start:end] = gauss

        return result
    
    def flat_gaussian(self,length, flat_ratio):
        result = torch.ones(length)
        flat_len = int(length * flat_ratio)
        tail_len = (length - flat_len) // 2

        if tail_len > 0:
        
           sigma = tail_len / 3.0
           x = torch.arange(tail_len, dtype=torch.float32)
           decay = torch.exp(-0.5 * ((x - (tail_len - 1)) / sigma) ** 2)

       
           result[:tail_len] = decay
           result[-tail_len:] = decay.flip(0)

        return result

    
    def group_consecutive(self, values, confidences):
        
        
        if isinstance(values, torch.Tensor):
            if values.dim() == 0:
                values = [values.item()]
            else:
                 values = values.tolist()
        elif isinstance(values, int):
            values = [values]
            
        if isinstance(confidences, torch.Tensor):
            if confidences.dim() == 0:
                confidences = [confidences.item()]
            else:
                confidences = confidences.tolist()
        elif isinstance(confidences, (int, float)):
            confidences = [confidences]
        
        if len(values) == 0 and len(confidences) == 0:
            return []

            
        values = [v.item() if isinstance(v, torch.Tensor) else v for v in values]
        confidences = [c.item() if isinstance(c, torch.Tensor) else c for c in confidences]

        sorted_pairs = sorted(zip(values, confidences), key=lambda x: x[0])
        result = []
        
        start = sorted_pairs[0][0]
        prev = sorted_pairs[0][0]
        conf_list = [sorted_pairs[0][1]]

        for curr, conf in sorted_pairs[1:]:
            if curr == prev + 1:
                conf_list.append(conf)
                prev = curr
            else:
                avg_conf = sum(conf_list) / len(conf_list)
                if start == prev:
                    result.append(([start], avg_conf))
                else:
                    result.append(([start, prev], avg_conf))
                 
                start = curr
                prev = curr
                conf_list = [conf]
        
        avg_conf = sum(conf_list) / len(conf_list)
        if start == prev:
           result.append(([start], avg_conf))
        else:
           result.append(([start, prev], avg_conf))

        return result
    
    
    
    
    def grouping(self, segment, gap):
        
        merged = []
        
        i=0
        
        while i< len(segment):
            current = segment[i]
            current_time = current[0]
            current_score = current[1]
            
            if len(current_time) != 2:
                
                start = current_time[0]
                end = start
            else:
                start,end = current_time
            
            time1 = end - start
                
            while i+1 < len(segment):
                    
                    next = segment[i+1]
                    next_time = next[0]
                    next_score = next[1] 
                    
                    if len(next_time) != 2:
                        
                        next_start = next_time[0]
                        next_end = next_start
                        
                    else:
                        next_start, next_end = next_time
                        
                    if next_start - end <= gap:
                        
                        time2 = next_end - next_start
                        end = next_end
                        total_time = time1 + time2
                        
                        if total_time == 0:
                            
                            current_score = (current_score+ next_score)/2
                        else: 
                            current_score = (current_score * time1 + next_score * time2) / total_time
                            
                        i+=1
                    else:
                        break
                    
                    
            merged.append(([start,end],current_score))
            i+=1
                        
        
        return merged
    
    
    def filter(self, segment, duration_min ):

        
        filtered = []
        
        for (time_range, score) in segment:
            start, end = time_range
            duration = end-start
            if duration > duration_min:
                filtered.append(([start,end],score))
        
        return filtered
    
    def cal_iou(self, s1,e1, s2, e2):
        inter_start = max(s1,s2)
        inter_end =min(e1,e2)
        intersection = max(0, inter_end - inter_start)
        union = (e1-s1) + (e2 - s2)- intersection
        return intersection/ (union + 1e-8)
          

    
    def new(self, result,lengths,thresh):
        
        
        zero_tensor = torch.zeros(lengths)
        
        if len( result) == 0:
           return zero_tensor
        
        for i in range(len(result)):
            
            start = result[i][0][0]
            end = result[i][0][1]
            con = result[i][1]
            
            zero_tensor[start:end] += con
        #binary_tensor = (zero_tensor >= 6).float()


        binary_tensor = torch.where(zero_tensor >= thresh, torch.tensor(1), torch.tensor(0.0))
        # binary_tensor = torch.where(
        #  zero_tensor >= 7, 
        #  torch.tensor(1, device=zero_tensor.device), 
        #   torch.where(
        # zero_tensor <= 3, 
        # torch.tensor(0.0, device=zero_tensor.device) ,
        # torch.tensor(-1.0, device=zero_tensor.device)
        #  )
        #    )
        
       # import pdb; pdb.set_trace()

        
        return binary_tensor
    
    
    

class LayerNorm1(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """
    def __init__(
        self,
        num_channels,
        eps = 1e-5,
        affine = True,
        device = None,
        dtype = None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x**2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps) 

        # apply weight and bias
        if self.affine:
            out *= self.weight
            out += self.bias

        return out


class PtTransformerClsHead(nn.Module):
    """
    1D Conv heads for classification
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        empty_cls = []
    ):
        super().__init__()
        self.act = act_layer()

        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm1(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        # classifier
        self.cls_head = MaskedConv1D(
                feat_dim, num_classes, kernel_size,
                stride=1, padding=kernel_size//2
            )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # a quick fix to empty categories:
        # the weights assocaited with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        # import pdb;pdb.set_trace()
    

        # apply the classifier for each pyramid level
        out_logits = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))

            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits, )

        # fpn_masks remains the same
        return out_logits
    


class FPNIdentity(nn.Module):
    def __init__(
        self,
        in_channels,      # input feature channels, len(in_channels) = #levels
        out_channel,      # output feature channel
        scale_factor=2.0, # downsampling rate between two fpn levels
        start_level=0,    # start fpn level
        end_level=-1,     # end fpn level
        with_ln=True,     # if to apply layer norm at the end
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.scale_factor = scale_factor

        self.start_level = start_level
        if end_level == -1:
            self.end_level = len(in_channels)
        else:
            self.end_level = end_level
        assert self.end_level <= len(in_channels)
        assert (self.start_level >= 0) and (self.start_level < self.end_level)

        self.fpn_norms = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            # check feat dims
            assert self.in_channels[i] == self.out_channel
            # layer norm for order (B C T)
            if with_ln:
                fpn_norm = LayerNorm(out_channel)
            else:
                fpn_norm = nn.Identity()
            self.fpn_norms.append(fpn_norm)

    def forward(self, inputs, fpn_masks):
        # inputs must be a list / tuple
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) ==  len(self.in_channels)

        # apply norms, fpn_masks will remain the same with 1x1 convs
        fpn_feats = tuple()
        new_fpn_masks = tuple()
        for i in range(len(self.fpn_norms)):
            x = self.fpn_norms[i](inputs[i + self.start_level])
            fpn_feats += (x, )
            new_fpn_masks += (fpn_masks[i + self.start_level], )

        return fpn_feats, new_fpn_masks
    
    
    
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)



class CPLVAD_pseudo(nn.Module):
    def __init__(self,
                 batch_size:int,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 grouping: int,
                 filter:int,
                 cumulative_thresh1:float,
                 cumulative_thresh2:float,
                 flat_ratio1:float,
                 flat_ratio2:float,
                 threshold1:List[float],
                 threshold2:List[float],
                 device):
        super().__init__()
        
        self.batch = batch_size
        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device
        self.with_ln = True
        self.max_len = 256
        self.n_in = 512
        self.n_embd = 512
        self.mha_win_size = [8,8,8,8,8,-1]

        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False
            
        self.arch = arch = (2, 2, 5)
            
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        self.stem = nn.ModuleList()
        if isinstance(self.n_in, (list, tuple)):
            assert isinstance(self.n_embd, (list, tuple)) and len(self.n_in) == len(self.n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(self.n_in, self.n_embd)
            ])
            self.n_in = self.n_embd = sum(self.n_embd)
        else:
            self.proj = None
        
        for idx in range(arch[0]):
            self.n_in = 512 if idx > 0 else self.n_in
            self.embd.append(
                MaskedConv1D(
                    in_channels=self.n_in, out_channels=512, kernel_size=3,
                    stride=1, padding=3//2, bias=(not self.with_ln)
                )
            )
            if self.with_ln:
                self.embd_norm.append(LayerNorm(512))
            else:
                self.embd_norm.append(nn.Identity())
                
                
        
        for idx in range(arch[1]):
            self.stem.append(TransformerBlock(  
                    n_embd=512, n_head=8,
                    n_ds_strides=(1, 1),
                    attn_pdrop=0.0,
                    proj_pdrop=0.0,
                    path_pdrop=0.1,
                    mha_win_size=8,
                    use_rel_pe=False
                
                )
            )
            
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(TransformerBlock(
                    n_embd=512, n_head=8,
                    n_ds_strides=(2, 2),
                    attn_pdrop=0.0,
                    proj_pdrop=0.0,
                    path_pdrop=0.1,
                    mha_win_size=self.mha_win_size[1+idx],
                    use_rel_pe=False
              
                )
            )
        
        self.neck =FPNIdentity(
            in_channels=[512,512,512,512,512,512],
            out_channel=512,
            scale_factor=2.0,
            start_level=0,
            end_level=-1,     # end fpn level
             with_ln=True    
        )
        self.use_abs_pe = True
   
        
        self.relu = nn.ReLU(inplace=True)
                
        pos_embd = get_sinusoid_encoding(256, 512) / (512**0.5)
        self.register_buffer("pos_embd", pos_embd, persistent=False)
        
        
        block1     = lambda: nn.Sequential(OrderedDict([
        ("c_fc",   nn.Linear(visual_width, visual_width * 4)),
        ("gelu",   QuickGELU()),
        ("c_proj", nn.Linear(visual_width * 4, visual_width))
          ]))
        self.mlp1 = nn.ModuleList([ block1() for _ in range(6) ])
        
        block2     = lambda: nn.Sequential(OrderedDict([
        ("c_fc",   nn.Linear(visual_width, visual_width * 4)),
        ("gelu",   QuickGELU()),
        ("c_proj", nn.Linear(visual_width * 4, visual_width))
          ]))
        self.mlp2 = nn.ModuleList([ block2() for _ in range(6) ])
        


        self.fg_head = PtTransformerClsHead(
            input_dim=512, feat_dim=512,num_classes= 1,
            kernel_size=3,
            prior_prob=0.01, 
            with_ln=True,
            num_layers=3,   
            empty_cls=[]
        )

        
        self.pseudo = Generate_gt()
        self.grouping = grouping #5
        self.filter = filter # 3
        self.cumulative_thresh1 = cumulative_thresh1#7
        self.cumulative_thresh2 = cumulative_thresh2
        self.flat_ratio1 = flat_ratio1 #0.55
        self.flat_ratio2 = flat_ratio2 #0.6
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)

        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)

        self.apply(self.__init_weights__)
       

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if not any(p.requires_grad for p in module.parameters()):
            return 
        
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None and module.bias.requires_grad:
                nn.init.constant_(module.bias, 0.0)
 
    def encode_textprompt(self, text):
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel.encode_token(word_tokens)
        text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat([len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)

        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]

        text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)

        return text_features
    
    

    

    def forward(self, visual, padding_mask, text, lengths):
        B = lengths.size(0) 
        T = 256 
        B,T,C = visual.size()
        vid = visual.permute(0,2,1)
        text_feat=[]
        
        text_features_ori = self.encode_textprompt(text)
        
        for l in range(6):
            text_feat.append(text_features_ori)
        
        time_idx = torch.arange(T, device=lengths.device).unsqueeze(0).expand(B, T)
        mask = time_idx < lengths.unsqueeze(1) 
        mask = mask.unsqueeze(2)
        mask = mask.permute(0,2,1)
        mask = mask.to(vid.device)
       
        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0] \
                    for proj, s in zip(self.proj, x.split(self.n_in, dim=1))
                ], dim=1
            )
        


        for idx in range(len(self.embd)):
            vid, mask = self.embd[idx](vid, mask)
            vid = self.relu(self.embd_norm[idx](vid))
            
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            vid = vid + pe[:, :, :T] * mask.to(vid.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            vid= vid + pe[:, :, :T] * mask.to(vid.dtype)
           
        for idx in range(len(self.stem)): #2
            vid, mask = self.stem[idx](vid, mask)
         
        lv_vid = tuple()
        lv_masks = tuple()
        lv_vid += (vid, )
        lv_masks += (mask, )
        
        for idx in range(len(self.branch)): 
            vid, mask = self.branch[idx](vid, mask) #B, C, T
            lv_vid += (vid, )
            lv_masks += (mask, )

        lv_vid, lv_masks = self.neck(lv_vid, lv_masks)

        logits1  = self.fg_head(lv_vid, lv_masks)
        
        lv_vid = [x.transpose(2, 1) for x in lv_vid]
        logits1 = [x.transpose(2, 1) for x in logits1]
    
            
        logits2=[]
        for i in range(len(lv_vid)):
   
            vision = lv_vid[i]   
            text_features = text_feat[i].unsqueeze(0)
            text_features = text_features.expand(B, text_features.shape[1], text_features.shape[2])

            text_features = text_features + self.mlp2[i](self.mlp1[i](text_features))
            text_features_norm   = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)
            text_features_norm = text_features_norm.permute(0, 2, 1)
            visual_features_norm = vision / (vision.norm(dim=-1, keepdim=True) + 1e-6)
            logits_2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07
            logits2.append(logits_2)
        

        
    
            
        logit_a=[]
        logit_b=[]
   
            
        for i in range(len(logits1)):
               logits1[i] = torch.sigmoid(logits1[i])
               log1 = logits1[i].repeat_interleave(2**i, dim=1)
               logit_a.append(log1)
               
               logits2[i] = (1-torch.softmax(logits2[i], dim=-1)[:,:,0]).unsqueeze(-1) 
               log2 = logits2[i].repeat_interleave(2**i, dim=1)
               logit_b.append(log2)
       
         
        pseudo1 = self.pseudo(self.batch,
                              logit_a, 
                              lengths,
                              self.grouping,
                              self.filter ,
                              self.cumulative_thresh1,
                              self.flat_ratio1,
                              self.flat_ratio2,
                              self.threshold1)
        
            
        pseudo2 = self.pseudo(self.batch,
                              logit_b, 
                              lengths,
                              self.grouping,
                              self.filter ,
                              self.cumulative_thresh1,
                              self.flat_ratio1,
                              self.flat_ratio2,
                              self.threshold1)
        
   
        return pseudo1, pseudo2

            




class CPLVAD(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 device):
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = 0
        self.prompt_postfix = prompt_postfix
        self.device = device
        self.with_ln = True
        self.max_len = 256
        self.n_in = 512
        self.n_embd = 512
        self.mha_win_size = [8,8,8,8,8,-1]

        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False
            
        self.arch = arch = (2, 2, 5)
            
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        self.stem = nn.ModuleList()
        
        
        
        if isinstance(self.n_in, (list, tuple)):
            assert isinstance(self.n_embd, (list, tuple)) and len(self.n_in) == len(self.n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(self.n_in, self.n_embd)
            ])
            self.n_in = self.n_embd = sum(self.n_embd)
        else:
            self.proj = None
        
        for idx in range(arch[0]):
            self.n_in = 512 if idx > 0 else self.n_in
            self.embd.append(
                MaskedConv1D(
                    in_channels=self.n_in, out_channels=512, kernel_size=3,
                    stride=1, padding=3//2, bias=(not self.with_ln)
                )
            )
            if self.with_ln:
                self.embd_norm.append(LayerNorm(512))
            else:
                self.embd_norm.append(nn.Identity())
                
                
        
        for idx in range(arch[1]):
            self.stem.append(TransformerBlock(  
                    n_embd=512, n_head=8,
                    n_ds_strides=(1, 1),
                    attn_pdrop=0.0,
                    proj_pdrop=0.0,
                    path_pdrop=0.1,
                    mha_win_size=8,
                    use_rel_pe=False
                
                )
            )
            
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(TransformerBlock(
                    n_embd=512, n_head=8,
                    n_ds_strides=(2, 2),
                    attn_pdrop=0.0,
                    proj_pdrop=0.0,
                    path_pdrop=0.1,
                    mha_win_size=self.mha_win_size[1+idx],
                    use_rel_pe=False
              
                )
            )
        
        self.neck =FPNIdentity(
            in_channels=[512,512,512,512,512,512],
            out_channel=512,
            scale_factor=2.0,
            start_level=0,
            end_level=-1,     # end fpn level
             with_ln=True
            
        )
        
        
        self.use_abs_pe = True
        self.relu = nn.ReLU(inplace=True)
                
        pos_embd = get_sinusoid_encoding(256, 512) / (512**0.5)
        self.register_buffer("pos_embd", pos_embd, persistent=False)
        
        
        block1     = lambda: nn.Sequential(OrderedDict([
        ("c_fc",   nn.Linear(visual_width, visual_width * 4)),
        ("gelu",   QuickGELU()),
        ("c_proj", nn.Linear(visual_width * 4, visual_width))
          ]))
        self.mlp1 = nn.ModuleList([ block1() for _ in range(6) ])
        
        block2     = lambda: nn.Sequential(OrderedDict([
        ("c_fc",   nn.Linear(visual_width, visual_width * 4)),
        ("gelu",   QuickGELU()),
        ("c_proj", nn.Linear(visual_width * 4, visual_width))
          ]))
        self.mlp2 = nn.ModuleList([ block2() for _ in range(6) ])

        self.fg_head = PtTransformerClsHead(
            input_dim=512, feat_dim=512,num_classes= 1,
            kernel_size=3,
            prior_prob=0.01, 
            with_ln=True,
            num_layers=3,   
            empty_cls=[]
        )
        self.normal_text_prompt_embeddings = nn.Embedding(2, self.embed_dim)
        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)
        self.level_text_prompt_embeddings = nn.Embedding(6, self.embed_dim)
        # self.deep_text_prompt_embeddings = nn.Embedding(12, self.embed_dim)
        nn.init.normal_(self.normal_text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.level_text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        self.apply(self.__init_weights__)
       

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if not any(p.requires_grad for p in module.parameters()):
            return 
        
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None and module.bias.requires_grad:
                nn.init.constant_(module.bias, 0.0)       
                
    def encode_textprompt(self, text):
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel.encode_token(word_tokens)
        text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat([len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)
        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]

        text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)

        return text_features
    
    def encode_level_textprompt(self, text):

        level_text_features =[]
        word_tokens = clip.tokenize(text).to(self.device)  # 7, 77
        word_embedding = self.clipmodel.encode_token(word_tokens) # 7, 77, 512
        text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat([len(text), 1, 1]) # 7, 77, 512
        level_embeddings = self.level_text_prompt_embeddings(torch.arange(6).to(self.device)) # 6, 512
        normality_embeddings = self.normal_text_prompt_embeddings(torch.arange(2).to(self.device))  # 2, 512
        text_tokens = torch.zeros(len(text), 77).to(self.device) # 7, 77

        for j in range(len(level_embeddings)):

            for i in range(len(text)):
                if i == 0:
                    ind = torch.argmax(word_tokens[i], -1) # eos 위치 찾기
                    text_embeddings[i, 0] = word_embedding[i, 0] # sos 토큰 가져오기
                    text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind] # learnable+유효한 토큰큰
                    text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix-1] = level_embeddings[j] # scale 임베딩 넣기
                    text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = normality_embeddings[0] # normal 임베딩 넣기
                    text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix + 1] = word_embedding[i, ind] # 얘가 마지막 EOS 
                    text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix+1] = word_tokens[i, ind] # 얘가 마지막 token
                else:
                    ind = torch.argmax(word_tokens[i], -1)
                    text_embeddings[i, 0] = word_embedding[i, 0]
                    text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]
                    text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix-1] = level_embeddings[j]
                    text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = normality_embeddings[1]
                    text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix + 1] = word_embedding[i, ind] # 얘가 마지막 EOS 
                    text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix+1] = word_tokens[i, ind] # 얘가 마지막 token
            text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)
            level_text_features.append(text_features)
        return level_text_features
    

    def forward(self, visual, padding_mask, text, lengths):
    
        B = lengths.size(0) 
        T = 256 
        B,T,C = visual.size()
        vid = visual.permute(0,2,1)
        text_feat=[]
        
        text_feat = self.encode_level_textprompt(text)
        text_features_ori = text_feat

        time_idx = torch.arange(T, device=lengths.device).unsqueeze(0).expand(B, T)
        mask = time_idx < lengths.unsqueeze(1) 
        mask = mask.unsqueeze(2)
        mask = mask.permute(0,2,1)
        mask = mask.to(vid.device)
       
        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0] \
                    for proj, s in zip(self.proj, x.split(self.n_in, dim=1))
                ], dim=1
            )
        


        for idx in range(len(self.embd)):
            vid, mask = self.embd[idx](vid, mask)
            vid = self.relu(self.embd_norm[idx](vid))
            
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            vid = vid + pe[:, :, :T] * mask.to(vid.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            vid= vid + pe[:, :, :T] * mask.to(vid.dtype)
           
        for idx in range(len(self.stem)): #2
            vid, mask = self.stem[idx](vid, mask)
         
        lv_vid = tuple()
        lv_masks = tuple()
        lv_vid += (vid, )
        lv_masks += (mask, )
        
        for idx in range(len(self.branch)): 
            vid, mask = self.branch[idx](vid, mask) #B, C, T
            lv_vid += (vid, )
            lv_masks += (mask, )

        lv_vid, lv_masks = self.neck(lv_vid, lv_masks)

        logits1  = self.fg_head(lv_vid, lv_masks)
        
        lv_vid = [x.transpose(2, 1) for x in lv_vid]
        logits1 = [x.transpose(2, 1) for x in logits1]
    
            
        logits2=[]
        for i in range(len(lv_vid)):
            vision = lv_vid[i]  
            text_features = text_feat[i].unsqueeze(0)
            text_features = text_features.expand(B, text_features.shape[1], text_features.shape[2])
            text_features = text_features + self.mlp2[i](self.mlp1[i](text_features))
            # text_features = text_features + self.mlp1[i](text_features)
            text_features_norm   = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)
            text_features_norm = text_features_norm.permute(0, 2, 1)
            visual_features_norm = vision / (vision.norm(dim=-1, keepdim=True) + 1e-6)
            logits_2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07
            logits2.append(logits_2)
        
        if self.training:
            
            logit_a=[]
            logit_b=[]
            for i in range(len(logits1)):
               log1 = logits1[i].repeat_interleave(2**i, dim=1)
               logit_a.append(log1)
               log2 = logits2[i].repeat_interleave(2**i, dim=1)
               logit_b.append(log2)
         
            return text_features_ori, logit_a, logit_b, lv_masks[0] 

            
        else:
            
            logits1_0 = torch.sigmoid(logits1[0])
            logits1_1 = torch.sigmoid(logits1[1].repeat_interleave(2, dim=1))
            logits1_2 = torch.sigmoid(logits1[2].repeat_interleave(4, dim=1))
            logits1_3 = torch.sigmoid(logits1[3].repeat_interleave(8, dim=1))
            logits1_4 = torch.sigmoid(logits1[4].repeat_interleave(16, dim=1))
            logits1_5 = torch.sigmoid(logits1[5].repeat_interleave(32, dim=1))
            
            logits2_0 = logits2[0]
            logits2_1 = logits2[1].repeat_interleave(2, dim=1)
            logits2_2 = logits2[2].repeat_interleave(4, dim=1)
            logits2_3 = logits2[3].repeat_interleave(8, dim=1)
            logits2_4 = logits2[4].repeat_interleave(16, dim=1)
            logits2_5 = logits2[5].repeat_interleave(32, dim=1)
       
             
            logit1 = (logits1_0+logits1_1+logits1_2+logits1_3+logits1_4+logits1_5)/6
            logit2 = (logits2_0+logits2_1+logits2_2+logits2_3+logits2_4+logits2_5)/6
        
            
            return logit1,logit2
     
            