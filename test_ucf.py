import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.cuda.amp import autocast, GradScaler
from models.model import CPLVAD
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.ucf_detectionMAP import getDetectionMAP as dmAP
import configs.ucf as ucf_option_pseudo
from scipy.ndimage import median_filter, gaussian_filter

def postprocess(scores):
    s = np.asarray(scores, dtype=np.float64)
    if len(s) >= 5:
        s = median_filter(s, size=5)

    if len(s) > 3:
        s = gaussian_filter(s, sigma=3.0, mode='reflect')

    return s

def test(model,full_path, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device):
    
    model.to(device)
    model.eval()

    element_logits2_stack = []
    lennn = []
    names = []

    with torch.no_grad():
        for i, item in enumerate(testdataloader):
            visual = item[0].squeeze(0)
            length = item[2]
            name = item[3]
            names.append(name)
            length = int(length)
            len_cur = length
            lennn.append(len_cur*16)
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)
            visual = visual.to(device)
            lengths = torch.zeros(int(length / maxlen) + 1)
            for j in range(int(length / maxlen) + 1):
                if j == 0 and length < maxlen:
                    lengths[j] = length
                elif j == 0 and length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                elif length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                else:
                    lengths[j] = length
            lengths = lengths.to(int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)
            with autocast():
                  logits1,logits2= model(visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
            prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))
            if i == 0:
                ap1 = prob1
                ap2 = prob2
            else:
                ap1 = torch.cat([ap1, prob1], dim=0)
                ap2 = torch.cat([ap2, prob2], dim=0)
            element_logits2 = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
            element_logits2 = np.repeat(element_logits2, 16, 0)
            element_logits2_stack.append(element_logits2)

    ap1 = ap1.cpu().numpy()
    ap2 = ap2.cpu().numpy()
    ap1 = ap1.tolist()
    ap2 = ap2.tolist()
    from scipy.ndimage import gaussian_filter

    ap1 = postprocess(ap1)
    ap2 = gaussian_filter(ap2, sigma = 3)

    ROC1 = roc_auc_score(gt, np.repeat(ap1, 16))
    AP1 = average_precision_score(gt, np.repeat(ap1, 16))
    ROC2 = roc_auc_score(gt, np.repeat(ap2, 16))
    AP2 = average_precision_score(gt, np.repeat(ap2, 16))
    print("AUC1: ", ROC1, " AP1: ", AP1)
    print("AUC2: ", ROC2, " AP2:", AP2)
    averageMAP = 0
    for i in range(5):
        print('mAP@{0:.1f} ={1:.2f}%'.format(iou[i], dmap[i]))
        averageMAP += dmap[i]
    averageMAP = averageMAP/(i+1)
    print('average MAP: {:.2f}'.format(averageMAP))

    return ROC1, AP1


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option_pseudo.parser.parse_args()
    full_path = "ckpts"
    label_map = dict({'Normal': 'Normal', 'Abuse': 'Abuse', 'Arrest': 'Arrest', 'Arson': 'Arson', 'Assault': 'Assault', 'Burglary': 'Burglary', 'Explosion': 'Explosion', 'Fighting': 'Fighting', 'RoadAccidents': 'RoadAccidents', 'Robbery': 'Robbery', 'Shooting': 'Shooting', 'Shoplifting': 'Shoplifting', 'Stealing': 'Stealing', 'Vandalism': 'Vandalism'})
    testdataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)
    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)
    model = CPLVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)
    model_param = torch.load('ckpts/ucf.pth', weights_only=False)['model_state_dict']
    model.load_state_dict(model_param)

    test(model, full_path, testdataloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)