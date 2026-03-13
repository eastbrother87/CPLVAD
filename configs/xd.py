import argparse

parser = argparse.ArgumentParser(description='VadCLIP')
parser.add_argument('--seed', default=234, type=int)

parser.add_argument('--embed-dim', default=512, type=int)

parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)

parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=1, type=int)
parser.add_argument('--attn-window', default=64, type=int)
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=7, type=int)

parser.add_argument('--model-name', type=str, default="best_model", required=False)
parser.add_argument('--max-epoch', default=8, type=int)

parser.add_argument('--use-checkpoint', default=True, type=bool)

parser.add_argument('--checkpoint-path2', default='/ssd1/VAD/CPL_VAD/exp_xd/novid1/novid1.pth')
parser.add_argument('--pseudo_path', default='')
parser.add_argument('--online-pseudo',default=True, type=bool)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--train-list', default='/ssd1/VAD/CPL_VAD/list/xd_CLIP_rgb.csv')
parser.add_argument('--test-list', default='/ssd1/VAD/CPL_VAD/list/xd_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', default='/ssd1/VAD/CPL_VAD/list/gt.npy')
parser.add_argument('--gt-segment-path', default='/ssd1/VAD/CPL_VAD/list/gt_segment.npy')
parser.add_argument('--gt-label-path', default='/ssd1/VAD/CPL_VAD/list/gt_label.npy')
parser.add_argument('--vision-lr', default=1e-3, type=float)
parser.add_argument('--text-lr', default=1e-3, type=float)
parser.add_argument('--scheduler-rate', default=0.1)
parser.add_argument('--scheduler-milestones', default=[3, 6, 10])


## pseudo label

parser.add_argument('--grouping', default=22, type=float)
parser.add_argument('--filter', default=5, type=float)
parser.add_argument('--flat-ratio1', default=0.95, type=float)
parser.add_argument('--flat-ratio2', default=0.95, type=float) 


#logit1으로 pseudo label주기기
parser.add_argument('--threshold1', default=[ 0.55,0.6,0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95 ]) 
parser.add_argument('--cumulative-thresh1', default=33, type=float)#31   17 18 19 20 21

#logit2으로 pseudo label주기기
parser.add_argument('--threshold2', default=[ 0.55,0.6,0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95 ]) 
parser.add_argument('--cumulative-thresh2', default=33, type=float)

