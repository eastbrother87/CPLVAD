import argparse

parser = argparse.ArgumentParser(description='VadCLIP') # 800
parser.add_argument('--seed', default=234, type=int)

parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=2, type=int)
parser.add_argument('--attn-window', default=8, type=int)
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=14, type=int)

parser.add_argument('--model-name', type=str, required=False)

parser.add_argument('--max-epoch', default=10, type=int)
parser.add_argument('--use-checkpoint', default=True, type=bool)
parser.add_argument('--online-pseudo',default=True, type=bool)
parser.add_argument('--checkpoint-path2', default='/ssd1/VAD/CPL_VAD/exp_ucf/novid7/novid7.pth')
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--train-list', default='/ssd1/VAD/CPL_VAD/list/ucf_CLIP_rgb.csv')
parser.add_argument('--test-list', default='/ssd1/VAD/CPL_VAD/list/ucf_CLIP_rgbtest.csv')  # anoauc
parser.add_argument('--gt-path', default='/ssd1/VAD/CPL_VAD/list/gt_ucf.npy')  # anoauc
parser.add_argument('--gt-segment-path', default='/ssd1/VAD/CPL_VAD/list/gt_segment_ucf.npy')
parser.add_argument('--gt-label-path', default='/ssd1/VAD/CPL_VAD/list/gt_label_ucf.npy')

parser.add_argument('--vision-lr', default=1e-4, type=float)
parser.add_argument('--text-lr', default=1e-4, type=float)  
parser.add_argument('--scheduler-rate', default=0.1)
parser.add_argument('--scheduler-milestones', default=[4, 8])



## pseudo label
parser.add_argument('--grouping', default=6, type=float)  # 1-15 (15)
parser.add_argument('--filter', default=4, type=float)   # 1-5 (5)
parser.add_argument('--flat-ratio1', default=0.55, type=float)  # 0.3-0.9 (13)
parser.add_argument('--flat-ratio2', default=0.6, type=float) # 0.3-0.9 (13)

# logit1으로 pseudo label 생성
parser.add_argument('--threshold1', default=[ 0.55,0.6,0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95 ]) # 1으로 2에게 pseudo label
parser.add_argument('--cumulative-thresh1', default=22, type=float)  # (21) 17 18 19 20 21 22 23

# logit2으로 pseudo label 생성
parser.add_argument('--threshold2', default=[ 0.55,0.6,0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95 ]) #2로 1에세 pseudo label
parser.add_argument('--cumulative-thresh2', default=22, type=float)  # 5-10