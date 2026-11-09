import argparse
import os
import time
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.student import StudentNet
from utils.dataset import get_voc
from utils.kd import response_kd_loss, feature_kd_loss
from utils.metrics import update_confusion, mean_iou
from utils.teacher import load_teacher, teacher_forward

parser = argparse.ArgumentParser()
parser.add_argument('--kd', choices=['none', 'response', 'feature'], default='none')
parser.add_argument('--data', default='./data')
parser.add_argument('-e', '--epochs', type=int, default=3)
parser.add_argument('-b', '--batch', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--size', type=int, default=320)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--tau', type=float, default=4.0)
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--iters', type=int, default=0,
                    help='kick-start: stop after this many iterations (0 = full training)')
args = parser.parse_args()

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader = DataLoader(get_voc(args.data, 'train', args.size), batch_size=args.batch,
                          shuffle=True, num_workers=args.workers, drop_last=True)
val_loader = DataLoader(get_voc(args.data, 'val', args.size), batch_size=args.batch,
                        shuffle=False, num_workers=args.workers)

student = StudentNet().to(device)
params = list(student.parameters())
teacher = load_teacher(device) if args.kd != 'none' else None

# 1x1 adapters map student features to the teacher's channel counts (training only)
adapters = None
if args.kd == 'feature':
    adapters = nn.ModuleDict({'low': nn.Conv2d(24, 512, 1),
                              'high': nn.Conv2d(576, 2048, 1)}).to(device)
    params += list(adapters.parameters())

optimizer = torch.optim.Adam(params, lr=args.lr)
print(f'Training student with kd={args.kd} on {device}' +
      (f' for {args.iters} iterations' if args.iters else f' for {args.epochs} epochs'))

losses = []
start = time.time()
for epoch in range(1, (10**6 if args.iters else args.epochs) + 1):
    student.train()
    for img, mask in train_loader:
        img, mask = img.to(device), mask.to(device)
        out = student(img)
        if args.kd == 'none':
            loss = F.cross_entropy(out['out'], mask, ignore_index=255)
        else:
            with torch.no_grad():
                z_t, f_t = teacher_forward(teacher, img)
            if args.kd == 'response':
                loss = response_kd_loss(out['out'], z_t, mask, args.alpha, args.beta, args.tau)
            else:
                loss = F.cross_entropy(out['out'], mask, ignore_index=255) + args.gamma * (
                    feature_kd_loss(adapters['low'](out['low']), f_t['layer2']) +
                    feature_kd_loss(adapters['high'](out['high']), f_t['layer4']))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if args.iters:
            if len(losses) % 5 == 0:
                print(f'iter {len(losses):3d}/{args.iters}  loss {loss.item():.3f}')
            if len(losses) >= args.iters:
                break
    if args.iters:
        if len(losses) >= args.iters:
            break
        continue

    student.eval()
    conf = torch.zeros(21, 21, dtype=torch.long)
    with torch.no_grad():
        for img, mask in val_loader:
            pred = student(img.to(device))['out'].argmax(1).cpu()
            conf = update_confusion(conf, pred, mask)
    print(f'epoch {epoch}/{args.epochs}  loss {sum(losses[-len(train_loader):]) / len(train_loader):.3f}'
          f'  val mIoU {100 * mean_iou(conf):.1f}  ({time.time() - start:.0f} s)')

os.makedirs('plots', exist_ok=True)
if args.iters:
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    print(f'Kick-start finished: average loss {first:.3f} (first 5) -> {last:.3f} (last 5)')
else:
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(student.state_dict(), f'checkpoints/student_{args.kd}.pth')
    print(f'Saved checkpoints/student_{args.kd}.pth')
plt.figure(figsize=(6, 4))
plt.plot(losses)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title(f'Training loss (kd={args.kd})')
plt.tight_layout()
plt.savefig(f'plots/loss_{args.kd}.png', dpi=100)
print(f'Saved plots/loss_{args.kd}.png')
