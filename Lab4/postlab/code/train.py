import argparse
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.student import StudentNet
from models.custom_student import LiteSegNet
from utils.dataset import get_voc
from utils.kd import response_kd_loss, feature_kd_loss
from utils.metrics import update_confusion, mean_iou
from utils.teacher import load_teacher, teacher_forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['student', 'custom'], default='student',
                        help='student = baseline StudentNet (Steps 1-2), custom = LiteSegNet (Step 3)')
    parser.add_argument('--kd', choices=['none', 'response', 'feature', 'both'], default='none',
                        help='both = response-based + feature-based KD')
    parser.add_argument('--aug', action='store_true', help='random horizontal flips during training')
    parser.add_argument('--data', default='./data')
    parser.add_argument('-e', '--epochs', type=int, default=3, help='maximum number of epochs')
    parser.add_argument('-b', '--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--size', type=int, default=320)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--tau', type=float, default=4.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--lr-patience', type=int, default=2,
                        help='epochs without val mIoU improvement before the LR is reduced')
    parser.add_argument('--lr-factor', type=float, default=0.5,
                        help='factor the LR is multiplied by when it is reduced')
    parser.add_argument('--patience', type=int, default=5,
                        help='early stopping: epochs without improvement before training stops')
    parser.add_argument('--tol', type=float, default=0.1,
                        help='early stopping: min mIoU gain (in points) that counts as an improvement')
    parser.add_argument('--iters', type=int, default=0,
                        help='kick-start: stop after this many iterations (0 = full training)')
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    train_loader = DataLoader(get_voc(args.data, 'train', args.size, args.aug), batch_size=args.batch,
                              shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(get_voc(args.data, 'val', args.size), batch_size=args.batch,
                            shuffle=False, num_workers=args.workers)

    student = (LiteSegNet() if args.model == 'custom' else StudentNet()).to(device)
    name = f'{args.model}_{args.kd}'   # e.g. student_none, custom_both
    params = list(student.parameters())
    teacher = load_teacher(device) if args.kd != 'none' else None

    # 1x1 adapters map student features to the teacher's channel counts (training only)
    TEACHER_CH = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}
    adapters = None
    if args.kd in ('feature', 'both'):
        adapters = nn.ModuleDict({tap: nn.Conv2d(ch, TEACHER_CH[t_layer], 1)
                                  for tap, (t_layer, ch) in student.KD_TAPS.items()}).to(device)
        params += list(adapters.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr)
    # Reduce the LR when the validation mIoU (higher is better) stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=args.lr_factor, patience=args.lr_patience)

    print(f'Training {args.model} ({sum(p.numel() for p in student.parameters()):,} params) '
          f'with kd={args.kd} on {device}' +
          (f' for {args.iters} iterations' if args.iters else f' for up to {args.epochs} epochs'))

    losses = []
    val_mious = []
    best_miou, best_epoch = -1.0, 0          # best raw val mIoU, used for the checkpoint
    best_for_stop, bad_epochs = float('-inf'), 0   # improvement tracking, with tolerance
    epochs_run = 0
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
                if args.kd in ('response', 'both'):
                    loss = response_kd_loss(out['out'], z_t, mask, args.alpha, args.beta, args.tau)
                else:
                    loss = F.cross_entropy(out['out'], mask, ignore_index=255)
                if args.kd in ('feature', 'both'):
                    loss = loss + args.gamma * sum(
                        feature_kd_loss(adapters[tap](out[tap]), f_t[t_layer])
                        for tap, (t_layer, _) in student.KD_TAPS.items())
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

        epochs_run = epoch
        student.eval()
        conf = torch.zeros(21, 21, dtype=torch.long)
        with torch.no_grad():
            for img, mask in val_loader:
                pred = student(img.to(device))['out'].argmax(1).cpu()
                conf = update_confusion(conf, pred, mask)
        miou = 100 * mean_iou(conf)
        val_mious.append(miou)
        lr_used = optimizer.param_groups[0]['lr']
        print(f'epoch {epoch}/{args.epochs}  loss {sum(losses[-len(train_loader):]) / len(train_loader):.3f}'
              f'  val mIoU {miou:.1f}  lr {lr_used:.2e}  ({time.time() - start:.0f} s)')

        # Checkpoint: keep the weights from the epoch with the best validation mIoU
        if miou > best_miou:
            best_miou, best_epoch = miou, epoch
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(student.state_dict(), f'checkpoints/{name}.pth')

        # Early stopping: an epoch counts as an improvement only if it beats the best by > tol
        if miou > best_for_stop + args.tol:
            best_for_stop, bad_epochs = miou, 0
        else:
            bad_epochs += 1

        scheduler.step(miou)

        if bad_epochs >= args.patience:
            print(f'Early stopping: no improvement > {args.tol} mIoU for {args.patience} epochs')
            break

    train_time = time.time() - start
    os.makedirs('plots', exist_ok=True)
    if args.iters:
        first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
        print(f'Kick-start finished: average loss {first:.3f} (first 5) -> {last:.3f} (last 5)')
    else:
        print(f'Finished: {epochs_run} epochs in {train_time:.0f} s ({train_time / 60:.1f} min) on {device}')
        print(f'Best val mIoU {best_miou:.1f} at epoch {best_epoch}')
        print(f'Saved checkpoints/{name}.pth (weights from epoch {best_epoch})')
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(val_mious) + 1), val_mious)
        plt.xlabel('epoch')
        plt.ylabel('val mIoU (%)')
        plt.title(f'Validation mIoU ({name})')
        plt.tight_layout()
        plt.savefig(f'plots/miou_{name}.png', dpi=100)
        print(f'Saved plots/miou_{name}.png')

    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title(f'Training loss ({name})')
    plt.tight_layout()
    plt.savefig(f'plots/loss_{name}.png', dpi=100)
    print(f'Saved plots/loss_{name}.png')


# The guard is needed on macOS/Windows, where DataLoader workers re-import this file
if __name__ == '__main__':
    main()
