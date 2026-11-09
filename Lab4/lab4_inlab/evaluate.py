import argparse
import os
import torch
from torch.utils.data import DataLoader

from models.student import StudentNet
from utils.dataset import get_voc
from utils.metrics import update_confusion, mean_iou
from utils.profiler import ms_per_image
from utils.teacher import load_teacher
from utils.visualization import save_comparison

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='./data')
parser.add_argument('--size', type=int, default=320)
parser.add_argument('-b', '--batch', type=int, default=16)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loader = DataLoader(get_voc(args.data, 'val', args.size), batch_size=args.batch, shuffle=False)

models = {'Teacher (FCN-ResNet50)': load_teacher(device)}
for kd, name in [('none', 'Student, no KD'), ('response', 'Student, response KD'),
                 ('feature', 'Student, feature KD')]:
    path = f'checkpoints/student_{kd}.pth'
    if os.path.exists(path):
        m = StudentNet(pretrained=False).to(device).eval()
        m.load_state_dict(torch.load(path, map_location=device))
        models[name] = m
    else:
        print(f'Skipping {name}: {path} not found')

rows, examples = [], {}
with torch.no_grad():
    for name, model in models.items():
        conf = torch.zeros(21, 21, dtype=torch.long)
        for i, (img, mask) in enumerate(loader):
            pred = model(img.to(device))['out'].argmax(1).cpu()
            conf = update_confusion(conf, pred, mask)
            if i == 0:
                examples[name] = pred[:4]
                first_img, first_mask = img[:4], mask[:4]
        params = sum(p.numel() for p in model.parameters())
        rows.append((name, 100 * mean_iou(conf), params, ms_per_image(model, device, args.size)))

print(f'\n{"Model":<26}{"mIoU":>8}{"Params":>14}{"ms/img":>10}')
for name, miou, params, ms in rows:
    print(f'{name:<26}{miou:>8.1f}{params:>14,}{ms:>10.1f}')

save_comparison(first_img, first_mask, examples, 'plots/compare.png')
print('\nSaved plots/compare.png')
