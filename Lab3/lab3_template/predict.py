"""
ELEC 475: Lab 3 - Test-Set Prediction Engine
Runs each trained model over the test set once and saves its predictions, so that
evaluate.py never needs to run a model.

For each model:
  1. Builds the model with your models/snoutnet.py and loads checkpoints/<model>.pth
  2. Checks that your PetNoseDataset labels match test_noses.txt
  3. Predicts every test image and converts the result to original-image pixels
  4. Measures the average inference time per image (batch size 1)
  5. Saves predictions/<model>.pt
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader
from models import get_model
from utils.dataset import PetNoseDataset, IMG_SIZE
from utils.profiler import benchmark_inference

MODEL_KEYS = ["snoutnet", "alexnet", "vgg16"]
TIMING_IMAGES = 50


def read_test_labels(data_dir: str):
    """Parses test_noses.txt directly. Returns (names, (N, 2) tensor of (u, v))."""
    names, coords = [], []
    with open(os.path.join(data_dir, "test_noses.txt")) as f:
        for line in f:
            if not line.strip():
                continue
            name, uv = line.split(",", 1)
            u, v = uv.strip().strip('"').strip("()").split(",")
            names.append(name.strip())
            coords.append([float(u), float(v)])
    return names, torch.tensor(coords)


def check_labels(gt: torch.Tensor, file_coords: torch.Tensor, names):
    """Stops with a clear message if the dataset's labels disagree with test_noses.txt."""
    if gt.shape != file_coords.shape:
        raise SystemExit(f"Your PetNoseDataset(split='test') has {gt.shape[0]} samples, "
                         f"but test_noses.txt has {file_coords.shape[0]}.")
    bad = ((gt - file_coords).abs().max(dim=1).values > 0.5).nonzero()
    if len(bad) > 0:
        i = bad[0].item()
        raise SystemExit(f"Label mismatch at test image {i} ({names[i]}): your dataset gives "
                         f"({gt[i, 0]:.1f}, {gt[i, 1]:.1f}), test_noses.txt gives "
                         f"({file_coords[i, 0]:.0f}, {file_coords[i, 1]:.0f}). "
                         f"Check your PetNoseDataset label parsing and rescaling.")


def predict_model(model_key: str, dataset: PetNoseDataset, names, file_coords: torch.Tensor,
                  checkpoint_dir: str, pred_dir: str, device: torch.device):
    # 1. Model & Checkpoint Loading
    model = get_model(model_key, pretrained=False)
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_key}.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=True)
    model = model.to(device)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"[{model_key}] Loaded {checkpoint_path} ({params:,} params)")

    # 2. Prediction Loop (outputs are normalized: (u/227, v/227))
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    preds, gts, sizes, timing_inputs = [], [], [], []
    with torch.no_grad():
        for imgs, labels, orig_size in loader:
            outputs = model(imgs.to(device)).float().cpu()
            preds.append(outputs * orig_size)                 # original-image pixels
            gts.append(labels / IMG_SIZE * orig_size)
            sizes.append(orig_size)
            if len(timing_inputs) < TIMING_IMAGES:
                timing_inputs.extend(imgs[:TIMING_IMAGES - len(timing_inputs)])
    pred, gt, orig_size = torch.cat(preds), torch.cat(gts), torch.cat(sizes)

    # 3. Label Verification
    check_labels(gt, file_coords, names)
    print(f"[{model_key}] Labels match test_noses.txt ({len(names)} images)")

    # 4. Inference Timing
    ms = benchmark_inference(model, timing_inputs, device)
    print(f"[{model_key}] Inference time: {ms:.1f} ms/image (batch size 1)")

    # 5. Save Predictions
    os.makedirs(pred_dir, exist_ok=True)
    save_path = os.path.join(pred_dir, f"{model_key}.pt")
    torch.save(dict(names=names, pred=pred, gt=gt, orig_size=orig_size,
                    params=params, ms_per_image=ms), save_path)
    print(f"[{model_key}] Saved {save_path}\n")


def main():
    parser = argparse.ArgumentParser(description="ELEC 475: Lab 3 Test-Set Prediction")
    parser.add_argument("--model", type=str, default="all", choices=["all"] + MODEL_KEYS,
                        help="Model to run: 'snoutnet', 'alexnet', 'vgg16' or 'all'")
    parser.add_argument("--data-dir", type=str, default="data/oxford-iiit-pet-noses",
                        help="Directory where the dataset is stored")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory containing the provided weights")
    parser.add_argument("--pred-dir", type=str, default="predictions",
                        help="Directory to save the predictions files")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_keys = MODEL_KEYS if args.model == "all" else [args.model]
    print("=" * 72)
    print("ELEC 475: Lab 3 Test-Set Prediction")
    print(f"Device        : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Models        : {', '.join(k.upper() for k in model_keys)}")
    print("=" * 72 + "\n")

    dataset = PetNoseDataset(root=args.data_dir, split="test")
    names, file_coords = read_test_labels(args.data_dir)
    for model_key in model_keys:
        predict_model(model_key, dataset, names, file_coords, args.checkpoint_dir, args.pred_dir, device)


if __name__ == "__main__":
    main()
