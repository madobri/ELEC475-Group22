"""
ELEC 475: Lab 3 - Correction Engine
Corrects the saved test-set predictions of SnoutNet, SnoutNet-V and the mean ensemble
with a trained CorrectionNet (or CorrectionNetModified, with --modified), and saves the
corrected predictions in the same format as predict.py.
Run evaluate.py afterwards to compare them with the In-Lab results.
"""
import os
import argparse
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from models.correction import CorrectionNet, CorrectionNetModified
from utils.dataset import MEAN, STD
from utils.correction_dataset import extract_patch, HALF_PATCH
from utils.profiler import benchmark_inference
from utils.ensemble import ensemble_mean

FIRST_STAGE = ["snoutnet", "vgg16", "ensemble_mean"]
TIMING_IMAGES = 50


def load_first_stage(pred_dir: str, key: str) -> dict:
    """Loads a model's saved predictions, or builds the mean ensemble of all three models."""
    if key != "ensemble_mean":
        return torch.load(os.path.join(pred_dir, f"{key}.pt"), weights_only=True)
    members = [torch.load(os.path.join(pred_dir, f"{k}.pt"), weights_only=True)
               for k in ["snoutnet", "alexnet", "vgg16"]]
    pred = ensemble_mean([m["pred"] for m in members])
    if pred is None:
        raise SystemExit("ensemble_mean() in utils/ensemble.py is not implemented yet.")
    return dict(members[0], pred=pred, params=sum(m["params"] for m in members),
                ms_per_image=sum(m["ms_per_image"] for m in members))


def correct_predictions(checkpoint_path: str, data_dir: str = "data/oxford-iiit-pet-noses",
                        pred_dir: str = "predictions", modified: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_dir = os.path.join(data_dir, "images-original", "images")
    tag = os.path.splitext(os.path.basename(checkpoint_path))[0]          # e.g. correction_aug
    print("=" * 72)
    print("ELEC 475: Lab 3 Prediction Correction")
    print(f"Loading checkpoint : {checkpoint_path}")
    print(f"Device             : {device}")
    print("=" * 72 + "\n")

    # 1. Model & Checkpoint Loading
    model = CorrectionNetModified() if modified else CorrectionNet()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model = model.to(device)
    model.eval()
    corr_params = sum(p.numel() for p in model.parameters())

    for key in FIRST_STAGE:
        data = load_first_stage(pred_dir, key)

        # 2. Patch centred on each first-stage prediction -> predicted offset -> corrected position
        centres = data["pred"].round()
        patches = torch.stack([
            TF.normalize(TF.to_tensor(extract_patch(
                Image.open(os.path.join(img_dir, name)).convert("RGB"), (int(c[0]), int(c[1])))), MEAN, STD)
            for name, c in zip(data["names"], centres)])
        with torch.no_grad():
            offsets = torch.cat([model(batch.to(device)).cpu() for batch in patches.split(64)])
        corrected = centres + offsets * HALF_PATCH

        # 3. Inference Timing & Save (cost includes both stages)
        ms = benchmark_inference(model, list(patches[:TIMING_IMAGES]), device)
        save_path = os.path.join(pred_dir, f"{key}_{tag}.pt")
        torch.save(dict(data, pred=corrected, params=data["params"] + corr_params,
                        ms_per_image=data["ms_per_image"] + ms), save_path)
        print(f"[{key}] Correction time: {ms:.1f} ms/image (batch size 1)")
        print(f"[{key}] Saved {save_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ELEC 475: Lab 3 Prediction Correction")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the CorrectionNet .pth checkpoint")
    parser.add_argument("--modified", action="store_true", help="The checkpoint is a CorrectionNetModified (Step 5)")
    parser.add_argument("--data-dir", type=str, default="data/oxford-iiit-pet-noses",
                        help="Directory where the dataset is stored")
    parser.add_argument("--pred-dir", type=str, default="predictions",
                        help="Directory containing the predictions files from predict.py")
    args = parser.parse_args()

    correct_predictions(checkpoint_path=args.checkpoint, data_dir=args.data_dir, pred_dir=args.pred_dir,
                        modified=args.modified)
