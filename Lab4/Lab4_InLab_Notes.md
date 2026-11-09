# ELEC 475 Lab 4 In-Lab: Run Steps and TA Answers

## Run it (Mac Terminal)

```bash
cd ~/Desktop/ELEC475-Group22/Lab4
source .venv/bin/activate
cd lab4_inlab
pip install matplotlib torchinfo        # plots, and the summary in models/student.py
python -m models.student                # Step 2: should say 3,265,173 parameters
python test_kd.py                       # Steps 3-4: must say 7/7 checks passed
python train.py --kd none     --iters 20 --data ../data -b 8 --workers 0
python train.py --kd response --iters 20 --data ../data -b 8 --workers 0
python train.py --kd feature  --iters 20 --data ../data -b 8 --workers 0
```

- `--data ../data` is needed because your VOC folder is in `Lab4/data`, not `Lab4/lab4_inlab/data`.
- `--workers 0` is required on a Mac. `train.py` runs at the top level of the file with no `if __name__ == "__main__":` guard, so macOS can't start DataLoader worker processes and you get a "bootstrapping phase" RuntimeError.
- `train.py` only picks CUDA or CPU, so on your Mac it runs on the CPU. `-b 8` keeps each run to a few minutes. The response and feature runs are slower because they also run the teacher.
- The handout's `download_voc.py` check isn't in your zip. If you have it, run it with the data path pointed at `../data`. Otherwise `train.py` loading without an error proves the dataset is in place.
- For the TA: screenshot the `test_kd.py` output, the three `Kick-start finished` lines, and `plots/loss_none.png`, `plots/loss_response.png`, `plots/loss_feature.png`.

## What `utils/kd.py` does

**`response_kd_loss`**
1. `hard = F.cross_entropy(z_s, y, ignore_index=255)`: the normal loss against the true labels.
2. `p_t = softmax(z_t / tau)` and `log_p_s = log_softmax(z_s / tau)`: both softened by the temperature.
3. `-(p_t * log_p_s).sum(dim=1)`: the cross-entropy at each pixel, summed over the 21 classes, giving an (N, H, W) map.
4. Keep only the pixels where `y != 255` and average them.
5. Return `alpha * hard + beta * soft`.

**`feature_kd_loss`**
1. Bilinearly resize the adapted student feature to the teacher feature's (H, W). The `high` feature is 10×10 and the teacher's `layer4` is 40×40.
2. Compute `F.cosine_similarity(..., dim=1)`, which is the cosine across channels at every pixel.
3. Return the mean of `1 - cos`.

## The three TA questions

**1. What each term of the response loss does, and why τ > 1.**
The first term, α·H(y, σ(z_s)), is ordinary cross-entropy against the ground-truth labels. It keeps the student correct.

The second term, β·H(σ(z_t/τ), σ(z_s/τ)), pushes the student's class distribution towards the teacher's. At T = 1 the teacher is very confident (e.g. 0.98 cat), so its output looks almost like the one-hot label and adds little. Dividing the logits by τ > 1 flattens the softmax, which brings out the small probabilities on wrong classes (cat 0.6, dog 0.3, car 0.01). This "dark knowledge" tells the student which classes are similar, which a one-hot label can't do.

**2. Why the adapters are needed, and why they don't count.**
The student's `low` feature has 24 channels and the teacher's `layer2` has 512. The student's `high` feature has 576 channels and the teacher's `layer4` has 2048. Cosine similarity needs vectors of the same length, so a 1×1 conv (an adapter) projects each student feature into the teacher's channel space. A 1×1 conv mixes channels at each pixel without changing the spatial layout, as in the VGG and ResNet slides.

The adapters exist only to compute the training loss. At inference the student uses only `out`, so the adapters are thrown away. They aren't part of the deployed model, so they don't count towards its parameters.

**3. What the cosine loss measures, and why it's 0 and 2.**
cos(f_s, f_t) = (f_s · f_t) / (‖f_s‖ ‖f_t‖) measures the angle between the two feature vectors at a pixel. It ignores their length, which is why scaling the teacher feature doesn't change the loss.
- Identical direction: cos = 1, so the loss is 1 − 1 = **0**.
- Opposite direction (f_s = −f_t): cos = −1, so the loss is 1 − (−1) = **2**.
- Unrelated (perpendicular): cos = 0, so the loss is 1.

So the loss ranges from 0 to 2. It asks the student to encode the same pattern as the teacher, not to match the teacher's exact values.
