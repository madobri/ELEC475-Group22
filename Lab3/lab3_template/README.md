# ELEC 475: Lab 3 Starter Template
## Pet Nose Localization: SnoutNet, AlexNet and VGG16

Welcome to ELEC 475 Lab 3! This repository contains the starter code and scaffolding for your **Pre-Lab**. You will implement a custom dataset for the oxford-iiit-pet-noses dataset and three nose localization models, then verify them against the provided trained weights.

### Directory Structure

```text
lab3_template/
├── checkpoints/           # Provided trained weights (snoutnet.pth, alexnet.pth, vgg16.pth)
├── data/                  # oxford-iiit-pet-noses dataset
├── models/
│   └── snoutnet.py        # TODO [Pre-Lab Task 4]: SnoutNet, SnoutNetAlexNet, SnoutNetVGG16
├── utils/
│   └── dataset.py         # TODO [Pre-Lab Task 3]: PetNoseDataset custom dataset
├── plots/                 # Generated figures (dataset reality check)
└── requirements.txt       # Python package dependencies (torch, torchvision, torchinfo)
```

---

### Step-by-Step Execution Guide

#### 1. Setup Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell (or source .venv/bin/activate on Linux/WSL/macOS)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130   # NVIDIA GPU only
pip install -r requirements.txt
```

#### 2. Download the Dataset & Trained Weights
- Extract the oxford-iiit-pet-noses dataset into `data/oxford-iiit-pet-noses/`.
- Place `snoutnet.pth`, `alexnet.pth` and `vgg16.pth` into `checkpoints/`.

#### 3. Pre-Lab Implementation
- Open `utils/dataset.py` and implement the `TODO [PRE-LAB TASK 3]` sections.
- Verify the dataset: `python -m utils.dataset`
- Write `utils/check_dataset.py` and confirm the labels sit on the noses:
  ```powershell
  python -m utils.check_dataset
  ```
- Open `models/snoutnet.py` and implement the `TODO [PRE-LAB TASK 4]` sections.
- Verify the architectures and load the provided weights:
  ```powershell
  python -m models.snoutnet
  ```
- Complete the readiness checklist in the Pre-Lab document before the in-lab session!
