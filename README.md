# PASTRY: Physics-Aware Super-resolution for hyperspecTRal imagerY

PASTRY is a PyTorch-based framework for Hyperspectral Image (HSI) Super-Resolution. It introduces a Neurosymbolic approach to super-resolve low-resolution hyperspectral images while maintaining strict physical plausibility through unmixing principles, specifically enforcing Abundance Non-negativity (ANC) and Abundance Sum-to-one (ASC) constraints.

## ✨ Features

* **Two-Stage Neurosymbolic Architecture**: 
  * *Stage 1 (Unmixing)*: Learns a spatial symbol library (endmembers) from the data.
  * *Stage 2 (Super-Resolution)*: Utilizes the learned spectral library to reconstruct high-resolution hyperspectral images from low-resolution inputs.
* **Physics-Aware Constraints**: Enforces physical rules governing light mixtures (ANC/ASC constraints).
* **Multiple Baselines**: Includes implementations for Neurosymbolic (`Nesy`), Convolutional (`Conv`), and `Bicubic` super-resolution architectures.
* **Extensive Analysis Tools**: Out-of-the-box tools for stability analysis, $K$-sensitivity (impact of spectral library size), constraint ablations, and intrinsic dimensionality estimation via HySime.
* **WandB Integration**: Fully integrated with Weights & Biases for experiment tracking and visual dashboard generation.

---

## ⚙️ Installation

1. Clone the repository:
```bash
git clone <your-repository-url> pastry
cd pastry
```

2. Create and activate the Conda environment using the provided `env.yaml` file:

```bash
conda env create -f env.yaml
conda activate pastry

```



---

## 🗂️ Data Download & Preparation

The framework expects datasets to be placed in the `data/benchmark_full/` directory by default (as specified in the experiment configs).

### 1. Pavia Center (`paviac`)

Pavia Center is an image acquired by the ROSIS sensor over Pavia, Italy.

* **Download**: Visit the [EHU Hyperspectral Remote Sensing Scenes page](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes).
* Scroll to the "Pavia Centre" section.
* Download the image file (`Pavia.mat`) and the ground truth file (`Pavia_gt.mat`).
* Move them into the data directory: `data/benchmark_full/Pavia.mat` and `data/benchmark_full/Pavia_gt.mat`.

### 2. Chikusei (`chikusei`)

Chikusei is an airborne hyperspectral dataset taken over Chikusei, Ibaraki, Japan.

* **Download**: Visit Dr. Naoto Yokoya's [Download Page](https://naotoyokoya.com/Download.html).
* Navigate to the "Chikusei" dataset section and fill out any required request forms if prompted, or download the `.mat` file directly.
* Move the downloaded file into the data directory and ensure it is named `HyperspecVNIR_Chikusei_20140729.mat`.

### Expected Directory Structure

```text
pastry/
├── data/
│   └── benchmark_full/
│       ├── HyperspecVNIR_Chikusei_20140729.mat
│       ├── Pavia.mat
│       └── Pavia_gt.mat
├── experiments/
├── src/
└── ...

```

---

## 🚀 Running Experiments

Experiments are configured via YAML files located in the `experiments/` directory.

To run a super-resolution experiment, use the `src.entrypoint` module and pass your target configuration file:

```bash
# Run Pavia Center with a Scale Factor of 4
python -m src.entrypoint --cfg experiments/paviac_4.yaml

# Run Chikusei with a Scale Factor of 2
python -m src.entrypoint --cfg experiments/chikusei_2.yaml

```

### Configuration Structure (`.yaml`)

A standard experiment configuration includes data paths, network parameters (scale factor, number of endmember symbols), and trainer hyperparameters.

```yaml
data:
  path: "data/benchmark_full/Pavia.mat"
  name: paviac
  batch_size: 32

net:
  name: Nesy
  params:
    scale_factor: 4
    num_symbols: 10
  ckpt_path: "output/models/paviac10.pth"

trainer:
  epochs: 200
  lr: 0.0005
  device: cuda

```

---

## 📊 Evaluation & Analysis

PASTRY contains several built-in routines to evaluate the physical validity and robustness of the model. These can be executed by invoking methods on the `Experiment` class directly:

* **Endmember Extraction Validation**: Matches learned spatial symbols to ground truth classes using Hungarian matching (`src/validation/endmembers.py`).
* **K-Sensitivity Analysis**: Evaluates how the number of endmembers ($K$) impacts Super-Resolution PSNR/SAM.
* **Constraint Ablation**: Tests the network without physical constraints, with only ANC (ReLU), with only ASC (Hyperplane projection), and with both (Softmax).
* **HySime / K-Means Elbow Analysis**: Estimates the optimal intrinsic dimensionality of the dataset before unmixing.

---

## 📁 Codebase Layout

* `src/architectures/`: Core PyTorch modules (`superresolution`, `unmixing`, `constraint.py`).
* `src/data/`: Data loading pipelines and patching logic for standard HSI layouts.
* `src/training/`: Training loops for Stage-1 (Library Unmixing) and Stage-2 (Super-Resolution).
* `src/validation/`: SAM calculation and automated endmember-to-GT class matching.
* `src/viz/`: Extensive plotting utilities for saving WandB-compatible spectral signature plots and abundance maps.
* `experiments/`: YAML configuration registry.
