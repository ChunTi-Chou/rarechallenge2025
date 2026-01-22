# RARE 2025 Challenge - MICCAI

This repository contains the codebase for the [RARE 2025 Challenge](https://rare25.grand-challenge.org/) (MICCAI) participation. The project is focused on the classification of endoscopic images, specifically targeting Barrett's Esophagus.

## Project Overview

The solution leverages deep learning models (ConvNeXt, Swin Transformer, etc.) with support for parameter-efficient fine-tuning (LoRA). The framework is built using:
*   [PyTorch Lightning](https://lightning.ai/) for training loop management.
*   [Hydra](https://hydra.cc/) for configuration management.
*   [Albumentations](https://albumentations.ai/) for image augmentation.
*   [Timm](https://timm.fast.ai/) for model backbones.

## Installation

### Prerequisites
*   Python 3.9+
*   Docker (for inference container verification)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd rare2025challenge
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## Training

The training pipeline fetches the dataset automatically from HuggingFace (`TimJaspersTue/RARE25-train`), so manual data downloading is not required.

To run an experiment using the default configuration:

```bash
python run_experiment.py
```

### Configuration

The project uses Hydra for configuration. You can override parameters from the command line.

**Examples:**

*   **Specify a model architecture:**
    ```bash
    python run_experiment.py model=swin_v2_t
    ```

*   **Change batch size and max epochs:**
    ```bash
    python run_experiment.py dataset.batch_size=32 training.max_epochs=50
    ```

*   **Run with specific config file:**
    Config files are located in `configs/`.
    ```bash
    python run_experiment.py --config-name=convnext_small
    ```

## Inference & Submission

The inference logic is encapsulated in `inference.py` and is designed to run within a Docker container, adhering to the Grand Challenge submission format.

### Local Testing

To verify the inference locally using Docker:

1.  **Build the container:**
    ```bash
    ./do_build.sh
    ```

2.  **Run the test:**
    This script mounts `test/input/` into the container and writes to `test/output/`.
    ```bash
    ./do_test_run.sh
    ```

### Exporting for Submission

To prepare the container image for upload:

```bash
./do_save.sh
```

## Project Structure

*   `src/`: Contains source code for datasets, models, metrics, and losses.
*   `configs/`: Hydra configuration files for different experiments (models, training params).
*   `model/`: Custom model definitions.
*   `resources/`: Stores model checkpoints and configurations used during inference inside the Docker container.
*   `run_experiment.py`: Main entry point for training models.
*   `inference.py`: Entry point for the Docker container inference.
