# SLURM Configuration Files

This directory contains YAML configuration files for each mode of `exp_yolo.slurm`.

## Structure

Each mode has its own config file:
- `mode1.yaml` - Standard training from raw folders
- `mode2.yaml` - Training from parquet splits
- `mode3.yaml` - Semi-supervised YOLO finetuning
- `mode4.yaml` - Full semi-supervised pipeline (YOLO + MedSAM)
- `mode5.yaml` - Predictions only (standalone)
- `mode6.yaml` - Training + predictions pipeline
- `mode7.yaml` - Semi-supervised + predictions pipeline
- `mode8.yaml` - MedSAM finetuning only

## Configuration Format

Each config file contains:

### `paths`
Path configurations. Can use template variables:
- `{REPO_DIR}` - Repository directory
- `{CONF_PCT}` - Confidence percentage (for modes 3, 4, 7)
- `{DUP_FACTOR}` - Duplication factor (for modes 3, 4, 7)
- `{PROCESSED_DIR}` - Processed directory (nested template)

### `actions`
List of action flags (e.g., `--train-yolo`, `--finetune-medsam`)

### `training`
Training hyperparameters:
- `epochs` - Number of training epochs
- `batch` - Batch size
- `imgsz` - Image size
- `workers` - Number of data workers

### `prediction`
Prediction settings:
- `splits` - Comma-separated splits to predict on
- `batch_size` - Batch size for predictions
- `sam_amp` - Enable AMP for MedSAM
- `resume` - Resume interrupted runs

## Editing Configs

To modify paths or parameters for a mode, simply edit the corresponding YAML file:

```yaml
# Example: mode8.yaml
paths:
  yolo_weights: "/path/to/your/weights.pt"
  yolo_ds: "/path/to/your/dataset"
  medsam_ckpt: "/path/to/medsam.pth"

training:
  epochs: 100  # Change from default 50
  batch: 16    # Change from default 8
```

## Overriding at Runtime

You can still override config values via CLI arguments:

```bash
# Override epochs
sbatch exp_yolo.slurm 8 --epochs=100

# Override paths
sbatch exp_yolo.slurm 8 --yolo-weights=/new/path.pt --yolo-ds=/new/dataset
```

CLI arguments take precedence over config file values.

