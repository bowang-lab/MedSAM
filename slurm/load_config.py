#!/usr/bin/env python3
"""
Helper script to load and process YAML config files for exp_yolo.slurm
Outputs bash array format for easy sourcing.
"""
import sys
import yaml
import json
from pathlib import Path

def load_config(mode: int, repo_dir: str, **kwargs) -> dict:
    """Load config for a given mode and substitute template variables."""
    config_path = Path(__file__).parent / "configs" / f"mode{mode}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Template variables
    template_vars = {
        "REPO_DIR": repo_dir,
        **kwargs
    }
    
    # Process paths with template substitution
    if "paths" in config:
        processed_paths = {}
        for key, value in config["paths"].items():
            if isinstance(value, str) and value.endswith("_template"):
                # This is a template path
                actual_key = key.replace("_template", "")
                template_str = value
                
                # Substitute variables
                for var, replacement in template_vars.items():
                    template_str = template_str.replace(f"{{{var}}}", str(replacement))
                
                # Handle nested templates (e.g., PROCESSED_DIR in images_parquet_template)
                if "PROCESSED_DIR" in template_str and "processed_dir_template" in config["paths"]:
                    processed_dir = config["paths"]["processed_dir_template"]
                    for var, replacement in template_vars.items():
                        processed_dir = processed_dir.replace(f"{{{var}}}", str(replacement))
                    template_str = template_str.replace("{PROCESSED_DIR}", processed_dir)
                
                processed_paths[actual_key] = template_str
            else:
                processed_paths[key] = value
        config["paths"] = processed_paths
    
    return config

def build_args(config: dict, yolo_devices: str, **overrides) -> list:
    """Build argument list from config."""
    args = []
    
    # Add actions
    if "actions" in config:
        args.extend(config["actions"])
    
    # Add paths (apply overrides if provided)
    if "paths" in config:
        paths = config["paths"].copy()  # Don't modify original
        
        # Apply path overrides from CLI/environment
        path_overrides = {
            "yolo_weights": overrides.get("yolo_weights"),
            "yolo_ds": overrides.get("yolo_ds"),
            "medsam_ckpt": overrides.get("medsam_ckpt"),
            "predict_parquet": overrides.get("predict_parquet"),
            "predict_out_dir": overrides.get("predict_out_dir"),
            "predict_splits": overrides.get("predict_splits"),
        }
        for key, value in path_overrides.items():
            if value is not None:
                paths[key] = value
        
        # Map path keys to argument names
        path_arg_map = {
            "splits_parquet": "--splits-parquet",
            "out_dir": "--yolo-out-dir",
            "images_parquet": "--images-parquet",
            "init_weights": "--init-weights",
            "runs_root": "--finetune-runs-root",
            "yolo_weights": "--yolo-weights",
            "yolo_ds": "--yolo-ds",
            "medsam_ckpt": "--medsam-ckpt",
            "predict_parquet": "--predict-parquet",
            "predict_out_dir": "--predict-out-dir",
        }
        
        for path_key, arg_name in path_arg_map.items():
            if path_key in paths and paths[path_key]:
                args.append(arg_name)
                args.append(str(paths[path_key]))
    
    # Add training parameters
    if "training" in config:
        training = config["training"]
        if "epochs" in training:
            args.extend(["--epochs", str(overrides.get("epochs", training["epochs"]))])
        if "batch" in training:
            args.extend(["--batch", str(overrides.get("batch", training["batch"]))])
        if "imgsz" in training:
            args.extend(["--imgsz", str(overrides.get("imgsz", training["imgsz"]))])
        if "workers" in training:
            args.extend(["--workers", str(overrides.get("workers", training["workers"]))])
    
    # Add prediction parameters
    if "prediction" in config:
        pred = config["prediction"]
        if "splits" in pred:
            args.extend(["--predict-splits", str(overrides.get("predict_splits", pred["splits"]))])
        if "batch_size" in pred:
            args.extend(["--predict-batch-size", str(overrides.get("predict_batch_size", pred["batch_size"]))])
        if pred.get("sam_amp", False):
            args.append("--predict-sam-amp")
        if pred.get("resume", False):
            args.append("--predict-resume")
    
    # Add device
    args.extend(["--yolo-device", yolo_devices])
    
    # Add special args
    if "args" in config:
        # Handle special cases
        if "--splits-batch-size" in config["args"]:
            args.extend(["--splits-batch-size", "2048"])
    
    return args

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: load_config.py <mode> <repo_dir> <yolo_devices> [key=value ...]", file=sys.stderr)
        sys.exit(1)
    
    mode = int(sys.argv[1])
    repo_dir = sys.argv[2]
    yolo_devices = sys.argv[3]
    
    # Parse key=value pairs
    overrides = {}
    for arg in sys.argv[4:]:
        if "=" in arg:
            key, value = arg.split("=", 1)
            # Try to convert to int/float if possible
            try:
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass
            overrides[key] = value
    
    try:
        # Extract template variables from overrides
        kwargs = {}
        for key in ["CONF_PCT", "DUP_FACTOR"]:
            if key in overrides:
                kwargs[key] = overrides.pop(key)
        
        config = load_config(mode, repo_dir, **kwargs)
        
        # Apply path overrides to config for output
        paths = config.get("paths", {}).copy()
        path_overrides = {
            "yolo_weights": overrides.get("yolo_weights"),
            "yolo_ds": overrides.get("yolo_ds"),
            "medsam_ckpt": overrides.get("medsam_ckpt"),
            "predict_parquet": overrides.get("predict_parquet"),
            "predict_out_dir": overrides.get("predict_out_dir"),
        }
        for key, value in path_overrides.items():
            if value is not None:
                paths[key] = value
        
        args = build_args(config, yolo_devices, **overrides)
        
        # Output both args and paths as JSON
        output = {
            "args": args,
            "paths": paths
        }
        print(json.dumps(output))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
