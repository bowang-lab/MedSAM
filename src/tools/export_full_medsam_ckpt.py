# tools/export_full_medsam_ckpt.py
import torch
from pathlib import Path
from segment_anything import sam_model_registry

def export_full(base_ckpt: str, finetuned_pth: str, out_ckpt: str, sam_type: str = "vit_b"):
    # 1) load base
    sam = sam_model_registry[sam_type](checkpoint=base_ckpt)

    # 2) load finetuned parts
    ft = torch.load(finetuned_pth, map_location="cpu")
    sam.mask_decoder.load_state_dict(ft["mask_decoder"], strict=True)

    # 3) if LoRA adapters were trained, try to merge them in
    if ft.get("using_lora", False) and "image_encoder" in ft:
        try:
            # If your training used PEFT adapters, you must wrap the encoder first
            from peft import LoraConfig, get_peft_model
            # minimal generic wrap; if you had a more specific target_modules list, reuse it
            lora_targets = [n for n, m in sam.image_encoder.named_modules() if isinstance(m, torch.nn.Linear)]
            cfg = LoraConfig(r=8, lora_alpha=16, target_modules=lora_targets, lora_dropout=0.0, bias="none", task_type="FEATURE_EXTRACTION")
            sam.image_encoder = get_peft_model(sam.image_encoder, cfg)
            sam.image_encoder.load_state_dict(ft["image_encoder"], strict=False)

            # merge adapters into the base weights if available
            if hasattr(sam.image_encoder, "merge_and_unload"):
                sam.image_encoder.merge_and_unload()
        except Exception as e:
            print(f"[WARN] LoRA adapters present but could not be merged: {e!r}. Continuing without LoRA.")
            # If this happens, your exported model will include the fine-tuned decoder but not LoRA changes.

    # 4) save a single checkpoint compatible with your existing loader
    torch.save(sam.state_dict(), out_ckpt)
    print(f"[OK] Exported merged checkpoint → {out_ckpt}")

if __name__ == "__main__":
    # Example:
    # python tools/export_full_medsam_ckpt.py \
    #   /path/to/medsam_vit_b.pth \
    #   /path/to/runs/.../weights/best.pth \
    #   /path/to/medsam_full_finetuned.pth
    import sys
    export_full(sys.argv[1], sys.argv[2], sys.argv[3])