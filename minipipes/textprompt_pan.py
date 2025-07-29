import os
import argparse

from copy import deepcopy
import numpy as np

import torch
from segment_anything import sam_model_registry

from extensions.text_prompt import TextPromptEncoder, MedSAMText,TextPromptDemo

from minipipes.process_img import get_image, segment_image, save_image, create_overlay

def get_model():
    medsam_ckpt_path = "work_dir/MedSAM/medsam_vit_b.pth"
    medsam_text_demo_checkpoint = "work_dir/MedSAM/medsam_text_prompt_flare22.pth"
    device = "cuda:0"

    medsam_model = sam_model_registry['vit_b'](checkpoint=medsam_ckpt_path)
    text_prompt_encoder = TextPromptEncoder(
        embed_dim = 256,
        image_embedding_size = (64, 64),
        input_image_size = (1024, 1024),                                                            
        mask_in_chans = 1
    )
    medsam_text_demo = MedSAMText(
        image_encoder=deepcopy(medsam_model.image_encoder),
        mask_decoder=deepcopy(medsam_model.mask_decoder),
        prompt_encoder=text_prompt_encoder,
        device = device
    )
    medsam_text_demo_weights = torch.load(medsam_text_demo_checkpoint)
    for key in medsam_text_demo.state_dict().keys():
        if not key.startswith('prompt_encoder.text_encoder.'):
            medsam_text_demo.state_dict()[key].copy_(medsam_text_demo_weights[key])
    medsam_text_demo = medsam_text_demo.to(device)
    medsam_text_demo.eval()

    model_demo = TextPromptDemo(medsam_text_demo)

    return model_demo

def main(args):
    args.output_dir = os.path.join(args.output_dir, f"norm_{args.norm_mode}")
    if args.norm_mode == "window":
        args.output_dir += f"_minwin_{args.window_min_percentile}_maxwin_{args.window_max_percentile}"
    if args.use_otsu:
        args.output_dir += "_otsu"
    if args.keep_largest_only:
        args.output_dir += "_largestseg"
    if args.fill_holes:
        args.output_dir += "_fillholes"

    prompts = args.prompts.split(",")

    for p in prompts:
        os.makedirs(os.path.join(args.output_dir, p), exist_ok=True)
        if args.save_overlay and not args.save_composed:
            os.makedirs(os.path.join(args.output_dir, p, "overlay"), exist_ok=True)

    if args.save_composed:
        os.makedirs(os.path.join(args.output_dir, "composed"), exist_ok=True)
        if args.save_overlay:
            os.makedirs(os.path.join(args.output_dir, "overlay"), exist_ok=True)

    # Load the model
    model_demo = get_model()
    
    nii_files = [f for f in os.listdir(args.input_dir) if f.endswith('.nii.gz')]
    for fname in nii_files:
        input_path = os.path.join(args.input_dir, fname)
        img_data = get_image(input_path, norm_type=args.norm_mode, window_min_percentile=args.window_min_percentile, window_max_percentile=args.window_max_percentile)

        if args.save_composed:
            composed_mask = np.zeros_like(img_data, dtype=np.uint8)

        for prompt in reversed(prompts): #reversed, so that we can have the last prompt on top in the composed mask
            seg = segment_image(img_data, model_demo, text_prompt=prompt, use_otsu=args.use_otsu, 
                              keep_largest_only=args.keep_largest_only, fill_holes=args.fill_holes)
            save_image(seg, os.path.join(args.output_dir, prompt, fname))

            if args.save_overlay and not args.save_composed:
                overlay_volume_uint8 = create_overlay(img_data, seg, alpha=0.5)
                save_image(overlay_volume_uint8, os.path.join(args.output_dir, prompt, "overlay", fname.replace('.nii.gz', '.tif')), is_RGB=True)
            
            if args.save_composed:
                composed_mask[seg > 0] = model_demo.caption_label_dict[prompt]

        if args.save_composed:
            save_image(composed_mask, os.path.join(args.output_dir, "composed", fname))

            if args.save_overlay:
                overlay_volume_uint8 = create_overlay(img_data, composed_mask if args.save_composed else seg, alpha=0.5)
                save_image(overlay_volume_uint8, os.path.join(args.output_dir, "overlay", fname.replace('.nii.gz', '.tif')), is_RGB=True)            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch segment .nii.gz files in a folder.")
    parser.add_argument("--input_dir", default="/group/glastonbury/soumick/dataset/ukbbnii/minisets/F20204_Liver_Imaging_T1_ShMoLLI_DICOM_H5v3/exported_images/NIFTI/", help="Folder containing input .nii.gz files")
    parser.add_argument("--output_dir", default="/group/glastonbury/soumick/dataset/ukbbnii/minisets/F20204_Liver_Imaging_T1_ShMoLLI_DICOM_H5v3/segmentations/MedSAM_textprompt", help="Folder to save segmentation outputs [a folder with the params will be created inside]")

    parser.add_argument("--norm_mode", type=str, default="window", choices=["minmax", "window"], help="Normalisation mode for input images")
    parser.add_argument("--window_min_percentile", type=float, default=1, help="Minimum percentile for windowing [To be used only for norm_mode 'window']")
    parser.add_argument("--window_max_percentile", type=float, default=99, help="Maximum percentile for windowing [To be used only for norm_mode 'window']")

    parser.add_argument("--use_otsu", action=argparse.BooleanOptionalAction, default=True, help="Use Otsu's method for thresholding the segmentation output (not done in the original work)")
    parser.add_argument("--keep_largest_only", action=argparse.BooleanOptionalAction, default=True, help="Keep only the largest connected component in each slice")
    parser.add_argument("--fill_holes", action=argparse.BooleanOptionalAction, default=True, help="Fill holes within the segmented regions")
    parser.add_argument("--prompts", type=str, default="pancreas,liver,right kidney,left kidney,spleen,stomach,gallbladder", help="Comma-separated text prompts for segmentation, in the order of importance. Default is 'liver'.")

    parser.add_argument("--save_composed", action=argparse.BooleanOptionalAction, default=True, help="Whether to save composed segmentation.")
    parser.add_argument("--save_overlay", action=argparse.BooleanOptionalAction, default=True, help="Whether to save overlay images.")

    args = parser.parse_args()
    main(args)