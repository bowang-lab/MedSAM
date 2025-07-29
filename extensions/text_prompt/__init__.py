import numpy as np
import cv2
import matplotlib.pyplot as plt

from ipywidgets import interact, widgets, FileUpload, IntSlider, VBox, HBox
from IPython.display import display, clear_output

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import CLIPTextModel, CLIPTokenizer 
from segment_anything.modeling import PromptEncoder

#%% Define text prompt encoder
# They used the pre-trained CLIP model as the text encoder. The prompt encoder in SAM maps all kinds of prompts to the same dimension (256) but the output dimension of CLIP model is 512. Thus, they added an additional projection layer `self.text_encoder_head = nn.Linear(512, embed_dim)` to the text encoder to align the dimension.

class TextPromptEncoder(PromptEncoder):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int = 1,
        activation = nn.GELU,
        ) -> None:
        super().__init__(embed_dim, image_embedding_size, input_image_size, mask_in_chans, activation)
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
        text_encoder.requires_grad_(False)
        self.text_encoder = text_encoder
        self.text_encoder_head = nn.Linear(512, embed_dim)

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        tokens: Optional[torch.Tensor],
    ):
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks, tokens)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        if tokens is not None:
            encoder_hidden_states = self.text_encoder(tokens)[0]
            text_embeddings = self.text_encoder_head(encoder_hidden_states)
            sparse_embeddings = torch.cat([sparse_embeddings, text_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings

    def _get_batch_size(self, points, boxes, masks, tokens):
        """
        Returns the batch size of the inputs.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        elif tokens is not None:
            return tokens.shape[0]
        else:
            return 1
        
#%% Define the whole model

# The whole model architecture is the same as the bounding box-based version. The only difference is that the prompt encoder was changed to text encoder.
# Since MedSAM already provided an image encoder trained on a large scale medical image datasets, they can freeze it and only fine-tune the mask decoder during training.

class MedSAMText(nn.Module):
    def __init__(self,
                image_encoder,
                mask_decoder,
                prompt_encoder,
                device,
                ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.device = device

    def forward(self, image, tokens):
        image_embedding = self.image_encoder(image) # (B, 256, 64, 64)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            tokens=tokens
        )
        low_res_logits, _ = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          ) # (B, 1, 256, 256)

        return low_res_logits
    

#%% Text prompt demo GUI

class TextPromptDemo:
    def __init__(self, model):
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        self.model = model
        self.model.eval()
        self.image = None  # Can be 2D (H, W) or 3D (D, H, W)
        self.is_volume = False
        self.image_embeddings = None  # For 3D: list of embeddings per slice
        self.img_size = None
        self.img_name = None
        self.gt = None
        self.current_slice = 0
        self.segmentations = {}  # Store segmentations for each prompt and slice
        self.current_prompt = ""
        
        self.label_dict = {
            1: ["liver"],
            2: ["right kidney"],
            3: ["spleen"],
            4: ["pancreas"],
            5: ["aorta"],
            6: ["inferior vena cava", "ivc"],
            7: ["right adrenal gland", "rag"],
            8: ["left adrenal gland", "lag"],
            9: ["gallbladder"],
            10: ["esophagus"],
            11: ["stomach"],
            12: ["duodenum"],
            13: ["left kidney"]
        }
        self.caption_label_dict = {}
        for label_id, label_list in self.label_dict.items():
            for label in label_list:
                self.caption_label_dict[label] = label_id

        avail_prompts = []
        for v in self.label_dict.values():
            avail_prompts += v
        self.avail_prompts = ", ".join(avail_prompts)

    def show_mask(self, mask, ax, random_color=False, alpha=0.95):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
        else:
            color = np.array([251/255, 252/255, 30/255, alpha])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    @torch.no_grad()
    def infer(self, text, slice_idx=None, return_raw=False):
        """
        Perform inference on a single slice or current slice
        
        Args:
            text: Text prompt for segmentation
            slice_idx: Specific slice index for 3D volumes, None for 2D or current slice
        """
        if self.is_volume:
            if slice_idx is None:
                slice_idx = self.current_slice
            image_embeddings = self.image_embeddings[slice_idx]
            img_size = self.img_size[1:]  # Remove depth dimension
        else:
            image_embeddings = self.image_embeddings
            img_size = self.img_size
            
        tokens = self.tokenize_text(text).to(self.model.device)
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points = None,
            boxes = None,
            masks = None,
            tokens = tokens
        )
        low_res_logits, _ = self.model.mask_decoder(
            image_embeddings=image_embeddings, # (B, 256, 64, 64)
            image_pe=self.model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
            )
        low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
        low_res_pred = F.interpolate(
            low_res_pred,
            size = img_size,
            mode = 'bilinear',
            align_corners = False
        )
        low_res_pred = low_res_pred.detach().cpu().numpy().squeeze()
        seg = np.uint8(low_res_pred > 0.5)

        if return_raw:
            return seg, low_res_pred
        else:
            return seg

    def show(self, fig_size=5, alpha=0.95):
        """
        Display the interactive segmentation interface.
        For 2D images: shows single image with text prompt
        For 3D volumes: shows image with text prompt and slice slider
        """
        assert self.image is not None, "Please set image first."
        
        print("Possible prompts:", self.avail_prompts)
        
        if self.is_volume:
            self._show_volume(fig_size, alpha)
        else:
            self._show_2d(fig_size, alpha)
    
    def _show_2d(self, fig_size=5, alpha=0.95):
        """Original 2D display functionality"""
        fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.canvas.toolbar_visible = False
        fig.canvas.resizable = False

        plt.tight_layout()

        ax.imshow(np.rot90(self.image, 2), cmap='gray')
        ax.axis('off')

        text = widgets.Text(
            value = '',
            placeholder = 'Prompt',
            description = 'Enter a prompt',
            disabled = False,
            style = {'description_width': 'initial'}
        )
        display(text)

        def callback(wget):
            caption = wget.value.lower().strip()
            if len(fig.texts) > 0:
                fig.texts[0].remove()
            if caption not in self.avail_prompts:
                fig.text(
                    0.50,
                    0.02,
                    f"Invalid prompt: {wget.value}",
                    horizontalalignment='center',
                    wrap=True,
                    color='r'
                )
            else:
                ax.clear()
                ax.imshow(np.rot90(self.image, 2), cmap='gray')
                ax.axis('off')
                seg = self.infer(caption)
                self.show_mask(np.rot90(seg, 2), ax, random_color=False, alpha=alpha)

        text.on_submit(callback)
    
    def _show_volume(self, fig_size=5, alpha=0.95):
        """New 3D volume display functionality with slider"""
        fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.canvas.toolbar_visible = False
        fig.canvas.resizable = False

        plt.tight_layout()

        # Text input widget
        text = widgets.Text(
            value='',
            placeholder='Prompt',
            description='Enter a prompt:',
            disabled=False,
            style={'description_width': 'initial'}
        )
        
        # Slice slider
        slice_slider = IntSlider(
            value=self.current_slice,
            min=0,
            max=self.image.shape[0] - 1,
            step=1,
            description='Slice:',
            disabled=False,
            continuous_update=True,
            style={'description_width': 'initial'}
        )
        
        # Info label
        info_label = widgets.Label(value=f"Volume shape: {self.image.shape}")
        
        # Button to segment all slices
        segment_all_btn = widgets.Button(
            description='Segment All Slices',
            disabled=False,
            button_style='info',
            tooltip='Segment all slices with current prompt'
        )
        
        def update_display():
            """Update the display based on current slice and prompt"""
            ax.clear()
            current_image = self.image[self.current_slice]
            ax.imshow(np.rot90(current_image, 2), cmap='gray')
            ax.axis('off')
            ax.set_title(f'Slice {self.current_slice + 1}/{self.image.shape[0]}')
            
            # Show segmentation if available
            if self.current_prompt and self.current_prompt in self.segmentations:
                if self.current_slice in self.segmentations[self.current_prompt]:
                    seg = self.segmentations[self.current_prompt][self.current_slice]
                    self.show_mask(np.rot90(seg, 2), ax, random_color=False, alpha=alpha)
            
            # Clear any error messages
            if len(fig.texts) > 0:
                fig.texts[0].remove()
            
            fig.canvas.draw()
        
        def on_slice_change(change):
            """Handle slice slider changes"""
            self.current_slice = change['new']
            update_display()
        
        def on_text_submit(wget):
            """Handle text prompt submission"""
            caption = wget.value.lower().strip()
            
            if caption not in self.avail_prompts:
                if len(fig.texts) > 0:
                    fig.texts[0].remove()
                fig.text(
                    0.50,
                    0.02,
                    f"Invalid prompt: {wget.value}",
                    horizontalalignment='center',
                    wrap=True,
                    color='r'
                )
                fig.canvas.draw()
            else:
                self.current_prompt = caption
                # Initialize segmentations dict for this prompt if needed
                if caption not in self.segmentations:
                    self.segmentations[caption] = {}
                
                # Segment current slice
                seg = self.infer(caption, self.current_slice)
                self.segmentations[caption][self.current_slice] = seg
                update_display()
        
        def on_segment_all_click(btn):
            """Segment all slices with current prompt"""
            if not self.current_prompt:
                return
                
            if self.current_prompt not in self.segmentations:
                self.segmentations[self.current_prompt] = {}
            
            # Show progress
            btn.description = 'Segmenting...'
            btn.disabled = True
            
            # Segment all slices
            for slice_idx in range(self.image.shape[0]):
                seg = self.infer(self.current_prompt, slice_idx)
                self.segmentations[self.current_prompt][slice_idx] = seg
            
            # Reset button
            btn.description = 'Segment All Slices'
            btn.disabled = False
            
            # Update current display
            update_display()
        
        # Connect event handlers
        slice_slider.observe(on_slice_change, names='value')
        text.on_submit(on_text_submit)
        segment_all_btn.on_click(on_segment_all_click)
        
        # Create layout
        controls = VBox([
            text,
            HBox([slice_slider, segment_all_btn]),
            info_label
        ])
        
        # Display widgets
        display(controls)
        
        # Initial display
        update_display()

    def set_image(self, image):
        """
        Set the image for segmentation.
        
        Args:
            image: numpy array of shape (H, W) for 2D or (D, H, W) for 3D volume
        """
        self.image = image
        self.img_size = image.shape
        
        # Determine if this is a volume or single slice
        if len(image.shape) == 3:
            self.is_volume = True
            self.current_slice = 0
            self.segmentations = {}  # Reset segmentations
            print(f"Set 3D volume with shape: {image.shape}")
            
            # Process all slices and compute embeddings
            self.image_embeddings = []
            print("Computing embeddings for all slices...")
            
            with torch.no_grad():
                for slice_idx in range(image.shape[0]):
                    slice_2d = image[slice_idx]
                    
                    # Convert to 3-channel if grayscale
                    if len(slice_2d.shape) == 2:
                        slice_2d = np.repeat(slice_2d[:,:,None], 3, -1)
                    
                    slice_preprocess = self.preprocess_image(slice_2d)
                    slice_embeddings = self.model.image_encoder(slice_preprocess)
                    self.image_embeddings.append(slice_embeddings)
                    
            print(f"Computed embeddings for {len(self.image_embeddings)} slices")
            
        else:
            self.is_volume = False
            print(f"Set 2D image with shape: {image.shape}")
            
            # Convert to 3-channel if grayscale
            if len(image.shape) == 2:
                image_3ch = np.repeat(image[:,:,None], 3, -1)
            else:
                image_3ch = image
                
            image_preprocess = self.preprocess_image(image_3ch)
            with torch.no_grad():
                self.image_embeddings = self.model.image_encoder(image_preprocess)

    def preprocess_image(self, image):
        img_resize = cv2.resize(
            image,
            (1024, 1024),
            interpolation=cv2.INTER_CUBIC
        )
        # Resizing
        img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None) # normalize to [0, 1], (H, W, 3
        # convert the shape to (3, H, W)
        assert np.max(img_resize)<=1.0 and np.min(img_resize)>=0.0, 'image should be normalized to [0, 1]'
        img_tensor = torch.tensor(img_resize).float().permute(2, 0, 1).unsqueeze(0).to(self.model.device)

        return img_tensor


    def tokenize_text(self, text):
        """
        Tokenize text using CLIP tokenizer
        """
        return self.tokenizer(
            text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.squeeze(1)

    def get_segmentation_volume(self, prompt):
        """
        Get the complete 3D segmentation volume for a given prompt.
        
        Args:
            prompt: Text prompt used for segmentation
            
        Returns:
            numpy array of shape (D, H, W) with segmentations, or None if not available
        """
        if not self.is_volume or prompt not in self.segmentations:
            return None
            
        seg_dict = self.segmentations[prompt]
        if len(seg_dict) != self.image.shape[0]:
            print(f"Warning: Only {len(seg_dict)} out of {self.image.shape[0]} slices segmented")
            
        # Create volume array
        volume_shape = self.image.shape
        seg_volume = np.zeros(volume_shape, dtype=np.uint8)
        
        for slice_idx, seg in seg_dict.items():
            seg_volume[slice_idx] = seg
            
        return seg_volume
    
    def save_segmentation_volume(self, prompt, filepath):
        """
        Save the 3D segmentation volume to a file.
        
        Args:
            prompt: Text prompt used for segmentation
            filepath: Path to save the volume (should end with .npy or .npz)
        """
        seg_volume = self.get_segmentation_volume(prompt)
        if seg_volume is not None:
            if filepath.endswith('.npz'):
                np.savez_compressed(filepath, segmentation=seg_volume, prompt=prompt)
            else:
                np.save(filepath, seg_volume)
            print(f"Saved segmentation volume to {filepath}")
        else:
            print("No complete segmentation volume available for this prompt")
    
    def get_segmentation_statistics(self, prompt):
        """
        Get statistics about segmentation coverage for a prompt.
        
        Args:
            prompt: Text prompt to analyze
            
        Returns:
            dict with statistics
        """
        if not self.is_volume or prompt not in self.segmentations:
            return None
            
        seg_dict = self.segmentations[prompt]
        total_slices = self.image.shape[0]
        segmented_slices = len(seg_dict)
        
        # Calculate volume statistics
        total_voxels = 0
        segmented_voxels = 0
        
        for slice_idx, seg in seg_dict.items():
            total_voxels += seg.size
            segmented_voxels += np.sum(seg)
        
        stats = {
            'total_slices': total_slices,
            'segmented_slices': segmented_slices,
            'coverage_percentage': (segmented_slices / total_slices) * 100,
            'total_voxels_analyzed': total_voxels,
            'segmented_voxels': segmented_voxels,
            'segmentation_percentage': (segmented_voxels / total_voxels) * 100 if total_voxels > 0 else 0
        }
        
        return stats