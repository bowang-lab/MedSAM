import gc
import numpy as np
from os import getcwd
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets, IntSlider, VBox, HBox
from IPython.display import display, clear_output
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

#%% Define point prompt inference pipeline and GUI
class PointPromptDemo:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.image = None  # Can be 2D (H, W) or 3D (D, H, W)
        self.image_embeddings = None  # For 3D: list of embeddings per slice
        self.img_size = None
        self.is_volume = False
        self.current_slice = 0
        self.segmentations = {}  # Store segmentations for each point and slice
        self.point_prompts = {}  # Store point coordinates for each slice
        self.use_same_point_for_all = False
        self.global_point = None  # Point to use for all slices when use_same_point_for_all is True
        

    def show_mask(self, mask, ax, random_color=False, alpha=0.95):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
        else:
            color = np.array([251/255, 252/255, 30/255, alpha])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def debug_coordinates(self, x, y, slice_idx=None):
        """
        Debug coordinate transformation to help identify issues
        
        Args:
            x, y: Click coordinates
            slice_idx: Slice index for 3D volumes
        """
        if self.is_volume:
            if slice_idx is None:
                slice_idx = self.current_slice
            img_size = self.img_size[1:]  # (H, W)
        else:
            img_size = self.img_size if len(self.img_size) == 2 else self.img_size[:2]
        
        # Calculate transformed coordinates
        x_1024 = x * 1024.0 / img_size[1]
        y_1024 = y * 1024.0 / img_size[0]
        
        print(f"Debug Coordinates:")
        print(f"  Original click: ({x:.2f}, {y:.2f})")
        print(f"  Image size: {img_size}")
        print(f"  Transformed to 1024x1024: ({x_1024:.2f}, {y_1024:.2f})")
        print(f"  Within bounds: {0 <= x_1024 <= 1024 and 0 <= y_1024 <= 1024}")
        
        return x_1024, y_1024

    @torch.no_grad()
    def infer(self, x, y, slice_idx=None, return_raw=False):
        """
        Perform inference on a single slice or current slice
        
        Args:
            x, y: Point coordinates in display coordinate system
            slice_idx: Specific slice index for 3D volumes, None for 2D or current slice
        """
        if self.is_volume:
            if slice_idx is None:
                slice_idx = self.current_slice
            image_embeddings = self.image_embeddings[slice_idx]
            img_size = self.img_size[1:]  # Remove depth dimension (H, W)
        else:
            image_embeddings = self.image_embeddings
            img_size = self.img_size if len(self.img_size) == 2 else self.img_size[:2]  # Ensure (H, W)
            
        # Convert click coordinates to model input coordinates (1024x1024)
        # Note: x corresponds to width (columns), y corresponds to height (rows)
        # SAM expects coordinates in (x, y) format where x=width, y=height
        coords_1024 = np.array([[[
            x * 1024.0 / img_size[1],  # x coordinate scaled to 1024 (width direction)
            y * 1024.0 / img_size[0]   # y coordinate scaled to 1024 (height direction)
        ]]])
        coords_torch = torch.tensor(coords_1024, dtype=torch.float32).to(self.model.device)
        labels_torch = torch.tensor([[1]], dtype=torch.long).to(self.model.device)
        point_prompt = (coords_torch, labels_torch)

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points = point_prompt,
            boxes = None,
            masks = None,
        )
        low_res_logits, _ = self.model.mask_decoder(
            image_embeddings=image_embeddings, # (B, 256, 64, 64)
            image_pe=self.model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
        )

        low_res_probs = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
        low_res_pred = F.interpolate(
            low_res_probs,
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

    def show(self, fig_size=5, alpha=0.95, scatter_size=10):
        """
        Display the interactive segmentation interface.
        For 2D images: shows single image with point prompts
        For 3D volumes: shows image with point prompts and slice slider
        """
        assert self.image is not None, "Please set image first."
        
        if self.is_volume:
            self._show_volume(fig_size, alpha, scatter_size)
        else:
            self._show_2d(fig_size, alpha, scatter_size)
    
    def _show_2d(self, fig_size=5, alpha=0.95, scatter_size=10):
        """Original 2D display functionality"""
        seg = None
        fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.canvas.toolbar_visible = False
        fig.canvas.resizable = False

        plt.tight_layout()

        # Display image with proper extent to ensure coordinate mapping
        ax.imshow(self.image, cmap='gray' if len(self.image.shape) == 2 else None, 
                 extent=[0, self.img_size[1], self.img_size[0], 0])
        ax.set_xlim(0, self.img_size[1])
        ax.set_ylim(self.img_size[0], 0)  # Invert y-axis to match image coordinates
        ax.axis('off')

        def onclick(event):
            nonlocal seg
            if event.inaxes == ax:
                x, y = float(event.xdata), float(event.ydata)
                
                # Validate coordinates
                if x < 0 or y < 0 or x >= self.img_size[1] or y >= self.img_size[0]:
                    print(f"Warning: Click coordinates ({x:.1f}, {y:.1f}) outside image bounds ({self.img_size[1]}, {self.img_size[0]})")
                    return
                
                print(f"Clicked at: ({x:.1f}, {y:.1f}), Image size: {self.img_size}")
                
                with torch.no_grad():
                    ## rescale x, y from canvas size to 1024 x 1024
                    seg = self.infer(x, y)

                ax.clear()
                ax.imshow(self.image, cmap='gray' if len(self.image.shape) == 2 else None,
                         extent=[0, self.img_size[1], self.img_size[0], 0])
                ax.set_xlim(0, self.img_size[1])
                ax.set_ylim(self.img_size[0], 0)
                ax.axis('off')
                ax.scatter(x, y, c='r', s=scatter_size)
                self.show_mask(seg, ax, random_color=False, alpha=alpha)

                gc.collect()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        save_button = widgets.Button(description="save")
        def __on_save_button_clicked(b):
            plt.savefig("seg_result.png", bbox_inches='tight', pad_inches=0)
            if seg is not None:
                cv2.imwrite("seg.png", seg * 255)  # Scale to 255 for proper saving
                print(f"Segmentation result saved to {getcwd()}")

        display(save_button)
        save_button.on_click(__on_save_button_clicked)
    
    def _show_volume(self, fig_size=5, alpha=0.95, scatter_size=10):
        """New 3D volume display functionality with slider"""
        fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.canvas.toolbar_visible = False
        fig.canvas.resizable = False

        plt.tight_layout()

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
        
        # Mode toggle checkbox
        mode_checkbox = widgets.Checkbox(
            value=self.use_same_point_for_all,
            description='Use same point for all slices',
            disabled=False,
            style={'description_width': 'initial'}
        )
        
        # Info label
        info_label = widgets.Label(value=f"Volume shape: {self.image.shape}")
        
        # Button to segment all slices
        segment_all_btn = widgets.Button(
            description='Segment All Slices',
            disabled=False,
            button_style='info',
            tooltip='Segment all slices with current/individual points'
        )
        
        # Button to clear current slice segmentation
        clear_btn = widgets.Button(
            description='Clear Current',
            disabled=False,
            button_style='warning',
            tooltip='Clear segmentation for current slice'
        )
        
        # Button to clear all segmentations
        clear_all_btn = widgets.Button(
            description='Clear All',
            disabled=False,
            button_style='danger',
            tooltip='Clear all segmentations'
        )
        
        # Save buttons
        save_slice_btn = widgets.Button(
            description='Save Current Slice',
            disabled=False,
            button_style='success',
            tooltip='Save current slice segmentation'
        )
        
        save_volume_btn = widgets.Button(
            description='Save Volume',
            disabled=False,
            button_style='success',
            tooltip='Save complete 3D segmentation volume'
        )
        
        def update_display():
            """Update the display based on current slice and point"""
            ax.clear()
            current_image = self.image[self.current_slice]
            current_shape = current_image.shape  # (H, W)
            
            # Display with proper extent for coordinate mapping
            ax.imshow(current_image, cmap='gray', 
                     extent=[0, current_shape[1], current_shape[0], 0])
            ax.set_xlim(0, current_shape[1])
            ax.set_ylim(current_shape[0], 0)  # Invert y-axis to match image coordinates
            ax.axis('off')
            ax.set_title(f'Slice {self.current_slice + 1}/{self.image.shape[0]}')
            
            # Show point if available
            point_to_show = None
            if self.use_same_point_for_all and self.global_point is not None:
                point_to_show = self.global_point
            elif self.current_slice in self.point_prompts:
                point_to_show = self.point_prompts[self.current_slice]
            
            if point_to_show is not None:
                ax.scatter(point_to_show[0], point_to_show[1], c='r', s=scatter_size)
            
            # Show segmentation if available
            if self.current_slice in self.segmentations:
                seg = self.segmentations[self.current_slice]
                self.show_mask(seg, ax, random_color=False, alpha=alpha)
            
            fig.canvas.draw()
        
        def onclick(event):
            """Handle mouse clicks on the image"""
            if event.inaxes == ax:
                x, y = float(event.xdata), float(event.ydata)
                
                # Get current slice dimensions for validation
                current_img_shape = self.img_size[1:]  # (H, W) for current slice
                
                # Validate coordinates
                if x < 0 or y < 0 or x >= current_img_shape[1] or y >= current_img_shape[0]:
                    print(f"Warning: Click coordinates ({x:.1f}, {y:.1f}) outside slice bounds ({current_img_shape[1]}, {current_img_shape[0]})")
                    return
                
                print(f"Clicked at: ({x:.1f}, {y:.1f}), Slice {self.current_slice}, Size: {current_img_shape}")
                
                if self.use_same_point_for_all:
                    # Store as global point
                    self.global_point = (x, y)
                else:
                    # Store as slice-specific point
                    self.point_prompts[self.current_slice] = (x, y)
                
                # Perform segmentation
                with torch.no_grad():
                    seg = self.infer(x, y, self.current_slice)
                    self.segmentations[self.current_slice] = seg
                
                update_display()
                gc.collect()
        
        def on_slice_change(change):
            """Handle slice slider changes"""
            self.current_slice = change['new']
            update_display()
        
        def on_mode_change(change):
            """Handle mode checkbox changes"""
            self.use_same_point_for_all = change['new']
            update_display()
        
        def on_segment_all_click(btn):
            """Segment all slices based on mode"""
            if self.use_same_point_for_all:
                if self.global_point is None:
                    print("Please click on a point first when using 'same point for all slices' mode")
                    return
                
                # Show progress
                btn.description = 'Segmenting...'
                btn.disabled = True
                
                # Segment all slices with the same point
                x, y = self.global_point
                for slice_idx in range(self.image.shape[0]):
                    with torch.no_grad():
                        seg = self.infer(x, y, slice_idx)
                        self.segmentations[slice_idx] = seg
                
                # Reset button
                btn.description = 'Segment All Slices'
                btn.disabled = False
                update_display()
                print(f"Segmented all {self.image.shape[0]} slices with the same point")
                
            else:
                # Segment only slices that have individual points
                if len(self.point_prompts) == 0:
                    print("Please click on points for individual slices first")
                    return
                
                # Show progress
                btn.description = 'Segmenting...'
                btn.disabled = True
                
                # Segment slices with their individual points
                for slice_idx, (x, y) in self.point_prompts.items():
                    with torch.no_grad():
                        seg = self.infer(x, y, slice_idx)
                        self.segmentations[slice_idx] = seg
                
                # Reset button
                btn.description = 'Segment All Slices'
                btn.disabled = False
                update_display()
                print(f"Segmented {len(self.point_prompts)} slices with individual points")
        
        def on_clear_click(btn):
            """Clear current slice segmentation"""
            if self.current_slice in self.segmentations:
                del self.segmentations[self.current_slice]
            if not self.use_same_point_for_all and self.current_slice in self.point_prompts:
                del self.point_prompts[self.current_slice]
            update_display()
        
        def on_clear_all_click(btn):
            """Clear all segmentations and points"""
            self.segmentations.clear()
            self.point_prompts.clear()
            self.global_point = None
            update_display()
        
        def on_save_slice_click(btn):
            """Save current slice segmentation"""
            if self.current_slice in self.segmentations:
                seg = self.segmentations[self.current_slice]
                filename = f"seg_slice_{self.current_slice + 1}.png"
                cv2.imwrite(filename, seg * 255)  # Scale to 255 for proper image saving
                print(f"Saved current slice segmentation to {filename}")
            else:
                print("No segmentation available for current slice")
        
        def on_save_volume_click(btn):
            """Save complete 3D segmentation volume"""
            if len(self.segmentations) > 0:
                filename = f"point_seg_volume_{len(self.segmentations)}slices.npz"
                self.save_segmentation_volume(filename)
                
                # Also save statistics
                stats = self.get_segmentation_statistics()
                if stats:
                    print("\nSegmentation Statistics:")
                    for key, value in stats.items():
                        if 'percentage' in key:
                            print(f"  {key}: {value:.2f}%")
                        else:
                            print(f"  {key}: {value}")
            else:
                print("No segmentations available to save")
        
        # Connect event handlers
        slice_slider.observe(on_slice_change, names='value')
        mode_checkbox.observe(on_mode_change, names='value')
        segment_all_btn.on_click(on_segment_all_click)
        clear_btn.on_click(on_clear_click)
        clear_all_btn.on_click(on_clear_all_click)
        save_slice_btn.on_click(on_save_slice_click)
        save_volume_btn.on_click(on_save_volume_click)
        fig.canvas.mpl_connect('button_press_event', onclick)
        
        # Create layout
        controls = VBox([
            mode_checkbox,
            HBox([slice_slider, segment_all_btn]),
            HBox([clear_btn, clear_all_btn]),
            info_label
        ])
        
        # Display widgets
        display(controls)
        
        # Show the plot
        plt.show()
        
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
            self.point_prompts = {}  # Reset point prompts
            self.global_point = None
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
            
            # Convert to 3-channel if grayscale and update self.image
            if len(image.shape) == 2:
                # Keep original 2D image for display, but store 3-channel version for processing
                self.image_3ch = np.repeat(image[:,:,None], 3, -1)
                image_preprocess = self.preprocess_image(self.image_3ch)
            else:
                self.image_3ch = image
                image_preprocess = self.preprocess_image(self.image)
                
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

    def get_segmentation_volume(self):
        """
        Get the complete 3D segmentation volume.
        
        Returns:
            numpy array of shape (D, H, W) with segmentations, or None if not available
        """
        if not self.is_volume or len(self.segmentations) == 0:
            return None
            
        if len(self.segmentations) != self.image.shape[0]:
            print(f"Warning: Only {len(self.segmentations)} out of {self.image.shape[0]} slices segmented")
            
        # Create volume array
        volume_shape = self.image.shape
        seg_volume = np.zeros(volume_shape, dtype=np.uint8)
        
        for slice_idx, seg in self.segmentations.items():
            seg_volume[slice_idx] = seg
            
        return seg_volume
    
    def save_segmentation_volume(self, filepath):
        """
        Save the 3D segmentation volume to a file.
        
        Args:
            filepath: Path to save the volume (should end with .npy or .npz)
        """
        seg_volume = self.get_segmentation_volume()
        if seg_volume is not None:
            if filepath.endswith('.npz'):
                point_info = {
                    'use_same_point_for_all': self.use_same_point_for_all,
                    'global_point': self.global_point,
                    'individual_points': self.point_prompts
                }
                np.savez_compressed(filepath, segmentation=seg_volume, point_info=point_info)
            else:
                np.save(filepath, seg_volume)
            print(f"Saved segmentation volume to {filepath}")
        else:
            print("No segmentation volume available")
    
    def get_segmentation_statistics(self):
        """
        Get statistics about segmentation coverage.
        
        Returns:
            dict with statistics
        """
        if not self.is_volume or len(self.segmentations) == 0:
            return None
            
        total_slices = self.image.shape[0]
        segmented_slices = len(self.segmentations)
        
        # Calculate volume statistics
        total_voxels = 0
        segmented_voxels = 0
        
        for slice_idx, seg in self.segmentations.items():
            total_voxels += seg.size
            segmented_voxels += np.sum(seg)
        
        stats = {
            'total_slices': total_slices,
            'segmented_slices': segmented_slices,
            'coverage_percentage': (segmented_slices / total_slices) * 100,
            'total_voxels_analyzed': total_voxels,
            'segmented_voxels': segmented_voxels,
            'segmentation_percentage': (segmented_voxels / total_voxels) * 100 if total_voxels > 0 else 0,
            'mode': 'same_point_for_all' if self.use_same_point_for_all else 'individual_points'
        }
        
        return stats
    
    def save_current_segmentation(self, filepath=None):
        """
        Save the current slice segmentation or full volume.
        For 2D: saves current segmentation
        For 3D: saves current slice or full volume based on availability
        """
        if filepath is None:
            filepath = f"seg_slice_{self.current_slice}.png" if self.is_volume else "seg.png"
        
        if self.is_volume:
            if self.current_slice in self.segmentations:
                seg = self.segmentations[self.current_slice]
                cv2.imwrite(filepath, seg)
                print(f"Saved current slice segmentation to {filepath}")
            else:
                print("No segmentation available for current slice")
        else:
            # For 2D, we need to get the segmentation from the current display
            print("For 2D images, use the save button in the interface")