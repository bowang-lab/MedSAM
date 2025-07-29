import gc
import numpy as np
from os import getcwd
from copy import deepcopy
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets, IntSlider, VBox, HBox
from IPython.display import display, clear_output
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper function to show a mask on a matplotlib axis
def show_mask(mask, ax, random_color=False, alpha=0.85, color=None):
    """
    Displays a segmentation mask on a given matplotlib axis.
    A specific color can be provided to ensure stability across redraws.

    Args:
        mask (np.ndarray): The segmentation mask (H, W).
        ax (matplotlib.axes.Axes): The axis to display the mask on.
        random_color (bool): If True, uses a random color for the mask.
        alpha (float): The transparency of the mask.
        color (tuple, optional): A (r, g, b) tuple for the mask color.
    """
    display_color = None
    if color is not None:
        # Use the provided color, applying the alpha channel
        display_color = np.array([*color, alpha])
    elif random_color:
        # Generate a random color
        display_color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
    else:
        # Default color
        display_color = np.array([30/255, 144/255, 255/255, alpha]) # Dodger blue
    
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * display_color.reshape(1, 1, -1)
    ax.imshow(mask_image)


class PointPromptDemo:
    """
    An interactive demonstration class for point-prompted image segmentation.
    Supports both 2D images and 3D volumes.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.image = None
        self.image_embeddings = None
        self.img_size = None
        self.is_volume = False
        self.current_slice = 0
        self.segmentations = {}
        self.point_prompts = {}
        self.use_same_point_for_all = False
        self.global_point = None

    def set_image(self, image):
        """
        Sets the image for segmentation. Handles both 2D and 3D images.
        For 3D volumes, it pre-computes embeddings for each slice.

        Args:
            image (np.ndarray): A 2D (H, W) or 3D (D, H, W) numpy array.
        """
        self.image = image
        self.img_size = image.shape
        self.is_volume = len(image.shape) == 3

        # Clear previous state
        self.segmentations.clear()
        self.point_prompts.clear()
        self.global_point = None
        self.current_slice = 0

        if self.is_volume:
            print(f"Set 3D volume with shape: {image.shape}")
            self.image_embeddings = []
            print("Computing embeddings for all slices...")
            with torch.no_grad():
                for i in range(image.shape[0]):
                    slice_img = image[i]
                    if len(slice_img.shape) == 2:
                        slice_img = np.repeat(slice_img[:, :, None], 3, axis=-1)
                    preprocessed_slice = self._preprocess_image(slice_img)
                    self.image_embeddings.append(self.model.image_encoder(preprocessed_slice))
            print(f"Computed embeddings for {len(self.image_embeddings)} slices.")
        else:
            print(f"Set 2D image with shape: {image.shape}")
            img_3c = image
            if len(image.shape) == 2:
                img_3c = np.repeat(image[:, :, None], 3, axis=-1)
            preprocessed_img = self._preprocess_image(img_3c)
            with torch.no_grad():
                self.image_embeddings = self.model.image_encoder(preprocessed_img)

    def _preprocess_image(self, image):
        """Preprocesses an image for the model."""
        img_resize = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None)
        img_tensor = torch.tensor(img_resize).float().permute(2, 0, 1).unsqueeze(0).to(self.model.device)
        return img_tensor

    @torch.no_grad()
    def infer(self, x, y, slice_idx=None, return_raw=False):
        """
        Performs inference for a point prompt.

        Args:
            x (float): The x-coordinate of the point.
            y (float): The y-coordinate of the point.
            slice_idx (int, optional): The index of the slice for 3D volumes.

        Returns:
            np.ndarray: The resulting segmentation mask.
        """
        if self.is_volume:
            if slice_idx is None:
                slice_idx = self.current_slice
            image_embeddings = self.image_embeddings[slice_idx]
            img_size = self.img_size[1:]
        else:
            image_embeddings = self.image_embeddings
            img_size = self.img_size

        coords_1024 = np.array([[[x * 1024.0 / img_size[1], y * 1024.0 / img_size[0]]]])
        coords_torch = torch.tensor(coords_1024, dtype=torch.float32, device=self.model.device)
        labels_torch = torch.tensor([[1]], dtype=torch.long, device=self.model.device)
        point_prompt = (coords_torch, labels_torch)

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=point_prompt, boxes=None, masks=None)
        low_res_logits, _ = self.model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        low_res_probs = torch.sigmoid(low_res_logits)
        low_res_pred = F.interpolate(low_res_probs, size=img_size, mode='bilinear', align_corners=False)
        low_res_pred = low_res_pred.detach().cpu().numpy().squeeze()

        seg = (low_res_pred > 0.5).astype(np.uint8)

        if return_raw:
            return seg, low_res_pred
        return seg

    def show(self, fig_size=5, alpha=0.95, scatter_size=10):
        """Displays the interactive segmentation GUI."""
        assert self.image is not None, "Please set an image first using set_image()."
        if self.is_volume:
            self._show_volume(fig_size, alpha, scatter_size)
        else:
            self._show_2d(fig_size, alpha, scatter_size)

    def _show_2d(self, fig_size, alpha, scatter_size):
        """GUI for 2D images."""
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        fig.canvas.toolbar_visible = False
        ax.imshow(self.image, cmap='gray')
        ax.axis('off')

        def onclick(event):
            if event.inaxes == ax:
                x, y = event.xdata, event.ydata
                seg = self.infer(x, y)
                ax.clear()
                ax.imshow(self.image, cmap='gray')
                show_mask(seg, ax, alpha=alpha)
                ax.scatter(x, y, c='r', s=scatter_size)
                ax.axis('off')
                fig.canvas.draw_idle()
                gc.collect()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

    def _show_volume(self, fig_size, alpha, scatter_size):
        """GUI for 3D volumes."""
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        fig.canvas.toolbar_visible = False
        plt.tight_layout()

        slice_slider = IntSlider(value=0, min=0, max=self.image.shape[0] - 1, description='Slice:')
        mode_checkbox = widgets.Checkbox(value=False, description='Use same point for all slices')
        segment_all_btn = widgets.Button(description='Segment All Slices', button_style='info')
        clear_btn = widgets.Button(description='Clear Current', button_style='warning')
        clear_all_btn = widgets.Button(description='Clear All', button_style='danger')
        save_volume_btn = widgets.Button(description='Save Volume', button_style='success')

        def update_display():
            ax.clear()
            ax.imshow(self.image[self.current_slice], cmap='gray')
            ax.set_title(f'Slice {self.current_slice + 1}/{self.image.shape[0]}')
            
            point_to_show = self.global_point if self.use_same_point_for_all else self.point_prompts.get(self.current_slice)
            if point_to_show:
                ax.scatter(point_to_show[0], point_to_show[1], c='r', s=scatter_size)
            
            if self.current_slice in self.segmentations:
                show_mask(self.segmentations[self.current_slice], ax, alpha=alpha)
            
            ax.axis('off')
            fig.canvas.draw_idle()

        def onclick(event):
            if event.inaxes == ax:
                x, y = event.xdata, event.ydata
                if self.use_same_point_for_all:
                    self.global_point = (x, y)
                else:
                    self.point_prompts[self.current_slice] = (x, y)
                
                seg = self.infer(x, y, self.current_slice)
                self.segmentations[self.current_slice] = seg
                update_display()
                gc.collect()

        def on_slice_change(change):
            self.current_slice = change['new']
            update_display()

        def on_mode_change(change):
            self.use_same_point_for_all = change['new']
            update_display()
        
        def on_segment_all_click(b):
            if self.use_same_point_for_all and self.global_point:
                x, y = self.global_point
                for i in range(self.image.shape[0]):
                    self.segmentations[i] = self.infer(x, y, i)
                print("Segmented all slices with the global point.")
            else:
                 for i, point in self.point_prompts.items():
                    self.segmentations[i] = self.infer(point[0], point[1], i)
                 print(f"Segmented {len(self.point_prompts)} slices with individual points.")
            update_display()

        def on_clear_click(b):
            self.segmentations.pop(self.current_slice, None)
            if not self.use_same_point_for_all:
                self.point_prompts.pop(self.current_slice, None)
            update_display()

        def on_clear_all_click(b):
            self.segmentations.clear()
            self.point_prompts.clear()
            self.global_point = None
            update_display()

        def on_save_volume_click(b):
            self.save_segmentation_volume("point_seg_volume.npz")

        slice_slider.observe(on_slice_change, names='value')
        mode_checkbox.observe(on_mode_change, names='value')
        segment_all_btn.on_click(on_segment_all_click)
        clear_btn.on_click(on_clear_click)
        clear_all_btn.on_click(on_clear_all_click)
        save_volume_btn.on_click(on_save_volume_click)
        fig.canvas.mpl_connect('button_press_event', onclick)

        controls = VBox([mode_checkbox, HBox([slice_slider, segment_all_btn]), HBox([clear_btn, clear_all_btn, save_volume_btn])])
        display(controls)
        plt.show()
        update_display()

    def get_segmentation_volume(self):
        if not self.is_volume or not self.segmentations:
            return None
        vol_shape = self.image.shape
        seg_volume = np.zeros(vol_shape, dtype=np.uint8)
        for i, seg in self.segmentations.items():
            seg_volume[i] = seg
        return seg_volume

    def save_segmentation_volume(self, filepath):
        seg_volume = self.get_segmentation_volume()
        if seg_volume is not None:
            np.savez_compressed(filepath, segmentation=seg_volume)
            print(f"Saved segmentation volume to {filepath}")
        else:
            print("No segmentation volume to save.")


class BboxPromptDemo:
    """
    An interactive demonstration class for bounding box-prompted image segmentation.
    This class has been extended to support both 2D images and 3D volumes.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
        # Image and state attributes
        self.image = None
        self.image_embeddings = None
        self.img_size = None
        self.is_volume = False
        self.current_slice = 0
        
        # Unified prompt storage for cleaner state management
        # For slice-specific prompts: {slice_idx: [{'mask': seg, 'bbox': bbox, 'color': color}, ...]}
        self.slice_prompts = {}
        # For 'use same for all' mode: {'bbox': bbox, 'color': color}
        self.global_prompt = None
        
        # UI and interaction state
        self.use_same_bbox_for_all = False
        self.currently_selecting = False
        self.start_point = None
        self.rect_patch = None
        self.fig = None
        self.ax = None

    def set_image(self, image):
        """
        Sets the image for segmentation. Handles both 2D and 3D images.
        For 3D volumes, it pre-computes embeddings for each slice.

        Args:
            image (np.ndarray): A 2D (H, W) or 3D (D, H, W) numpy array.
        """
        self.image = image
        self.img_size = image.shape
        self.is_volume = len(image.shape) == 3

        # Clear previous state
        self.slice_prompts.clear()
        self.global_prompt = None
        self.current_slice = 0

        if self.is_volume:
            print(f"Set 3D volume with shape: {image.shape}")
            self.image_embeddings = []
            print("Computing embeddings for all slices...")
            with torch.no_grad():
                for i in range(image.shape[0]):
                    slice_img = image[i]
                    if len(slice_img.shape) == 2:
                        slice_img = np.repeat(slice_img[:, :, None], 3, axis=-1)
                    preprocessed_slice = self._preprocess_image(slice_img)
                    self.image_embeddings.append(self.model.image_encoder(preprocessed_slice))
            print(f"Computed embeddings for {len(self.image_embeddings)} slices.")
        else:
            print(f"Set 2D image with shape: {image.shape}")
            img_3c = image
            if len(image.shape) == 2:
                img_3c = np.repeat(image[:, :, None], 3, axis=-1)
            preprocessed_img = self._preprocess_image(img_3c)
            with torch.no_grad():
                self.image_embeddings = self.model.image_encoder(preprocessed_img)

    def _preprocess_image(self, image):
        """Preprocesses a single image frame for the model."""
        img_resize = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None)
        img_tensor = torch.tensor(img_resize).float().permute(2, 0, 1).unsqueeze(0).to(self.model.device)
        return img_tensor

    @torch.no_grad()
    def infer(self, bbox, slice_idx=None, return_raw=False):
        """
        Performs inference for a bounding box prompt.

        Args:
            bbox (np.ndarray): The bounding box [x_min, y_min, x_max, y_max].
            slice_idx (int, optional): The index of the slice for 3D volumes.

        Returns:
            np.ndarray: The resulting segmentation mask.
        """
        if self.is_volume:
            if slice_idx is None:
                slice_idx = self.current_slice
            image_embeddings = self.image_embeddings[slice_idx]
            img_size = self.img_size[1:]
        else:
            image_embeddings = self.image_embeddings
            img_size = self.img_size
        
        ori_H, ori_W = img_size
        scale_to_1024 = 1024 / np.array([ori_W, ori_H, ori_W, ori_H])
        bbox_1024 = bbox * scale_to_1024
        bbox_torch = torch.as_tensor(bbox_1024, dtype=torch.float, device=self.model.device).unsqueeze(0)
        if len(bbox_torch.shape) == 2:
            bbox_torch = bbox_torch.unsqueeze(1)

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=None, boxes=bbox_torch, masks=None)
        low_res_logits, _ = self.model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        low_res_pred = torch.sigmoid(low_res_logits)
        low_res_pred = F.interpolate(low_res_pred, size=img_size, mode="bilinear", align_corners=False)
        low_res_pred = low_res_pred.detach().cpu().numpy().squeeze()

        seg = (low_res_pred > 0.5).astype(np.uint8)

        if return_raw:
            return seg, low_res_pred
        return seg

    def show(self, fig_size=6):
        """Displays the interactive segmentation GUI."""
        assert self.image is not None, "Please set an image first using set_image()."
        if self.is_volume:
            self._show_volume(fig_size)
        else:
            self._show_2d(fig_size)

    def _show_2d(self, fig_size):
        """GUI for 2D images."""
        self.fig, self.ax = plt.subplots(figsize=(fig_size, fig_size))
        self.fig.canvas.toolbar_visible = False
        plt.tight_layout()

        show_bbox_checkbox = widgets.Checkbox(value=False, description='Show BBoxes')
        clear_button = widgets.Button(description="Clear All")
        
        def update_display():
            self.ax.clear()
            self.ax.imshow(self.image, cmap='gray')
            
            prompts_to_show = self.slice_prompts.get(0, [])
            for prompt in prompts_to_show:
                show_mask(prompt['mask'], self.ax, color=prompt['color'])
                if show_bbox_checkbox.value:
                    bbox = prompt['bbox']
                    x_min, y_min, x_max, y_max = bbox
                    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                         edgecolor='cyan', facecolor='none', lw=1.5, linestyle='--')
                    self.ax.add_patch(rect)

            self.ax.axis('off')
            self.fig.canvas.draw_idle()

        def on_press(event):
            if event.inaxes == self.ax:
                self.currently_selecting = True
                self.start_point = (event.xdata, event.ydata)
                self.rect_patch = plt.Rectangle(self.start_point, 0, 0, edgecolor="magenta", facecolor='none', lw=2)
                self.ax.add_patch(self.rect_patch)

        def on_motion(event):
            if self.currently_selecting and event.inaxes == self.ax:
                if self.start_point is None: return
                x0, y0 = self.start_point
                x1, y1 = event.xdata, event.ydata
                self.rect_patch.set_width(x1 - x0)
                self.rect_patch.set_height(y1 - y0)
                self.fig.canvas.draw_idle()

        def on_release(event):
            if self.currently_selecting and event.inaxes == self.ax:
                self.currently_selecting = False
                if self.start_point is None: return
                x0, y0 = self.start_point
                x1, y1 = event.xdata, event.ydata
                self.start_point = None
                self.rect_patch.remove()

                bbox = np.array([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)])
                new_color = np.random.random(3)
                seg = self.infer(bbox, slice_idx=0)
                new_prompt = {'mask': seg, 'bbox': bbox, 'color': new_color}

                if 0 not in self.slice_prompts: self.slice_prompts[0] = []
                self.slice_prompts[0].append(new_prompt)

                update_display()
                gc.collect()

        def on_clear_click(b):
            self.slice_prompts.clear()
            update_display()
        
        def on_checkbox_change(change):
            update_display()

        self.fig.canvas.mpl_connect('button_press_event', on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', on_motion)
        self.fig.canvas.mpl_connect('button_release_event', on_release)
        clear_button.on_click(on_clear_click)
        show_bbox_checkbox.observe(on_checkbox_change, names='value')

        display(HBox([clear_button, show_bbox_checkbox]))
        plt.show()
        update_display()

    def _show_volume(self, fig_size):
        """GUI for 3D volumes."""
        self.fig, self.ax = plt.subplots(figsize=(fig_size, fig_size))
        self.fig.canvas.toolbar_visible = False
        plt.tight_layout()

        # --- Widgets ---
        slice_slider = IntSlider(value=0, min=0, max=self.image.shape[0] - 1, description='Slice:', continuous_update=False)
        mode_checkbox = widgets.Checkbox(value=False, description='Use same BBox for all slices')
        show_bbox_checkbox = widgets.Checkbox(value=False, description='Show BBoxes')
        segment_all_btn = widgets.Button(description='Segment All Slices', button_style='info')
        clear_btn = widgets.Button(description='Clear Current Slice', button_style='warning')
        clear_all_btn = widgets.Button(description='Clear All', button_style='danger')
        save_volume_btn = widgets.Button(description='Save Volume', button_style='success')

        def update_display():
            self.ax.clear()
            self.ax.imshow(self.image[self.current_slice], cmap='gray')
            self.ax.set_title(f'Slice {self.current_slice + 1}/{self.image.shape[0]}')
            
            # Handle global prompt mode
            if self.use_same_bbox_for_all and self.global_prompt:
                seg = self.infer(self.global_prompt['bbox'], self.current_slice)
                show_mask(seg, self.ax, color=self.global_prompt['color'])
                if show_bbox_checkbox.value:
                    bbox = self.global_prompt['bbox']
                    x_min, y_min, x_max, y_max = bbox
                    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, edgecolor='cyan', facecolor='none', lw=1.5, linestyle='--')
                    self.ax.add_patch(rect)
            # Handle slice-specific prompts
            else:
                prompts_to_show = self.slice_prompts.get(self.current_slice, [])
                for prompt in prompts_to_show:
                    show_mask(prompt['mask'], self.ax, color=prompt['color'])
                    if show_bbox_checkbox.value:
                        bbox = prompt['bbox']
                        x_min, y_min, x_max, y_max = bbox
                        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, edgecolor='cyan', facecolor='none', lw=1.5, linestyle='--')
                        self.ax.add_patch(rect)

            self.ax.axis('off')
            self.fig.canvas.draw_idle()

        def on_press(event):
            if event.inaxes == self.ax:
                self.currently_selecting = True
                self.start_point = (event.xdata, event.ydata)
                self.rect_patch = plt.Rectangle(self.start_point, 0, 0, edgecolor="magenta", facecolor='none', lw=2)
                self.ax.add_patch(self.rect_patch)

        def on_motion(event):
            if self.currently_selecting and event.inaxes == self.ax:
                if self.start_point is None: return
                x0, y0 = self.start_point
                x1, y1 = event.xdata, event.ydata
                self.rect_patch.set_width(x1 - x0)
                self.rect_patch.set_height(y1 - y0)
                self.fig.canvas.draw_idle()

        def on_release(event):
            if self.currently_selecting and event.inaxes == self.ax:
                self.currently_selecting = False
                if self.start_point is None: return
                x0, y0 = self.start_point
                x1, y1 = event.xdata, event.ydata
                self.start_point = None
                self.rect_patch.remove()

                bbox = np.array([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)])
                new_color = np.random.random(3)
                
                if self.use_same_bbox_for_all:
                    self.global_prompt = {'bbox': bbox, 'color': new_color}
                    self.slice_prompts.clear()
                else:
                    seg = self.infer(bbox, self.current_slice)
                    new_prompt = {'mask': seg, 'bbox': bbox, 'color': new_color}
                    if self.current_slice not in self.slice_prompts:
                        self.slice_prompts[self.current_slice] = []
                    self.slice_prompts[self.current_slice].append(new_prompt)
                    self.global_prompt = None

                update_display()
                gc.collect()

        def on_slice_change(change):
            self.current_slice = change['new']
            update_display()

        def on_mode_change(change):
            self.use_same_bbox_for_all = change['new']
            update_display()
        
        def on_checkbox_change(change):
            update_display()

        def on_segment_all_click(b):
            if self.use_same_bbox_for_all and self.global_prompt:
                print("Segmenting all slices with the global bounding box...")
                self.use_same_bbox_for_all = False
                mode_checkbox.value = False
                
                global_p = self.global_prompt
                self.global_prompt = None
                
                for i in range(self.image.shape[0]):
                    seg = self.infer(global_p['bbox'], i)
                    new_prompt = {'mask': seg, 'bbox': global_p['bbox'], 'color': global_p['color']}
                    self.slice_prompts[i] = [new_prompt]
                print("Segmentation complete. Switched to slice-specific mode.")
            else:
                print("To segment all slices, enable 'Use same BBox' and draw a box.")
            update_display()

        def on_clear_click(b):
            self.slice_prompts.pop(self.current_slice, None)
            if self.use_same_bbox_for_all:
                self.global_prompt = None
            update_display()

        def on_clear_all_click(b):
            self.slice_prompts.clear()
            self.global_prompt = None
            update_display()

        def on_save_volume_click(b):
            self.save_segmentation_volume("bbox_seg_volume.npz")

        # --- Connect Events ---
        self.fig.canvas.mpl_connect('button_press_event', on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', on_motion)
        self.fig.canvas.mpl_connect('button_release_event', on_release)
        slice_slider.observe(on_slice_change, names='value')
        mode_checkbox.observe(on_mode_change, names='value')
        show_bbox_checkbox.observe(on_checkbox_change, names='value')
        segment_all_btn.on_click(on_segment_all_click)
        clear_btn.on_click(on_clear_click)
        clear_all_btn.on_click(on_clear_all_click)
        save_volume_btn.on_click(on_save_volume_click)

        # --- Display ---
        controls = VBox([
            HBox([mode_checkbox, show_bbox_checkbox]), 
            HBox([slice_slider, segment_all_btn]), 
            HBox([clear_btn, clear_all_btn, save_volume_btn])
        ])
        display(controls)
        plt.show()
        update_display()

    def get_segmentation_volume(self):
        """
        Constructs the final 3D segmentation volume from individual slice segmentations.
        If multiple segmentations exist on a slice, they are merged with different integer labels.
        
        Returns:
            np.ndarray or None: The complete 3D segmentation volume.
        """
        if not self.is_volume or not self.slice_prompts:
            return None
        
        vol_shape = self.image.shape
        seg_volume = np.zeros(vol_shape, dtype=np.uint8)
        
        for slice_idx, prompts_list in self.slice_prompts.items():
            final_slice_mask = np.zeros(vol_shape[1:], dtype=np.uint8)
            for i, prompt in enumerate(prompts_list, 1):
                final_slice_mask[prompt['mask'] > 0] = i
            seg_volume[slice_idx] = final_slice_mask
            
        return seg_volume

    def save_segmentation_volume(self, filepath):
        """Saves the final 3D segmentation volume to a .npz file."""
        seg_volume = self.get_segmentation_volume()
        if seg_volume is not None:
            np.savez_compressed(filepath, segmentation=seg_volume)
            print(f"Saved segmentation volume to {filepath}")
        else:
            print("No segmentation volume to save.")
            
    def get_segmentation_statistics(self):
        """Calculates and returns statistics about the segmentation."""
        if not self.is_volume:
            return None
        
        seg_volume = self.get_segmentation_volume()
        if seg_volume is None:
            return {
                'total_slices': self.image.shape[0], 'segmented_slices': 0,
                'coverage_percentage': 0, 'total_voxels': np.prod(self.image.shape),
                'segmented_voxels': 0, 'segmentation_percentage': 0,
            }
            
        total_slices = self.image.shape[0]
        segmented_slices = len(self.slice_prompts)
        total_voxels = np.prod(self.image.shape)
        segmented_voxels = np.count_nonzero(seg_volume)
        
        stats = {
            'total_slices': total_slices,
            'segmented_slices': segmented_slices,
            'coverage_percentage': (segmented_slices / total_slices) * 100,
            'total_voxels': total_voxels,
            'segmented_voxels': segmented_voxels,
            'segmentation_percentage': (segmented_voxels / total_voxels) * 100 if total_voxels > 0 else 0,
        }
        return stats

