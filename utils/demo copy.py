import gc
import torch
import numpy as np
import cv2
from torch.nn import functional as F
from os import listdir, makedirs, getcwd
from os.path import join, exists, isfile, isdir, basename
from glob import glob
from ipywidgets import interact, widgets, FileUpload, IntSlider, VBox, HBox
from IPython.display import display
from matplotlib import patches as patches
from matplotlib import pyplot as plt
from copy import deepcopy

def show_mask(mask, ax, random_color=False, alpha=0.95, color=None):
    if color is not None:
        pass
    elif random_color:
        color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


class BboxPromptDemo:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.image = None  # Can be 2D (H, W) or 3D (D, H, W)
        self.image_embeddings = None  # For 3D: list of embeddings per slice
        self.img_size = None
        self.gt = None
        self.currently_selecting = False
        self.x0, self.y0, self.x1, self.y1 = 0., 0., 0., 0.
        self.rect = None
        self.fig, self.axes = None, None
        self.segs = []
        
        # 3D volume support
        self.is_volume = False
        self.current_slice = 0
        self.segmentations = {}  # Store segmentations for each bbox and slice
        self.bbox_prompts = {}  # Store bbox coordinates for each slice
        self.use_same_bbox_for_all = False
        self.global_bbox = None  # Bbox to use for all slices when use_same_bbox_for_all is True

    def show(self, image=None, fig_size=5, random_color=True, alpha=0.65):
        """
        Display the interactive segmentation interface.
        For 2D images: shows single image with bbox prompts
        For 3D volumes: shows image with bbox prompts and slice slider
        """
        if image is not None:
            self.set_image(image)
        
        assert self.image is not None, "Please set image first."
        
        if self.is_volume:
            self._show_volume(fig_size, random_color, alpha)
        else:
            self._show_2d(fig_size, random_color, alpha)

    def _show_2d(self, fig_size=5, random_color=True, alpha=0.65):
        """Original 2D display functionality"""
        assert self.image is not None, "Please set image first."

        self.fig, self.axes = plt.subplots(1, 1, figsize=(fig_size, fig_size))
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False
        self.fig.canvas.toolbar_visible = False
        self.fig.canvas.resizable = False

        plt.tight_layout()
        
        # Display image with proper extent to ensure coordinate mapping
        if len(self.image.shape) == 2:
            self.axes.imshow(self.image, cmap='gray', extent=[0, self.img_size[1], self.img_size[0], 0])
        else:
            self.axes.imshow(self.image, extent=[0, self.img_size[1], self.img_size[0], 0])
        
        self.axes.set_xlim(0, self.img_size[1])
        self.axes.set_ylim(self.img_size[0], 0)  # Invert y-axis to match image coordinates
        self.axes.axis('off')

        def __on_press(event):
            if event.inaxes == self.axes:
                self.x0 = float(event.xdata) 
                self.y0 = float(event.ydata)
                self.currently_selecting = True
                self.rect = plt.Rectangle(
                    (self.x0, self.y0),
                    1,1, linestyle="--",
                    edgecolor="crimson", fill=False
                )
                self.axes.add_patch(self.rect)
                self.rect.set_visible(False)

        def __on_release(event):
            if event.inaxes == self.axes:
                if self.currently_selecting:
                    self.x1 = float(event.xdata)
                    self.y1 = float(event.ydata)
                    self.fig.canvas.draw_idle()
                    self.currently_selecting = False
                    self.rect.set_visible(False)
                    if self.rect in self.axes.patches:
                        self.rect.remove()
                    x_min = min(self.x0, self.x1)
                    x_max = max(self.x0, self.x1)
                    y_min = min(self.y0, self.y1)
                    y_max = max(self.y0, self.y1)
                    bbox = np.array([x_min, y_min, x_max, y_max])
                    with torch.no_grad():
                        seg = self._infer(bbox)
                        torch.cuda.empty_cache()
                    show_mask(seg, self.axes, random_color=random_color, alpha=alpha)
                    self.segs.append(deepcopy(seg))
                    del seg
                    self.rect = None
                    gc.collect()

        def __on_motion(event):
            if event.inaxes == self.axes:
                if self.currently_selecting:
                    self.x1 = float(event.xdata)
                    self.y1 = float(event.ydata)
                    #add rectangle for selection here
                    self.rect.set_visible(True)
                    xlim = np.sort([self.x0, self.x1])
                    ylim = np.sort([self.y0, self.y1])
                    self.rect.set_xy((xlim[0],ylim[0] ) )
                    rect_width = np.diff(xlim)[0]
                    self.rect.set_width(rect_width)
                    rect_height = np.diff(ylim)[0]
                    self.rect.set_height(rect_height)

        clear_button = widgets.Button(description="clear")
        def __on_clear_button_clicked(b):
            for i in range(len(self.axes.images)):
                if self.axes.images:
                    self.axes.images[0].remove()
            self.axes.clear()
            self.axes.axis('off')
            if len(self.image.shape) == 2:
                self.axes.imshow(self.image, cmap='gray', extent=[0, self.img_size[1], self.img_size[0], 0])
            else:
                self.axes.imshow(self.image, extent=[0, self.img_size[1], self.img_size[0], 0])
            self.axes.set_xlim(0, self.img_size[1])
            self.axes.set_ylim(self.img_size[0], 0)
            self.segs = []
            self.fig.canvas.draw_idle()

        save_button = widgets.Button(description="save")
        def __on_save_button_clicked(b):
            plt.savefig("seg_result.png", bbox_inches='tight', pad_inches=0)
            if len(self.segs) > 0:
                save_seg = np.zeros_like(self.segs[0])
                for i, seg in enumerate(self.segs, start=1):
                    save_seg[seg > 0] = i
                cv2.imwrite("segs.png", save_seg)
                print(f"Segmentation result saved to {getcwd()}")
        
        display(clear_button)
        clear_button.on_click(__on_clear_button_clicked)

        self.fig.canvas.mpl_connect('button_press_event', __on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', __on_motion)
        self.fig.canvas.mpl_connect('button_release_event', __on_release)

        plt.show()

        display(save_button)
        save_button.on_click(__on_save_button_clicked)
    def _show_volume(self, fig_size=5, random_color=True, alpha=0.65):
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
            value=self.use_same_bbox_for_all,
            description='Use same bbox for all slices',
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
            tooltip='Segment all slices with current/individual bboxes'
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
        
        # Bbox selection state variables
        currently_selecting = False
        rect = None
        x0, y0, x1, y1 = 0., 0., 0., 0.
        
        def update_display():
            """Update the display based on current slice and bbox"""
            ax.clear()
            current_image = self.image[self.current_slice]
            current_shape = current_image.shape  # (H, W)
            
            # Display with proper extent for coordinate mapping
            if len(current_image.shape) == 2:
                ax.imshow(current_image, cmap='gray', extent=[0, current_shape[1], current_shape[0], 0])
            else:
                ax.imshow(current_image, extent=[0, current_shape[1], current_shape[0], 0])
            
            ax.set_xlim(0, current_shape[1])
            ax.set_ylim(current_shape[0], 0)  # Invert y-axis to match image coordinates
            ax.axis('off')
            ax.set_title(f'Slice {self.current_slice + 1}/{self.image.shape[0]}')
            
            # Show segmentation if available
            if self.current_slice in self.segmentations:
                segs = self.segmentations[self.current_slice]
                for seg, color in segs:
                    show_mask(seg, ax, color=color, alpha=alpha)
            
            fig.canvas.draw()
        
        def on_press(event):
            """Handle mouse press events for bbox selection"""
            nonlocal currently_selecting, rect, x0, y0
            if event.inaxes == ax:
                x0 = float(event.xdata)
                y0 = float(event.ydata)
                currently_selecting = True
                rect = plt.Rectangle((x0, y0), 1, 1, linestyle="--", edgecolor="crimson", fill=False)
                ax.add_patch(rect)
                rect.set_visible(False)
        
        def on_motion(event):
            """Handle mouse motion events for bbox selection"""
            nonlocal currently_selecting, rect, x0, y0, x1, y1
            if event.inaxes == ax and currently_selecting:
                x1 = float(event.xdata)
                y1 = float(event.ydata)
                rect.set_visible(True)
                xlim = np.sort([x0, x1])
                ylim = np.sort([y0, y1])
                rect.set_xy((xlim[0], ylim[0]))
                rect.set_width(np.diff(xlim)[0])
                rect.set_height(np.diff(ylim)[0])
                fig.canvas.draw_idle()
        
        def on_release(event):
            """Handle mouse release events for bbox selection"""
            nonlocal currently_selecting, rect, x0, y0, x1, y1
            if event.inaxes == ax and currently_selecting:
                x1 = float(event.xdata)
                y1 = float(event.ydata)
                currently_selecting = False
                rect.set_visible(False)
                if rect in ax.patches:
                    rect.remove()
                
                x_min = min(x0, x1)
                x_max = max(x0, x1)
                y_min = min(y0, y1)
                y_max = max(y0, y1)
                bbox = np.array([x_min, y_min, x_max, y_max])
                
                # Store bbox based on mode
                if self.use_same_bbox_for_all:
                    self.global_bbox = bbox
                else:
                    if self.current_slice not in self.bbox_prompts:
                        self.bbox_prompts[self.current_slice] = []
                    self.bbox_prompts[self.current_slice].append(bbox)
                
                # Perform segmentation
                with torch.no_grad():
                    seg = self._infer(bbox, slice_idx=self.current_slice)
                    torch.cuda.empty_cache()
                
                if self.current_slice not in self.segmentations:
                    self.segmentations[self.current_slice] = []
                color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
                self.segmentations[self.current_slice].append((seg, color))
                update_display()
                del seg
                rect = None
                gc.collect()
        
        def on_slice_change(change):
            """Handle slice slider changes"""
            self.current_slice = change['new']
            update_display()
        
        def on_mode_change(change):
            """Handle mode checkbox changes"""
            self.use_same_bbox_for_all = change['new']
            update_display()
        
        def on_segment_all_click(btn):
            """Segment all slices based on mode"""
            if self.use_same_bbox_for_all:
                if self.global_bbox is not None:
                    print("Segmenting all slices with global bbox...")
                    for slice_idx in range(self.image.shape[0]):
                        with torch.no_grad():
                            seg = self._infer(self.global_bbox, slice_idx=slice_idx)
                        if slice_idx not in self.segmentations:
                            self.segmentations[slice_idx] = []
                        color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
                        self.segmentations[slice_idx].append((seg, color))
                    print(f"Segmented {self.image.shape[0]} slices")
                    update_display()
                else:
                    print("No global bbox defined. Please draw a bbox first.")
            else:
                if len(self.bbox_prompts) > 0:
                    print("Segmenting slices with individual bboxes...")
                    for slice_idx, bboxes in self.bbox_prompts.items():
                        if slice_idx not in self.segmentations:
                            self.segmentations[slice_idx] = []
                        for bbox in bboxes:
                            with torch.no_grad():
                                seg = self._infer(bbox, slice_idx=slice_idx)
                            color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
                            self.segmentations[slice_idx].append((seg, color))
                    print(f"Segmented {len(self.bbox_prompts)} slices")
                    update_display()
                else:
                    print("No individual bboxes defined. Please draw bboxes first.")
        
        def on_clear_click(btn):
            """Clear current slice segmentation"""
            if self.current_slice in self.segmentations:
                del self.segmentations[self.current_slice]
            if not self.use_same_bbox_for_all and self.current_slice in self.bbox_prompts:
                del self.bbox_prompts[self.current_slice]
            update_display()
        
        def on_clear_all_click(btn):
            """Clear all segmentations and bboxes"""
            self.segmentations.clear()
            self.bbox_prompts.clear()
            self.global_bbox = None
            update_display()
        
        def on_save_slice_click(btn):
            """Save current slice segmentation"""
            if self.current_slice in self.segmentations:
                filename = f"bbox_seg_slice_{self.current_slice}.png"
                segs_to_save = self.segmentations[self.current_slice]
                if len(segs_to_save) > 0:
                    save_seg = np.zeros_like(segs_to_save[0][0])
                    for i, (seg, _) in enumerate(segs_to_save, start=1):
                        save_seg[seg > 0] = i
                    cv2.imwrite(filename, save_seg)
                    print(f"Saved current slice segmentation to {filename}")
                else:
                    print("No segmentation available for current slice")
            else:
                print("No segmentation available for current slice")
        
        def on_save_volume_click(btn):
            """Save complete 3D segmentation volume"""
            if len(self.segmentations) > 0:
                filename = f"bbox_seg_volume_{len(self.segmentations)}slices.npz"
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
        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('button_release_event', on_release)
        
        # Create layout
        controls = VBox([
            mode_checkbox,
            HBox([slice_slider, segment_all_btn]),
            HBox([clear_btn, clear_all_btn]),
            HBox([save_slice_btn, save_volume_btn]),
            info_label
        ])
        
        # Display widgets
        display(controls)
        
        # Show the plot
        plt.show()
        
        # Initial display
        update_display()

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
                np.savez_compressed(filepath, segmentation=seg_volume)
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
            'mode': 'same_bbox_for_all' if self.use_same_bbox_for_all else 'individual_bboxes'
        }
        
        return stats
    
    def set_image(self, image):
        """
        Set the image or volume to be segmented.
        
        Args:
            image: 2D numpy array (H, W) or 3D numpy array (D, H, W)
        """
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                self.image = image
                self.img_size = image.shape
                self.is_volume = False
            elif len(image.shape) == 3:
                self.image = image
                self.img_size = image.shape[1:]
                self.is_volume = True
                self.image_embeddings = [self.model.encode_image(image_slice) for image_slice in image]
            else:
                raise ValueError("Image must be 2D (H, W) or 3D (D, H, W) numpy array")
        else:
            raise TypeError("Image must be a numpy array")
        
    def _infer(self, bbox, slice_idx=None):
        """
        Perform inference on the given bounding box.
        
        Args:
            bbox: numpy array of shape (4,) with [x_min, y_min, x_max, y_max]
            slice_idx: index of the slice to use for 3D volumes (if applicable)
        
        Returns:
            segmentation mask as a numpy array
        """
        if self.is_volume and slice_idx is not None:
            image_slice = self.image[slice_idx]
            image_embedding = self.image_embeddings[slice_idx]
        else:
            image_slice = self.image
            image_embedding = self.model.encode_image(image_slice)
        
        # Convert bbox to tensor
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32).unsqueeze(0).to(self.model.device)
        
        # Perform inference
        with torch.no_grad():
            seg = self.model.predict(image_embedding, bbox_tensor)
        
        return seg.cpu().numpy()
    
    


