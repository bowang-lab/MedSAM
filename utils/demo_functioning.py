import gc
import torch
import numpy as np
import cv2
from torch.nn import functional as F
from os import listdir, makedirs, getcwd
from os.path import join, exists, isfile, isdir, basename
from glob import glob
from ipywidgets import interact, widgets, FileUpload
from IPython.display import display
from matplotlib import patches as patches
from matplotlib import pyplot as plt
from copy import deepcopy

def show_mask(mask, ax, random_color=False, alpha=0.95):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


class BboxPromptDemo:
    def __init__(self, model, use_text_prompts=False, dataroot=None):
        self.model = model
        self.model.eval()
        
        # Image data attributes
        self.image = None  # Current 2D slice being displayed
        self.image_data = None  # Full image data (2D or 3D)
        self.is_3d = False
        self.current_slice = 0
        self.num_slices = 1
        self.image_embeddings = None
        self.img_size = None
        
        # UI state
        self.currently_selecting = False
        self.x0, self.y0, self.x1, self.y1 = 0., 0., 0., 0.
        self.rect = None
        self.fig, self.axes = None, None
        
        # Segmentation results
        self.segs = []  # For current slice
        self.all_slice_segs = {}  # For all slices in 3D
        self.global_bbox = None  # Single bbox for all slices
        self.use_global_bbox = False
        
        # Text prompt support (optional)
        self.use_text_prompts = use_text_prompts
        if use_text_prompts:
            from transformers import CLIPTokenizer
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
            self.gt = None
            self.img_name = None
            self.label_dict = {
                1: ["Liver", "liver"],
                2: ["Right Kidney", "right kidney"],
                3: ["Spleen", "spleen"],
                4: ["Pancreas", "pancreas"],
                5: ["Aorta", "aorta"],
                6: ["Inferior Vena Cava", "IVC", "inferior vena cava", "ivc"],
                7: ["Right Adrenal Gland", "RAG", "right adrenal gland", "rag"],
                8: ["Left Adrenal Gland", "LAG", "left adrenal gland", "lag"],
                9: ["Gallbladder", "gallbladder"],
                10: ["Esophagus", "esophagus"],
                11: ["Stomach", "stomach"],
                12: ["Duodenum", "duodenum"],
                13: ["Left Kidney", "left kidney"]
            }
            self.caption_label_dict = {}
            for label_id, label_list in self.label_dict.items():
                for label in label_list:
                    self.caption_label_dict[label] = label_id
            
            if dataroot:
                self.dataroot = dataroot
                self.img_path = join(dataroot, 'imgs')
                self.gt_path = join(dataroot, 'gts_ts')
                self.gt_path_files = sorted(glob(join(self.gt_path, '**/*.npy'), recursive=True))
                self.gt_path_files = [file for file in self.gt_path_files if isfile(join(self.img_path, basename(file)))]

    def _show(self, fig_size=5, random_color=True, alpha=0.65):
        assert self.image is not None, "Please set image first."

        self.fig, self.axes = plt.subplots(1, 1, figsize=(fig_size, fig_size))
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False
        self.fig.canvas.toolbar_visible = False
        self.fig.canvas.resizable = False

        plt.tight_layout()
        self.axes.imshow(self.image)
        self.axes.axis('off')
        
        # Set title with slice info for 3D
        title = f"Slice {self.current_slice}/{self.num_slices-1}" if self.is_3d else "2D Image"
        if self.is_3d and self.use_global_bbox:
            title += " (Global BBox Mode)"
        self.axes.set_title(title)

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
                    if len(self.axes.patches) > 0:
                        self.axes.patches[0].remove()
                    
                    x_min = min(self.x0, self.x1)
                    x_max = max(self.x0, self.x1)
                    y_min = min(self.y0, self.y1)
                    y_max = max(self.y0, self.y1)
                    bbox = np.array([x_min, y_min, x_max, y_max])
                    
                    if self.is_3d and self.use_global_bbox:
                        # Store global bbox for later application to all slices
                        self.global_bbox = bbox
                        print(f"Global bbox stored: {bbox}")
                        print("Use apply_global_bbox_to_all_slices() to apply to all slices")
                    else:
                        # Apply to current slice only
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

        # Create UI controls
        controls = []
        
        # Clear button
        clear_button = widgets.Button(description="Clear")
        def __on_clear_button_clicked(b):
            for i in range(len(self.axes.images)):
                self.axes.images[0].remove()
            self.axes.clear()
            self.axes.axis('off')
            self.axes.imshow(self.image)
            title = f"Slice {self.current_slice}/{self.num_slices-1}" if self.is_3d else "2D Image"
            if self.is_3d and self.use_global_bbox:
                title += " (Global BBox Mode)"
            self.axes.set_title(title)
            if len(self.axes.patches) > 0:
                self.axes.patches[0].remove()
            self.segs = []
            self.fig.canvas.draw_idle()
        controls.append(clear_button)
        clear_button.on_click(__on_clear_button_clicked)

        # Save button
        save_button = widgets.Button(description="Save")
        def __on_save_button_clicked(b):
            if self.is_3d:
                # Save current slice segmentations first
                if self.segs:
                    self.all_slice_segs[self.current_slice] = deepcopy(self.segs)
                
                plt.savefig(f"seg_result_slice_{self.current_slice}.png", bbox_inches='tight', pad_inches=0)
                self.export_3d_segmentation("segmentation_3d.npy")
            else:
                plt.savefig("seg_result.png", bbox_inches='tight', pad_inches=0)
                if len(self.segs) > 0:
                    save_seg = np.zeros_like(self.segs[0])
                    for i, seg in enumerate(self.segs, start=1):
                        save_seg[seg > 0] = i
                    cv2.imwrite("segs.png", save_seg)
                    print(f"Segmentation result saved to {getcwd()}")
        controls.append(save_button)
        save_button.on_click(__on_save_button_clicked)

        # 3D-specific controls
        if self.is_3d:
            # Slice navigation
            slice_slider = widgets.IntSlider(
                value=self.current_slice,
                min=0,
                max=self.num_slices-1,
                step=1,
                description='Slice:',
                continuous_update=False
            )
            def __on_slice_change(change):
                self.go_to_slice(change['new'])
            slice_slider.observe(__on_slice_change, names='value')
            controls.append(slice_slider)
            
            # Global bbox toggle
            global_bbox_toggle = widgets.ToggleButton(
                value=self.use_global_bbox,
                description='Global BBox',
                tooltip='Use single bbox for all slices'
            )
            def __on_global_toggle(change):
                self.set_global_bbox_mode(change['new'])
                # Update title
                title = f"Slice {self.current_slice}/{self.num_slices-1}"
                if self.use_global_bbox:
                    title += " (Global BBox Mode)"
                self.axes.set_title(title)
                self.fig.canvas.draw_idle()
            global_bbox_toggle.observe(__on_global_toggle, names='value')
            controls.append(global_bbox_toggle)
            
            # Apply global bbox button
            apply_global_button = widgets.Button(description="Apply Global BBox")
            def __on_apply_global_clicked(b):
                self.apply_global_bbox_to_all_slices()
                self._refresh_display()
            controls.append(apply_global_button)
            apply_global_button.on_click(__on_apply_global_clicked)

        # Display controls
        controls_box = widgets.HBox(controls)
        display(controls_box)

        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', __on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', __on_motion)
        self.fig.canvas.mpl_connect('button_release_event', __on_release)

        plt.show()

    def show(self, image_data=None, image_path=None, fig_size=5, random_color=True, alpha=0.65):
        """
        Show the interactive bbox demo
        
        Args:
            image_data: numpy array (H, W) for 2D or (D, H, W) for 3D. Takes precedence over image_path.
            image_path: path to image file (for backward compatibility)
            fig_size: figure size
            random_color: use random colors for masks
            alpha: mask transparency
        """
        if image_data is not None:
            self.set_image(image_data)
        elif image_path is not None:
            self.set_image_path(image_path)
        else:
            assert self.image is not None, "Please provide image_data, image_path, or call set_image_data first."
        
        self._show(fig_size=fig_size, random_color=random_color, alpha=alpha)

    def set_image_path(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self._set_image(image)
    
    def _set_image(self, image):
        self.image = image
        self.img_size = image.shape[:2]
        image_preprocess = self._preprocess_image(image)
        with torch.no_grad():
            self.image_embeddings = self.model.image_encoder(image_preprocess)

    def _preprocess_image(self, image):
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
    
    @torch.no_grad()
    def _infer(self, bbox, return_raw=False):
        ori_H, ori_W = self.img_size
        scale_to_1024 = 1024 / np.array([ori_W, ori_H, ori_W, ori_H])
        bbox_1024 = bbox * scale_to_1024
        bbox_torch = torch.as_tensor(bbox_1024, dtype=torch.float).unsqueeze(0).to(self.model.device)
        if len(bbox_torch.shape) == 2:
            bbox_torch = bbox_torch.unsqueeze(1)
    
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=None,
            boxes=bbox_torch,
            masks=None,
        )
        low_res_logits, _ = self.model.mask_decoder(
            image_embeddings = self.image_embeddings, # (B, 256, 64, 64)
            image_pe = self.model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings = sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
            )

        low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

        low_res_pred = F.interpolate(
            low_res_pred,
            size=self.img_size,
            mode="bilinear",
            align_corners=False,
        )  # (1, 1, gt.shape)
        low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
        medsam_seg = (low_res_pred > 0.5).astype(np.uint8)

        if return_raw:
            return medsam_seg, low_res_pred
        else:
            return medsam_seg

    def infer(self, bbox, return_raw=False):
        return self._infer(bbox, return_raw=True)

    def set_image(self, image_data):
        """
        Set image data directly (2D or 3D numpy array)
        
        Args:
            image_data: numpy array of shape (H, W) for 2D or (D, H, W) for 3D
        """
        self.image_data = np.array(image_data)
        
        if len(self.image_data.shape) == 2:
            # 2D image
            self.is_3d = False
            self.num_slices = 1
            self.current_slice = 0
            self._set_current_slice(self.image_data)
        elif len(self.image_data.shape) == 3:
            # 3D volume
            self.is_3d = True
            self.num_slices = self.image_data.shape[0]
            self.current_slice = self.num_slices // 2  # Start at middle slice
            self.all_slice_segs = {}
            self._set_current_slice(self.image_data[self.current_slice])
        else:
            raise ValueError("Image data must be 2D (H, W) or 3D (D, H, W)")
    
    def _set_current_slice(self, slice_data):
        """Set the current 2D slice for display and processing"""
        # Ensure the slice is 2D
        if len(slice_data.shape) == 2:
            # Convert grayscale to RGB for display
            self.image = cv2.cvtColor(slice_data.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif len(slice_data.shape) == 3 and slice_data.shape[2] == 3:
            # Already RGB
            self.image = slice_data.astype(np.uint8)
        else:
            raise ValueError("Slice must be 2D grayscale or 3D RGB")
        
        self.img_size = self.image.shape[:2]
        
        # Get embeddings for current slice
        image_preprocess = self._preprocess_image(self.image)
        with torch.no_grad():
            self.image_embeddings = self.model.image_encoder(image_preprocess)
        
        # Load segmentations for current slice if available
        if self.is_3d and self.current_slice in self.all_slice_segs:
            self.segs = self.all_slice_segs[self.current_slice]
        else:
            self.segs = []

    def go_to_slice(self, slice_idx):
        """Navigate to a specific slice in 3D volume"""
        if not self.is_3d:
            print("Not a 3D volume")
            return
        
        if 0 <= slice_idx < self.num_slices:
            # Save current slice segmentations
            if self.segs:
                self.all_slice_segs[self.current_slice] = deepcopy(self.segs)
            
            self.current_slice = slice_idx
            self._set_current_slice(self.image_data[self.current_slice])
            
            # Refresh display if it exists
            if hasattr(self, 'fig') and self.fig is not None:
                self._refresh_display()
        else:
            print(f"Slice index {slice_idx} out of range [0, {self.num_slices-1}]")

    def _refresh_display(self):
        """Refresh the display with current slice"""
        if hasattr(self, 'axes') and self.axes is not None:
            self.axes.clear()
            self.axes.imshow(self.image)
            self.axes.axis('off')
            self.axes.set_title(f"Slice {self.current_slice}/{self.num_slices-1}" if self.is_3d else "2D Image")
            
            # Redraw existing segmentations
            for seg in self.segs:
                show_mask(seg, self.axes, random_color=True, alpha=0.65)
            
            self.fig.canvas.draw_idle()

    def set_global_bbox_mode(self, use_global=True):
        """
        Set whether to use a single bounding box for all slices (3D only)
        
        Args:
            use_global: If True, one bbox will be applied to all slices
        """
        if not self.is_3d:
            print("Global bbox mode only available for 3D volumes")
            return
        
        self.use_global_bbox = use_global
        if use_global:
            print("Global bbox mode enabled: Draw one bbox to apply to all slices")
        else:
            print("Individual bbox mode enabled: Draw bbox for each slice separately")

    def apply_global_bbox_to_all_slices(self):
        """Apply the stored global bbox to all slices in the volume"""
        if not self.is_3d or self.global_bbox is None:
            print("No global bbox available or not in 3D mode")
            return
        
        print(f"Applying global bbox to all {self.num_slices} slices...")
        current_slice_backup = self.current_slice
        
        for slice_idx in range(self.num_slices):
            self.go_to_slice(slice_idx)
            with torch.no_grad():
                seg = self._infer(self.global_bbox)
                torch.cuda.empty_cache()
            
            if slice_idx not in self.all_slice_segs:
                self.all_slice_segs[slice_idx] = []
            self.all_slice_segs[slice_idx].append(deepcopy(seg))
        
        # Return to original slice
        self.go_to_slice(current_slice_backup)
        print("Global bbox applied to all slices")

    def get_all_segmentations(self):
        """
        Get segmentations for all slices
        
        Returns:
            dict: Dictionary mapping slice indices to lists of segmentations
        """
        # Include current slice segmentations
        if self.segs:
            self.all_slice_segs[self.current_slice] = deepcopy(self.segs)
        
        return self.all_slice_segs

    def export_3d_segmentation(self, filename="segmentation_3d.npy"):
        """
        Export 3D segmentation volume
        
        Args:
            filename: Output filename
        """
        if not self.is_3d:
            print("Not a 3D volume")
            return
        
        # Create 3D segmentation volume
        seg_volume = np.zeros_like(self.image_data, dtype=np.uint8)
        
        # Include current slice
        if self.segs:
            self.all_slice_segs[self.current_slice] = deepcopy(self.segs)
        
        for slice_idx, slice_segs in self.all_slice_segs.items():
            if slice_segs:
                # Combine multiple segmentations in slice
                combined_seg = np.zeros_like(slice_segs[0])
                for i, seg in enumerate(slice_segs, start=1):
                    combined_seg[seg > 0] = i
                seg_volume[slice_idx] = combined_seg
        
        np.save(filename, seg_volume)
        print(f"3D segmentation saved to {filename}")

    # Text prompt methods (only available if use_text_prompts=True)
    def show_mask(self, mask, ax, random_color=False, alpha=0.95):
        if not self.use_text_prompts:
            print("Text prompt methods not available. Initialize with use_text_prompts=True")
            return
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
        else:
            color = np.array([251/255, 252/255, 30/255, alpha])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    
    @torch.no_grad()
    def infer_text(self, text):
        if not self.use_text_prompts:
            print("Text prompt methods not available. Initialize with use_text_prompts=True")
            return None
        tokens = self.tokenize_text(text).to(self.model.device)
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points = None,
            boxes = None,
            masks = None,
            tokens = tokens
        )
        low_res_logits, _ = self.model.mask_decoder(
            image_embeddings=self.image_embeddings, # (B, 256, 64, 64)
            image_pe=self.model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
            )
        low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
        low_res_pred = F.interpolate(
            low_res_pred,
            size = self.img_size,
            mode = 'bilinear',
            align_corners = False
        )
        low_res_pred = low_res_pred.detach().cpu().numpy().squeeze()
        seg = np.uint8(low_res_pred > 0.5)

        return seg

    def show_text_prompt_interface(self, fig_size=3, alpha=0.95):
        if not self.use_text_prompts:
            print("Text prompt interface not available. Initialize with use_text_prompts=True")
            return
        
        assert self.image is not None, "Please set image first."
        fig, axes = plt.subplots(1, 2, figsize=(2 * fig_size, fig_size))
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.canvas.toolbar_visible = False
        fig.canvas.resizable = False

        if hasattr(self, 'gt') and self.gt is not None:
            avil_ids = np.unique(self.gt)[1:]
            avail_prompts = []
            for id in avil_ids:
                if id in self.label_dict:
                    avail_prompts += self.label_dict[id]
            print("Possible prompts: ", avail_prompts)

        plt.tight_layout()

        for i in range(2):
            axes[i].imshow(self.image)
            axes[i].axis('off')
            if i == 0:
                axes[i].set_title('Ground Truth')
            else:
                axes[i].set_title('Segmentation')

        text = widgets.Text(
            value='',
            placeholder='Prompt',
            description='Prompt',
            disabled=False
        )
        display(text)

        def callback(wget):
            for i in range(2):
                axes[i].clear()
                axes[i].imshow(self.image)
                axes[i].axis('off')
            caption = wget.value
            seg = self.infer_text(caption)
            if seg is not None:
                axes[1].set_title('Segmentation')
                self.show_mask(seg, axes[1], random_color=False, alpha=alpha)

                if hasattr(self, 'gt') and self.gt is not None:
                    axes[0].set_title('Ground Truth')
                    try:
                        gt_label_id = self.caption_label_dict[caption]
                    except:
                        gt_label_id = self.guess_gt_label_id(self.gt, seg)
                    if gt_label_id in self.label_dict:
                        gt_show = np.uint8(self.gt == gt_label_id)
                        self.show_mask(gt_show, axes[0], random_color=False, alpha=alpha)
                    else:
                        axes[0].clear()
                        axes[0].imshow(self.image)
                        axes[0].axis('off')
                        axes[0].set_title('Ground Truth')

        text.on_submit(callback)

    def guess_gt_label_id(self, gt, seg):
        if not self.use_text_prompts:
            return None
        mask_area = seg > 0
        gt_area = gt[mask_area]
        gt_label_id = np.argmax(np.bincount(gt_area))
        return gt_label_id

    def set_image_text(self, image_index):
        if not self.use_text_prompts:
            print("set_image method not available. Use set_image instead or initialize with use_text_prompts=True")
            return
        image_path = join(self.img_path, basename(self.gt_path_files[image_index]))
        image = np.load(image_path)
        self.image = image
        self.img_size = image.shape[:2]
        self.img_name = basename(image_path)
        image_preprocess = self.preprocess_image(image)
        with torch.no_grad():
            self.image_embeddings = self.model.image_encoder(image_preprocess)
        
        gt_path = self.gt_path_files[image_index]
        gt = np.load(gt_path)
        gt_resize = cv2.resize(
            gt,
            (self.img_size[1], self.img_size[0]),
            interpolation=cv2.INTER_NEAREST
        )
        self.gt = gt_resize

    def preprocess_image(self, image):
        if not self.use_text_prompts:
            print("preprocess_image method not available. Use _preprocess_image instead or initialize with use_text_prompts=True")
            return None
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
        if not self.use_text_prompts:
            print("Text tokenization not available. Initialize with use_text_prompts=True")
            return None
        return self.tokenizer(
            text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt" 
        ).input_ids.squeeze(1)