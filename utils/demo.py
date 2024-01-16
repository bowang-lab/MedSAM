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
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.image = None
        self.image_embeddings = None
        self.img_size = None
        self.gt = None
        self.currently_selecting = False
        self.x0, self.y0, self.x1, self.y1 = 0., 0., 0., 0.
        self.rect = None
        self.fig, self.axes = None, None
        self.segs = []

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
                    self.axes.patches[0].remove()
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
                self.axes.images[0].remove()
            self.axes.clear()
            self.axes.axis('off')
            self.axes.imshow(self.image)
            if len(self.axes.patches) > 0:
                self.axes.patches[0].remove()
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

    def show(self, image_path, fig_size=5, random_color=True, alpha=0.65):
        self.set_image_path(image_path)
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
    def _infer(self, bbox):
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
        return medsam_seg



class PointPromptDemo:
    def __init__(self, model, dataroot):
        self.model = model
        self.model.eval()
        self.image = None
        self.image_embeddings = None
        self.img_size = None
        self.img_name = None
        self.gt = None
        
        ## load demo data
        self.dataroot = dataroot
        self.img_path = join(dataroot, 'imgs')
        self.gt_path = join(dataroot, 'gts_ts')
        self.gt_path_files = sorted(glob(join(self.gt_path, '**/*.npy'), recursive=True))
        self.gt_path_files = [file for file in self.gt_path_files if isfile(join(self.img_path, basename(file)))]

    def show_mask(self, mask, ax, random_color=False, alpha=0.95):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
        else:
            color = np.array([251/255, 252/255, 30/255, alpha])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    @torch.no_grad()
    def infer(self, x, y):
        coords_1024 = np.array([[[
            x * 1024 / self.img_size[1],
            y * 1024 / self.img_size[0]
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
            image_embeddings=self.image_embeddings, # (B, 256, 64, 64)
            image_pe=self.model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
        )

        low_res_probs = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
        low_res_pred = F.interpolate(
            low_res_probs,
            size = self.img_size,
            mode = 'bilinear',
            align_corners = False
        )
        low_res_pred = low_res_pred.detach().cpu().numpy().squeeze()

        seg = np.uint8(low_res_pred > 0.5)

        return seg

    def show(self, fig_size=3, alpha=0.95, scatter_size=25):

        assert self.image is not None, "Please set image first."
        fig, axes = plt.subplots(1, 2, figsize=(2 * fig_size, fig_size))
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.canvas.toolbar_visible = False
        fig.canvas.resizable = False

        plt.tight_layout()

        for i in range(2):
            axes[i].imshow(self.image)
            axes[i].axis('off')
            if i == 0:
                axes[i].set_title('Ground Truth')
            else:
                axes[i].set_title('Segmentation')

        def onclick(event):
            if event.inaxes == axes[1]:
                x, y = float(event.xdata), float(event.ydata)
                with torch.no_grad():
                    ## rescale x, y from canvas size to 1024 x 1024
                    seg = self.infer(x, y)

                for i in range(2):
                    axes[i].clear()
                    axes[i].imshow(self.image)
                    axes[i].axis('off')
                    axes[i].scatter(x, y, c='r', s=scatter_size)

                axes[1].set_title('Segmentation')
                self.show_mask(seg, axes[1], random_color=False, alpha=alpha)

                axes[0].set_title('Ground Truth')
                gt_label_id = self.get_label_id((x, y))
                if gt_label_id > 0:
                    gt_show = np.uint8(self.gt == gt_label_id)
                    self.show_mask(gt_show, axes[0], random_color=False, alpha=alpha)

                gc.collect()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

    def set_image(self, image_index):
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
    
    def get_label_id(self, coords):
        x, y = coords
        label_id = self.gt[int(y), int(x)]

        return label_id


class TextPromptDemo:
    def __init__(self, model, dataroot):
        ## Requires transformers only if users would like to try text prompt
        from transformers import CLIPTokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")   
        self.model = model
        self.model.eval()
        self.image = None
        self.image_embeddings = None
        self.img_size = None
        self.img_name = None
        self.gt = None
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
        
        ## load demo data
        self.dataroot = dataroot
        self.img_path = join(dataroot, 'imgs')
        self.gt_path = join(dataroot, 'gts_ts')
        self.gt_path_files = sorted(glob(join(self.gt_path, '**/*.npy'), recursive=True))
        self.gt_path_files = [file for file in self.gt_path_files if isfile(join(self.img_path, basename(file)))]

    def show_mask(self, mask, ax, random_color=False, alpha=0.95):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
        else:
            color = np.array([251/255, 252/255, 30/255, alpha])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    
    @torch.no_grad()
    def infer(self, text):
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

    def show(self, fig_size=3, alpha=0.95):
        assert self.image is not None, "Please set image first."
        fig, axes = plt.subplots(1, 2, figsize=(2 * fig_size, fig_size))
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.canvas.toolbar_visible = False
        fig.canvas.resizable = False

        avil_ids = np.unique(self.gt)[1:]
        avail_prompts = []
        for id in avil_ids:
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
            seg = self.infer(caption)
            axes[1].set_title('Segmentation')
            self.show_mask(seg, axes[1], random_color=False, alpha=alpha)

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
        mask_area = seg > 0
        gt_area = gt[mask_area]
        gt_label_id = np.argmax(np.bincount(gt_area))

        return gt_label_id

    def set_image(self, image_index):
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