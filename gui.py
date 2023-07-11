# -*- coding: utf-8 -*-
import sys
import time
import random

from PyQt5.QtGui import (
    QBrush,
    QPainter,
    QPen,
    QPixmap,
    QKeySequence,
    QPen,
    QBrush,
    QColor,
    QImage,
)
from PyQt5.QtWidgets import (
    QFileDialog,
    QApplication,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsPixmapItem,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
    QShortcut,
)

import numpy as np
from skimage import transform, io
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

# freeze seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()
torch.cuda.manual_seed(2023)
np.random.seed(2023)

SAM_MODEL_TYPE = "vit_b"
MedSAM_CKPT_PATH = "./work_dir/sam_vit_b_medsam.pth"
MEDSAM_IMG_INPUT_SIZE = 1024
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, height, width):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


print("Loading MedSam model, a sec")
tic = time.perf_counter()

# %% set up model
medsam_model = sam_model_registry["vit_b"](checkpoint=MedSAM_CKPT_PATH).to(device)
medsam_model.eval()

print(f"MedSam loaded, took {time.perf_counter() - tic}")


def np2pixmap(np_img):
    height, width, channel = np_img.shape
    bytesPerLine = 3 * width
    qImg = QImage(np_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
    return QPixmap.fromImage(qImg)


colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (255, 255, 255),
    (192, 192, 192),
    (64, 64, 64),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
    (0, 0, 127),
    (192, 0, 192),
]


class Window(QWidget):
    def __init__(self):
        super().__init__()

        # configs
        self.half_point_size = 5  # radius of bbox starting and ending points

        # app stats
        self.image_path = None
        self.color_idx = 0
        self.bg_img = None
        self.is_mouse_down = False
        self.rect = None
        self.point_size = self.half_point_size * 2
        self.start_point = None
        self.end_point = None
        self.start_pos = (None, None)
        self.embedding = None
        self.prev_mask = None
        self.sam_model = None

        self.scene = QGraphicsScene(0, 0, 800, 800)

        self.load_image()

        view = QGraphicsView(self.scene)
        view.setRenderHint(QPainter.Antialiasing)

        vbox = QVBoxLayout(self)
        vbox.addWidget(view)

        load_button = QPushButton("Load Image")
        save_button = QPushButton("Save Mask")
        compare_button = QPushButton("Compare with SAM")
        compare_button.setCheckable(True)
        compare_button.setChecked(False)

        hbox = QHBoxLayout(self)
        hbox.addWidget(load_button)
        hbox.addWidget(save_button)
        hbox.addWidget(compare_button)
        self.compare_button = compare_button

        vbox.addLayout(hbox)

        self.setLayout(vbox)

        # keyboard shortcuts
        self.quit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), self)
        self.quit_shortcut.activated.connect(lambda: quit())

        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.undo)

        load_button.clicked.connect(self.load_image)
        save_button.clicked.connect(self.save_mask)
        compare_button.clicked.connect(self.toggle_sam)

        # events
        self.scene.mousePressEvent = self.mouse_press
        self.scene.mouseMoveEvent = self.mouse_move
        self.scene.mouseReleaseEvent = self.mouse_release

    def toggle_sam(self):
        if self.sam_model is None:
            print("Loading FAIR origional SAM model, a sec.", end=" ")
            sam = sam_model_registry["vit_b"](
                checkpoint="work_dir/SAM/sam_vit_b_01ec64.pth"
            ).to(device)
            self.sam_model = SamPredictor(sam)
            print("Done")

    def undo(self):
        if self.prev_mask is None:
            print("No previous mask record")
            return

        self.color_idx -= 1

        bg = Image.fromarray(self.img_3c.astype("uint8"), "RGB")
        mask = Image.fromarray(self.prev_mask.astype("uint8"), "RGB")
        img = Image.blend(bg, mask, 0.2)

        self.scene.removeItem(self.bg_img)
        self.bg_img = self.scene.addPixmap(np2pixmap(np.array(img)))

        self.medsam_mask_c = self.prev_mask
        self.prev_mask = None

    def load_image(self):
        file_path, file_type = QFileDialog.getOpenFileName(
            self, "Choose Image to Segment", ".", "Image Files (*.png *.jpg *.bmp)"
        )

        if file_path is None or len(file_path) == 0:
            print("No image path specified, plz select an image")
            exit()

        img_np = io.imread(file_path)
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np
        H, W, _ = img_3c.shape

        self.img_3c = img_3c
        self.image_path = file_path

        pixmap = np2pixmap(self.img_3c)

        self.scene.clear()
        self.end_point = None
        self.rect = None
        self.bg_img = self.scene.addPixmap(pixmap)
        self.bg_img.setPos(0, 0)
        self.medsam_mask_c = np.zeros((*self.img_3c.shape[:2], 3), dtype="uint8")
        self.sam_mask_c = np.zeros((*self.img_3c.shape[:2], 3), dtype="uint8")

    def mouse_press(self, ev):
        x, y = ev.scenePos().x(), ev.scenePos().y()
        self.is_mouse_down = True
        self.start_pos = ev.scenePos().x(), ev.scenePos().y()
        self.start_point = self.scene.addEllipse(
            x - self.half_point_size,
            y - self.half_point_size,
            self.point_size,
            self.point_size,
            pen=QPen(QColor("red")),
            brush=QBrush(QColor("red")),
        )

    def mouse_move(self, ev):
        if not self.is_mouse_down:
            return

        x, y = ev.scenePos().x(), ev.scenePos().y()

        if self.end_point is not None:
            self.scene.removeItem(self.end_point)
        self.end_point = self.scene.addEllipse(
            x - self.half_point_size,
            y - self.half_point_size,
            self.point_size,
            self.point_size,
            pen=QPen(QColor("red")),
            brush=QBrush(QColor("red")),
        )

        if self.rect is not None:
            self.scene.removeItem(self.rect)
        sx, sy = self.start_pos
        xmin = min(x, sx)
        xmax = max(x, sx)
        ymin = min(y, sy)
        ymax = max(y, sy)
        self.rect = self.scene.addRect(
            xmin, ymin, xmax - xmin, ymax - ymin, pen=QPen(QColor("red"))
        )

    def mouse_release(self, ev):
        x, y = ev.scenePos().x(), ev.scenePos().y()
        sx, sy = self.start_pos
        xmin = min(x, sx)
        xmax = max(x, sx)
        ymin = min(y, sy)
        ymax = max(y, sy)

        self.is_mouse_down = False

        H, W, _ = self.img_3c.shape
        box_np = np.array([[xmin, ymin, xmax, ymax]])
        box_1024 = box_np / np.array([W, H, W, H]) * 1024

        img_1024 = transform.resize(
            self.img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )  # normalize to [0, 1], (H, W, 3)rr
        # convert the shape to (3, H, W)
        img_1024_tensor = (
            torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
        )

        if self.embedding is None:
            with torch.no_grad():
                self.embedding = medsam_model.image_encoder(
                    img_1024_tensor
                )  # (1, 256, 64, 64)

        medsam_mask = medsam_inference(medsam_model, self.embedding, box_1024, H, W)

        self.prev_mask = self.medsam_mask_c.copy()
        self.medsam_mask_c[medsam_mask != 0] = colors[self.color_idx % len(colors)]

        bg = Image.fromarray(self.img_3c.astype("uint8"), "RGB")
        mask = Image.fromarray(self.medsam_mask_c.astype("uint8"), "RGB")
        medsam_blend = Image.blend(bg, mask, 0.2)

        self.scene.removeItem(self.bg_img)
        self.bg_img = self.scene.addPixmap(np2pixmap(np.array(medsam_blend)))

        if self.compare_button.isChecked():
            self.sam_model.set_image(self.img_3c)
            sam_seg, _, _ = self.sam_model.predict(
                point_coords=None, point_labels=None, box=box_np, multimask_output=False
            )
            sam_seg = sam_seg[0]
            self.sam_mask_c[sam_seg != 0] = colors[self.color_idx % len(colors)]
            mask = Image.fromarray(self.sam_mask_c.astype("uint8"), "RGB")
            sam_blend = Image.blend(bg, mask, 0.2)

            plt.suptitle("Comparing MedSAM vs SAM results. Zoom by pressing the magnifying glass button then select a region. Quit by pressing Q")
            ax_medsam = plt.subplot(1, 2, 1)
            plt.title("MedSAM Result")
            plt.imshow(medsam_blend)
            plt.subplot(1, 2, 2, sharex=ax_medsam, sharey=ax_medsam)
            plt.title("SAM Result")
            plt.imshow(sam_blend)
            plt.get_current_fig_manager().full_screen_toggle()
            plt.show()

        self.color_idx += 1

    def save_mask(self):
        out_path = f"{self.image_path.split('.')[0]}_mask.png"
        io.imsave(out_path, self.medsam_mask_c)


app = QApplication(sys.argv)

w = Window()
w.show()

app.exec()
