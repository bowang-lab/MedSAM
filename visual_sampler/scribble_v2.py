# --------------------------------------------------------
# Adapted from 
# SEEM -- Segment Everything Everywhere All At Once
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import random
import torch
from .mask_generators import get_mask_by_input_strokes
import copy
import cv2
import numpy as np
import largestinteriorrectangle as lir
import cc3d

def set_edges_to_false(mask, size):
    # Set the edges to False
    mask[0:size, :] = False  # Set top rows to False
    mask[-size:, :] = False  # Set bottom rows to False
    mask[:, 0:size] = False  # Set left columns to False
    mask[:, -size:] = False  # Set right columns to False
    return mask

class Scribble:
    def __init__(self, cfg, is_train):
        self.num_stroke = cfg['STROKE_SAMPLER']['SCRIBBLE']['NUM_STROKES']
        self.stroke_preset = cfg['STROKE_SAMPLER']['SCRIBBLE']['STROKE_PRESET']
        self.stroke_prob = cfg['STROKE_SAMPLER']['SCRIBBLE']['STROKE_PROB']
        self.eval_stroke = cfg['STROKE_SAMPLER']['EVAL']['MAX_ITER']
        self.is_train = is_train

    @staticmethod
    def get_stroke_preset(stroke_preset):
        if stroke_preset == 'rand_curve':
            return {
                "nVertexBound": [10, 30],
                "maxHeadSpeed": 20,
                "maxHeadAcceleration": (15, 0.5),
                "brushWidthBound": (3, 10),
                "nMovePointRatio": 0.5,
                "maxPiontMove": 3,
                "maxLineAcceleration": (5, 0.5),
                "boarderGap": None,
                "maxInitSpeed": 6
            }
        elif stroke_preset == 'rand_curve_small':
            return {
                "nVertexBound": [6, 22],
                "maxHeadSpeed": 12,
                "maxHeadAcceleration": (8, 0.5),
                "brushWidthBound": (2.5, 5),
                "nMovePointRatio": 0.5,
                "maxPiontMove": 1.5,
                "maxLineAcceleration": (3, 0.5),
                "boarderGap": None,
                "maxInitSpeed": 3
            }
        elif stroke_preset == 'rand_curve_tiny':
            return {
                "nVertexBound": [3, 11],
                "maxHeadSpeed": 6,
                "maxHeadAcceleration": (4, 0.5),
                "brushWidthBound": (2, 4),
                "nMovePointRatio": 0.5,
                "maxPiontMove": 1.0,
                "maxLineAcceleration": (2, 0.5),
                "boarderGap": None,
                "maxInitSpeed": 2
            }
        else:
            raise NotImplementedError(f'The stroke presetting "{stroke_preset}" does not exist.')

    def get_random_points_from_mask(self, mask, n=5):
        h,w = mask.shape
        view_mask = mask.reshape(h*w)
        non_zero_idx = view_mask.nonzero()[:,0]
        selected_idx = torch.randperm(len(non_zero_idx))[:n]
        non_zero_idx = non_zero_idx[selected_idx]
        y = (non_zero_idx // w)*1.0
        x = (non_zero_idx % w)*1.0
        return torch.cat((x[:,None], y[:,None]), dim=1).numpy()

    def draw(self, mask=None, box=None):
        if not self.is_train:
            return self.draw_eval(mask=mask, box=box)
        stroke_preset_name = random.choices(self.stroke_preset, weights=self.stroke_prob, k=1)[0]
        preset = Scribble.get_stroke_preset(stroke_preset_name)
        nStroke = random.randint(1, min(self.num_stroke, mask.sum().item()))
        contours, _ = cv2.findContours(np.uint8(mask.numpy()) * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour = np.array([contours[0][:, 0, :]])
        min_x, min_y, width, height = lir.lir(contour)
        max_x = min_x + width
        max_y = min_y + height
        mask2 = copy.deepcopy(mask)
        submask = mask2[min_y:max_y+1, min_x:max_x+1]
        
        
        h,w = submask.shape
        points = self.get_random_points_from_mask(submask, n=nStroke) #shape 1,2
        rand_mask = get_mask_by_input_strokes(
            init_points=points,
            imageWidth=w, imageHeight=h, nStroke=min(nStroke, len(points)), **preset)
        new_mask = torch.zeros_like(mask2).bool()
        new_mask[min_y:(max_y+1), min_x:(max_x+1)] = ~torch.tensor(rand_mask)
        
        result = new_mask * mask
        return result

    def draw(self, mask=None, box=None):
        if not self.is_train:
            return self.draw_eval(mask=mask, box=box)
        stroke_preset_name = random.choices(self.stroke_preset, weights=self.stroke_prob, k=1)[0]
        preset = Scribble.get_stroke_preset(stroke_preset_name)
        nStroke = random.randint(1, min(self.num_stroke, mask.sum().item()))
        while True:
            contours, tmp = cv2.findContours(np.uint8(mask.numpy()) * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for i in range(tmp.shape[1]):
                try:
                    if tmp[0,i,-1]==-1:
                        contour = np.array([contours[i][:, 0, :]])
                        min_x, min_y, width, height = lir.lir(contour)
                        break
                except:
                    #print(i)
                    pass
            try:
                max_x = min_x + width
                max_y = min_y + height
            except:
                continue
            mask2 = copy.deepcopy(mask)
            submask = mask2[min_y:max_y+1, min_x:max_x+1]
            h,w = submask.shape
            points = self.get_random_points_from_mask(submask, n=nStroke) #shape 1,2
            rand_mask = get_mask_by_input_strokes(
                init_points=points,
                imageWidth=w, imageHeight=h, nStroke=min(nStroke, len(points)), **preset)
            new_mask = torch.zeros_like(mask2).bool()
            new_mask[min_y:(max_y+1), min_x:(max_x+1)] = ~torch.tensor(rand_mask)
            if new_mask.sum() > 0:
                break
        return new_mask


    def draw_background(self, mask=None, box=None):
        if not self.is_train:
            return self.draw_eval(mask=mask, box=box)
        stroke_preset_name = random.choices(self.stroke_preset, weights=self.stroke_prob, k=1)[0]
        preset = Scribble.get_stroke_preset(stroke_preset_name)
        nStroke = random.randint(1, min(self.num_stroke, mask.sum().item()))
        while True:
            mask0 = copy.deepcopy(mask)
            mask0 = set_edges_to_false(mask0, int(mask0.shape[0]*random.uniform(0.05, 0.25)))
            h,w = mask0.shape
            points = self.get_random_points_from_mask(mask0, n=nStroke)
            rand_mask = get_mask_by_input_strokes(
                init_points=points,
                imageWidth=w, imageHeight=h, nStroke=min(nStroke, len(points)), **preset)
            rand_mask = (~torch.from_numpy(rand_mask)) * mask0
            if rand_mask.sum() == 0:
                continue
            labels_out = cc3d.connected_components(rand_mask.numpy().astype(np.uint8))
            largest_component = np.argmax(np.bincount(labels_out.flat)[1:]) + 1
            rand_mask = labels_out == largest_component
            if rand_mask.sum()>0:
                break
        return torch.tensor(rand_mask)

    def draw_eval(self, mask=None, box=None):
        stroke_preset_name = random.choices(self.stroke_preset, weights=self.stroke_prob, k=1)[0]
        preset = Scribble.get_stroke_preset(stroke_preset_name)
        nStroke = min(self.eval_stroke, mask.sum().item())
        h,w = mask.shape
        points = self.get_random_points_from_mask(mask, n=nStroke)
        rand_masks = []
        for i in range(len(points)):
            rand_mask = get_mask_by_input_strokes(
                init_points=points[:i+1],
                imageWidth=w, imageHeight=h, nStroke=min(i, len(points)), **preset)
            rand_mask = (~torch.from_numpy(rand_mask)) * mask
            rand_masks += [rand_mask]
        return torch.stack(rand_masks)

    @staticmethod
    def draw_by_points(points, mask, h, w):
        stroke_preset_name = random.choices(['rand_curve', 'rand_curve_small'], weights=[0.5, 0.5], k=1)[0]
        preset = Scribble.get_stroke_preset(stroke_preset_name)
        rand_mask = get_mask_by_input_strokes(
            init_points=points,
            imageWidth=w, imageHeight=h, nStroke=len(points), **preset)[None,]
        rand_masks = (~torch.from_numpy(rand_mask)) * mask
        return rand_masks

    def __repr__(self,):
        return 'scribble'