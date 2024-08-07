import gc
import math
import os
import shutil
import ffmpeg
import zipfile
import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.utils.transforms import SAM2Transforms
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor
import cv2



def clean(Seg_Tracker):
    if Seg_Tracker is not None:
        predictor, inference_state, image_predictor = Seg_Tracker
        predictor.reset_state(inference_state)
        del predictor
        del inference_state
        del image_predictor
        del Seg_Tracker
        gc.collect()
        torch.cuda.empty_cache()
    return None, ({}, {}), None, None, 0, None, None, None, 0

def get_meta_from_video(Seg_Tracker, input_video, scale_slider, checkpoint):

    output_dir = 'output_frames'
    output_masks_dir = 'output_masks'
    output_combined_dir = 'output_combined'
    clear_folder(output_dir)
    clear_folder(output_masks_dir)
    clear_folder(output_combined_dir)
    if input_video is None:
        return None, ({}, {}), None, None, 0, None, None, None, 0
    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    output_frames = int(total_frames * scale_slider)
    frame_interval = max(1, total_frames // output_frames)
    ffmpeg.input(input_video, hwaccel='cuda').output(
        os.path.join(output_dir, '%07d.jpg'), q=2, start_number=0, 
        vf=rf'select=not(mod(n\,{frame_interval}))', vsync='vfr'
    ).run()

    first_frame_path = os.path.join(output_dir, '0000000.jpg')
    first_frame = cv2.imread(first_frame_path)
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    if Seg_Tracker is not None:
        del Seg_Tracker
        Seg_Tracker = None
        gc.collect()
        torch.cuda.empty_cache()
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if checkpoint == "tiny":
        sam2_checkpoint = "checkpoints/sam2_hiera_tiny.pt"
        model_cfg = "sam2_hiera_t.yaml"
    elif checkpoint == "samll":
        sam2_checkpoint = "checkpoints/sam2_hiera_small.pt"
        model_cfg = "sam2_hiera_s.yaml"
    elif checkpoint == "base-plus":
        sam2_checkpoint = "checkpoints/sam2_hiera_base_plus.pt"
        model_cfg = "sam2_hiera_b+.yaml"
    elif checkpoint == "large":
        sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
    
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device="cuda")
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

    image_predictor = SAM2ImagePredictor(sam2_model)
    inference_state = predictor.init_state(video_path=output_dir)
    predictor.reset_state(inference_state)
    return (predictor, inference_state, image_predictor), ({}, {}), first_frame_rgb, first_frame_rgb, 0, None, None, None, 0

def mask2bbox(mask):
    if len(np.where(mask > 0)[0]) == 0:
        print(f'not mask')
        return np.array([0, 0, 0, 0]).astype(np.int64), False
    x_ = np.sum(mask, axis=0)
    y_ = np.sum(mask, axis=1)
    x0 = np.min(np.nonzero(x_)[0])
    x1 = np.max(np.nonzero(x_)[0])
    y0 = np.min(np.nonzero(y_)[0])
    y1 = np.max(np.nonzero(y_)[0])
    return np.array([x0, y0, x1, y1]).astype(np.int64), True

def sam_stroke(Seg_Tracker, drawing_board, last_draw, frame_num, ann_obj_id):
    predictor, inference_state, image_predictor = Seg_Tracker
    image_path = f'output_frames/{frame_num:07d}.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    display_image = drawing_board["image"]
    image_predictor.set_image(image)
    input_mask = drawing_board["mask"]
    input_mask[input_mask != 0] = 255
    if last_draw is not None:
        diff_mask = cv2.absdiff(input_mask, last_draw)
        input_mask = diff_mask
    bbox, hasMask = mask2bbox(input_mask[:, :, 0]) 
    if not hasMask :
        return Seg_Tracker, display_image, display_image
    masks, scores, logits = image_predictor.predict( point_coords=None, point_labels=None, box=bbox[None, :], multimask_output=False,)
    mask = masks > 0.0
    masked_frame = show_mask(mask, display_image, ann_obj_id)
    masked_with_rect = draw_rect(masked_frame, bbox, ann_obj_id)
    frame_idx, object_ids, masks = predictor.add_new_mask(inference_state, frame_idx=frame_num, obj_id=ann_obj_id, mask=mask[0])
    last_draw = drawing_board["mask"]
    return Seg_Tracker, masked_with_rect, masked_with_rect, last_draw

def draw_rect(image, bbox, obj_id):
    cmap = plt.get_cmap("tab10")
    color = np.array(cmap(obj_id)[:3])
    rgb_color = tuple(map(int, (color[:3] * 255).astype(np.uint8)))
    inv_color = tuple(map(int, (255 - color[:3] * 255).astype(np.uint8)))
    x0, y0, x1, y1 = bbox
    image_with_rect = cv2.rectangle(image.copy(), (x0, y0), (x1, y1), inv_color, thickness=2)
    return image_with_rect

def sam_click(Seg_Tracker, frame_num, point_mode, click_stack, ann_obj_id, evt: gr.SelectData):
    points_dict, labels_dict = click_stack
    predictor, inference_state, image_predictor = Seg_Tracker
    ann_frame_idx = frame_num  # the frame index we interact with
    print(f'ann_frame_idx: {ann_frame_idx}')
    point = np.array([[evt.index[0], evt.index[1]]], dtype=np.float32)
    if point_mode == "Positive":
        label = np.array([1], np.int32)
    else:
        label = np.array([0], np.int32)

    if ann_frame_idx not in points_dict:
        points_dict[ann_frame_idx] = {}
    if ann_frame_idx not in labels_dict:
        labels_dict[ann_frame_idx] = {}

    if ann_obj_id not in points_dict[ann_frame_idx]:
        points_dict[ann_frame_idx][ann_obj_id] = np.empty((0, 2), dtype=np.float32)
    if ann_obj_id not in labels_dict[ann_frame_idx]:
        labels_dict[ann_frame_idx][ann_obj_id] = np.empty((0,), dtype=np.int32)

    points_dict[ann_frame_idx][ann_obj_id] = np.append(points_dict[ann_frame_idx][ann_obj_id], point, axis=0)
    labels_dict[ann_frame_idx][ann_obj_id] = np.append(labels_dict[ann_frame_idx][ann_obj_id], label, axis=0)

    click_stack = (points_dict, labels_dict)

    frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points_dict[ann_frame_idx][ann_obj_id],
        labels=labels_dict[ann_frame_idx][ann_obj_id],
    )

    image_path = f'output_frames/{ann_frame_idx:07d}.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masked_frame = image.copy()
    for i, obj_id in enumerate(out_obj_ids):
        mask = (out_mask_logits[i] > 0.0).cpu().numpy()
        masked_frame = show_mask(mask, image=masked_frame, obj_id=obj_id)
    masked_frame_with_markers = draw_markers(masked_frame, points_dict[ann_frame_idx], labels_dict[ann_frame_idx])

    return Seg_Tracker, masked_frame_with_markers, masked_frame_with_markers, click_stack

def draw_markers(image, points_dict, labels_dict):
    cmap = plt.get_cmap("tab10")
    image_h, image_w = image.shape[:2]
    marker_size = max(1, int(min(image_h, image_w) * 0.05))

    for obj_id in points_dict:
        color = np.array(cmap(obj_id)[:3])
        rgb_color = tuple(map(int, (color[:3] * 255).astype(np.uint8)))
        inv_color = tuple(map(int, (255 - color[:3] * 255).astype(np.uint8)))
        for point, label in zip(points_dict[obj_id], labels_dict[obj_id]):
            x, y = int(point[0]), int(point[1])
            if label == 1:
                cv2.drawMarker(image, (x, y), inv_color, markerType=cv2.MARKER_CROSS, markerSize=marker_size, thickness=2)
            else:
                cv2.drawMarker(image, (x, y), inv_color, markerType=cv2.MARKER_TILTED_CROSS, markerSize=int(marker_size / np.sqrt(2)), thickness=2)
    
    return image

def show_mask(mask, image=None, obj_id=None):
    cmap = plt.get_cmap("tab10")
    cmap_idx = 0 if obj_id is None else obj_id
    color = np.array([*cmap(cmap_idx)[:3], 0.6])
    
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask_image = (mask_image * 255).astype(np.uint8)
    if image is not None:
        image_h, image_w = image.shape[:2]
        if (image_h, image_w) != (h, w):
            raise ValueError(f"Image dimensions ({image_h}, {image_w}) and mask dimensions ({h}, {w}) do not match")
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        for c in range(3):
            colored_mask[..., c] = mask_image[..., c]
        alpha_mask = mask_image[..., 3] / 255.0
        for c in range(3):
            image[..., c] = np.where(alpha_mask > 0, (1 - alpha_mask) * image[..., c] + alpha_mask * colored_mask[..., c], image[..., c])
        return image
    return mask_image

def show_res_by_slider(frame_per, click_stack):
    image_path = 'output_frames'
    output_combined_dir = 'output_combined'
    
    combined_frames = sorted([os.path.join(output_combined_dir, img_name) for img_name in os.listdir(output_combined_dir)])
    if combined_frames:
        output_masked_frame_path = combined_frames
    else:
        original_frames = sorted([os.path.join(image_path, img_name) for img_name in os.listdir(image_path)])
        output_masked_frame_path = original_frames
       
    total_frames_num = len(output_masked_frame_path)
    if total_frames_num == 0:
        print("No output results found")
        return None, None
    else:
        frame_num = math.floor(total_frames_num * frame_per / 100)
        if frame_per == 100:
            frame_num = frame_num - 1
        chosen_frame_path = output_masked_frame_path[frame_num]
        print(f"{chosen_frame_path}")
        chosen_frame_show = cv2.imread(chosen_frame_path)
        chosen_frame_show = cv2.cvtColor(chosen_frame_show, cv2.COLOR_BGR2RGB)
        points_dict, labels_dict = click_stack
        if frame_num in points_dict and frame_num in labels_dict:
            chosen_frame_show = draw_markers(chosen_frame_show, points_dict[frame_num], labels_dict[frame_num])
        return chosen_frame_show, chosen_frame_show, frame_num

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def zip_folder(folder_path, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_STORED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))

def tracking_objects(Seg_Tracker, frame_num, input_video):
    output_dir = 'output_frames'
    output_masks_dir = 'output_masks'
    output_combined_dir = 'output_combined'
    output_video_path = 'output_video.mp4'
    output_zip_path = 'output_masks.zip'
    clear_folder(output_masks_dir)
    clear_folder(output_combined_dir)
    if os.path.exists(output_video_path):
        os.remove(output_video_path)
    if os.path.exists(output_zip_path):
        os.remove(output_zip_path)
    video_segments = {}
    predictor, inference_state, image_predictor = Seg_Tracker
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
    # for frame_idx in sorted(video_segments.keys()):
    for frame_file in frame_files:
        frame_idx = int(os.path.splitext(frame_file)[0])
        frame_path = os.path.join(output_dir, frame_file)
        image = cv2.imread(frame_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masked_frame = image.copy()
        if frame_idx in video_segments:
            for obj_id, mask in video_segments[frame_idx].items():
                masked_frame = show_mask(mask, image=masked_frame, obj_id=obj_id)
                mask_output_path = os.path.join(output_masks_dir, f'{obj_id}_{frame_idx:07d}.png')
                cv2.imwrite(mask_output_path, show_mask(mask))
        combined_output_path = os.path.join(output_combined_dir, f'{frame_idx:07d}.png')
        combined_image_bgr = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(combined_output_path, combined_image_bgr)
        if frame_idx == frame_num:
            final_masked_frame = masked_frame

    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    # output_frames = int(total_frames * scale_slider)
    output_frames = len([name for name in os.listdir(output_combined_dir) if os.path.isfile(os.path.join(output_combined_dir, name)) and name.endswith('.png')])
    out_fps = fps * output_frames / total_frames
    ffmpeg.input(os.path.join(output_combined_dir, '%07d.png'), framerate=out_fps).output(output_video_path, vcodec='h264_nvenc', pix_fmt='yuv420p').run()
    zip_folder(output_masks_dir, output_zip_path)
    print("done")
    return final_masked_frame, final_masked_frame, output_video_path, output_video_path, output_zip_path

def increment_ann_obj_id(ann_obj_id):
    ann_obj_id += 1
    return ann_obj_id

def drawing_board_get_input_first_frame(input_first_frame):
    return input_first_frame

def seg_track_app():

    ##########################################################
    ######################  Front-end ########################
    ##########################################################
    css = """
    #input_output_video video {
        max-height: 550px;
        max-width: 100%;
        height: auto;
    }
    """

    app = gr.Blocks(css=css)

    with app:
        gr.Markdown(
            '''
            <div style="text-align:center;">
                <span style="font-size:3em; font-weight:bold;">SAM2 for Video Segmentation 🔥</span>
            </div>
            This api supports using box (generated by scribble) and point prompts for video segmentation with SAM2.

            1. Upload video file 
            2. Select mdoel size and downsample frame rate and run `Preprocess`
            3. Use `Stroke to Box Prompt` to draw box on the first frame or `Point Prompt` to click on the first frame
            4. Click `Segment` to get the segmentation result
            5. Click `Add New Object` to add new object
            6. Click `Start Tracking` to track objects in the video
            7. Click `Reset` to reset the app
            8. Download the video with segmentation result
            '''
        )

        click_stack = gr.State(({}, {}))
        Seg_Tracker = gr.State(None)
        frame_num = gr.State(value=(int(0)))
        ann_obj_id = gr.State(value=(int(0)))
        last_draw = gr.State(None)

        with gr.Row():
            with gr.Column(scale=0.5):
                with gr.Row():
                    tab_video_input = gr.Tab(label="Video input")
                    with tab_video_input:
                        input_video = gr.Video(label='Input video', elem_id="input_output_video")
                        with gr.Row():
                            checkpoint = gr.Dropdown(label="Model Size", choices=["tiny", "small", "base-plus", "large"], value="tiny")
                            scale_slider = gr.Slider(
                                label="Downsampe Frame Rate",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.25,
                                value=1.0,
                                interactive=True
                            )
                            preprocess_button = gr.Button(
                                value="Preprocess",
                                interactive=True,
                            )

                with gr.Row():
                    tab_stroke = gr.Tab(label="Stroke to Box Prompt")
                    with tab_stroke:
                        drawing_board = gr.Image(label='Drawing Board', tool="sketch", brush_radius=10, interactive=True)
                        with gr.Row():
                            seg_acc_stroke = gr.Button(value="Segment", interactive=True)
                            
                    tab_click = gr.Tab(label="Point Prompt")
                    with tab_click:
                        input_first_frame = gr.Image(label='Segment result of first frame',interactive=True).style(height=550)
                        with gr.Row():
                            point_mode = gr.Radio(
                                        choices=["Positive",  "Negative"],
                                        value="Positive",
                                        label="Point Prompt",
                                        interactive=True)
                            
                with gr.Row():
                    with gr.Column():
                        frame_per = gr.Slider(
                            label = "Percentage of Frames Viewed",
                            minimum= 0.0,
                            maximum= 100.0,
                            step=0.01,
                            value=0.0,
                        )
                        new_object_button = gr.Button(
                            value="Add New Object", 
                            interactive=True
                        )
                        track_for_video = gr.Button(
                            value="Start Tracking",
                                interactive=True,
                                )
                        reset_button = gr.Button(
                            value="Reset",
                            interactive=True,
                        )

            with gr.Column(scale=0.5):
                output_video = gr.Video(label='Visualize Results', elem_id="input_output_video")
                output_mp4 = gr.File(label="Predicted video")
                output_mask = gr.File(label="Predicted masks")


        gr.Markdown(
            '''
            <div style="text-align:center; margin-top: 20px;">
                The authors of this work highly appreciate Meta AI for making SAM2 publicly available to the community. 
                The interface was built on <a href="https://github.com/z-x-yang/Segment-and-Track-Anything/blob/main/tutorial/tutorial%20for%20WebUI-1.0-Version.md" target="_blank">SegTracker</a>. 
                <a href="https://docs.google.com/document/d/1idDBV0faOjdjVs-iAHr0uSrw_9_ZzLGrUI2FEdK-lso/edit?usp=sharing" target="_blank">Data Source</a>.
            </div>
                '''
        )

    ##########################################################
    ######################  back-end #########################
    ##########################################################

        # listen to the preprocess button click to get the first frame of video with scaling
        preprocess_button.click(
            fn=get_meta_from_video,
            inputs=[
                Seg_Tracker,
                input_video,
                scale_slider,
                checkpoint
            ],
            outputs=[
                Seg_Tracker, click_stack, input_first_frame, drawing_board, frame_per, output_video, output_mp4, output_mask, ann_obj_id
            ]
        )

        frame_per.release(
            fn=show_res_by_slider, 
            inputs=[
                frame_per, click_stack
                ], 
            outputs=[
                input_first_frame, drawing_board, frame_num
            ]
        )

        # Interactively modify the mask acc click
        input_first_frame.select(
            fn=sam_click,
            inputs=[
                Seg_Tracker, frame_num, point_mode, click_stack, ann_obj_id
            ],
            outputs=[
                Seg_Tracker, input_first_frame, drawing_board, click_stack
            ]
        )

        # Track object in video
        track_for_video.click(
            fn=tracking_objects,
            inputs=[
                Seg_Tracker,
                frame_num,
                input_video,
            ],
            outputs=[
                input_first_frame,
                drawing_board,
                output_video,
                output_mp4,
                output_mask
            ]
        )

        reset_button.click(
            fn=clean,
            inputs=[
                Seg_Tracker
            ],
            outputs=[
                Seg_Tracker, click_stack, input_first_frame, drawing_board, frame_per, output_video, output_mp4, output_mask, ann_obj_id
            ]
        )

        new_object_button.click(
            fn=increment_ann_obj_id, 
            inputs=[
                ann_obj_id
                ], 
            outputs=[
                ann_obj_id
                ]
        )

        tab_stroke.select(
            fn=drawing_board_get_input_first_frame,
            inputs=[input_first_frame,],
            outputs=[drawing_board,],
        )

        seg_acc_stroke.click(
            fn=sam_stroke,
            inputs=[
                Seg_Tracker, drawing_board, last_draw, frame_num, ann_obj_id
            ],
            outputs=[
                Seg_Tracker, input_first_frame, drawing_board, last_draw
            ]
        )
        
    app.queue(concurrency_count=1)
    app.launch(debug=True, enable_queue=True, share=True)

if __name__ == "__main__":
    seg_track_app()