import SimpleITK as sitk
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import remove_small_objects, binary_closing, disk
from scipy.ndimage import binary_fill_holes
import matplotlib

def get_image(pth, norm_type="window", window_min_percentile=1, window_max_percentile=99):
    file_sitk = sitk.ReadImage(pth)
    image_data = sitk.GetArrayFromImage(file_sitk) #channel (or slice in 3D cases) first, then x, y. TODO: need to check when the data is more than 3D, e.g. 4D or 5D.

    assert len(image_data.shape) in [2, 3], "Image data must be 2D or 3D, as the function get_image isn't implemented for more. Current shape: {}".format(image_data.shape)

    # Check if the last 2 dimensions are not identical and pad to make it a perfect square
    height, width = image_data.shape[-2], image_data.shape[-1]
    if height != width:
        # Calculate padding needed to make it square
        max_dim = max(height, width)
        pad_height = max_dim - height
        pad_width = max_dim - width
        
        # Apply padding to the last two dimensions
        if len(image_data.shape) == 2:
            # 2D image
            image_data = np.pad(image_data, 
                                ((pad_height//2, pad_height - pad_height//2), 
                                (pad_width//2, pad_width - pad_width//2)), 
                                mode='constant', constant_values=0)
        elif len(image_data.shape) == 3:
            # 3D image (multiple slices)
            image_data = np.pad(image_data, 
                                ((0, 0), 
                                (pad_height//2, pad_height - pad_height//2), 
                                (pad_width//2, pad_width - pad_width//2)), 
                                mode='constant', constant_values=0)

    # normalise to [0, 255]    
    if norm_type == "minmax": # Method 0: Simple normalisation with Min/Max scaling
        image_data_pre = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 255.0
        return np.uint8(image_data_pre)
    elif norm_type == "window":  # Method 1: Window/Level adjustment (like CT/MRI viewing)
        lower_bound = np.percentile(image_data[image_data > 0], window_min_percentile) if np.any(image_data > 0) else 0
        upper_bound = np.percentile(image_data, window_max_percentile)
        image_data_windowed = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (image_data_windowed - lower_bound) / (upper_bound - lower_bound) * 255.0
        return np.uint8(image_data_pre)
    else:
        raise ValueError("Unsupported normalisation type: {}".format(norm_type))

def segment_image(image_data, model_demo, prompt, use_otsu=False, keep_largest_only=False, fill_holes=False):
    segs = []
    model_demo.set_image(image_data)
    for i in range(image_data.shape[0]):
        seg, seg_raw = model_demo.infer(prompt, slice_idx=i, return_raw=True)
        
        if use_otsu:
            thresh = threshold_otsu(seg_raw)
            seg = (seg_raw > thresh).astype(np.uint8)
        
        if keep_largest_only or fill_holes:
            seg_bool = seg.astype(bool)
            
            # Keep only the largest connected component if requested
            if keep_largest_only and np.any(seg_bool):
                labelled = label(seg_bool)
                if labelled.max() > 0:
                    # Find the largest connected component
                    component_sizes = np.bincount(labelled.ravel())
                    component_sizes[0] = 0  # Ignore background
                    largest_component = np.argmax(component_sizes)
                    seg_bool = (labelled == largest_component)
            
            # Fill holes if requested
            if fill_holes and np.any(seg_bool):
                seg_bool = binary_fill_holes(seg_bool)
            
            seg = seg_bool.astype(np.uint8)

        segs.append(seg)
    return np.array(segs)

def _pepare_prompt_from_text_seg(text_model_demo, text_prompt, text_model_slice, use_otsu_text=False, second_model_mode=None):
    text_seg, seg_raw = text_model_demo.infer(text_prompt, slice_idx=text_model_slice, return_raw=True)
    if use_otsu_text:
        thresh = threshold_otsu(seg_raw)
        text_seg = (seg_raw > thresh).astype(np.uint8)
    text_seg_bool = text_seg.astype(bool)
    # Keep only the largest connected component
    if np.any(text_seg_bool):
        labelled = label(text_seg_bool)
        if labelled.max() > 0:
            # Find the largest connected component
            component_sizes = np.bincount(labelled.ravel())
            component_sizes[0] = 0  # Ignore background
            largest_component = np.argmax(component_sizes)
            text_seg_bool = (labelled == largest_component)
    # Fill holes
    if second_model_mode == "point":
        text_seg_bool = binary_fill_holes(text_seg_bool)
    text_seg = text_seg_bool.astype(np.uint8)
    
    if np.any(text_seg):
        y_coords, x_coords = np.where(text_seg > 0)
        if second_model_mode == "point":            
            centre_y = int(np.mean(y_coords))
            centre_x = int(np.mean(x_coords))
            prompt = {"x": centre_x, "y": centre_y}
        elif second_model_mode == "bbox":
            x_min = x_coords.min()
            x_max = x_coords.max()
            y_min = y_coords.min()
            y_max = y_coords.max()
            prompt = {"bbox": np.array([x_min, y_min, x_max, y_max])}
    else:
        prompt = None

    return text_seg, prompt
    
def segment_image_pretext(image_data, text_model_demo, text_prompt, second_model_demo, second_model_mode, text_model_slice=-1, use_otsu_text=False, use_otsu=False, keep_largest_only=False, fill_holes=False):
    text_model_demo.set_image(image_data)
    second_model_demo.set_image(image_data)

    if text_model_slice != -1:
        _, prompt = _pepare_prompt_from_text_seg(text_model_demo, text_prompt, text_model_slice, use_otsu_text=use_otsu_text, second_model_mode=second_model_mode)

    segs = []    
    for i in range(image_data.shape[0]):
        if text_model_slice == -1:
            _, prompt = _pepare_prompt_from_text_seg(text_model_demo, text_prompt, i, use_otsu_text=use_otsu_text, second_model_mode=second_model_mode)

        if prompt is not None:
            seg, seg_raw = second_model_demo.infer(**prompt, slice_idx=i, return_raw=True)
        
            if use_otsu:
                thresh = threshold_otsu(seg_raw)
                seg = (seg_raw > thresh).astype(np.uint8)
        
            if keep_largest_only or fill_holes:
                seg_bool = seg.astype(bool)
                
                # Keep only the largest connected component if requested
                if keep_largest_only and np.any(seg_bool):
                    labelled = label(seg_bool)
                    if labelled.max() > 0:
                        # Find the largest connected component
                        component_sizes = np.bincount(labelled.ravel())
                        component_sizes[0] = 0  # Ignore background
                        largest_component = np.argmax(component_sizes)
                        seg_bool = (labelled == largest_component)
                
                # Fill holes if requested
                if fill_holes and np.any(seg_bool):
                    seg_bool = binary_fill_holes(seg_bool)
                
                seg = seg_bool.astype(np.uint8)
        else:
            seg = np.zeros_like(image_data[i], dtype=np.uint8)

        segs.append(seg)
    return np.array(segs)

def create_overlay(img_data, segs, alpha=0.5):
    overlay_volume_uint8 = np.zeros(img_data.shape + (3,), dtype=np.uint8)  # (slices, H, W, 3)
    cmap = matplotlib.colormaps["tab10"]

    for i in range(img_data.shape[0]):
        img_slice = img_data[i]
        mask_slice = segs[i]
        img_disp = (img_slice - img_slice.min()) / (np.ptp(img_slice) + 1e-8)
        colour_mask = cmap(mask_slice % 10)[..., :3]
        overlay_float = (1 - alpha) * np.stack([img_disp]*3, axis=-1) + alpha * colour_mask * (mask_slice[..., None] > 0)
        overlay_volume_uint8[i] = (overlay_float * 255).astype(np.uint8)

    return overlay_volume_uint8
    

def save_image(image_data, pth, is_RGB=False):
    """
    Save the image data to a file.
    """
    assert (is_RGB and len(image_data.shape) in [3, 4]) or ((not is_RGB) and (len(image_data.shape) in [2, 3])), "Image data must be 2D or 3D, as the function save_image isn't implemented for more. Current shape: {}".format(image_data.shape)

    sitk_image = sitk.GetImageFromArray(image_data, isVector=is_RGB)
    sitk.WriteImage(sitk_image, pth)