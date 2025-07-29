# Enhanced TextPromptDemo with 3D Volume Support

This enhanced version of `TextPromptDemo` supports both 2D slices and 3D volumes with an interactive slider interface for navigation through slices.

## Key Features

### 1. **Dual Mode Support**
- **2D Mode**: Works with single slices as before
- **3D Mode**: Handles full volumes with slice-by-slice segmentation

### 2. **Interactive Volume Navigation**
- Slice slider for navigating through volume
- Real-time display updates
- Slice counter display

### 3. **Batch Processing**
- "Segment All Slices" button for processing entire volumes
- Caches segmentations for efficient navigation
- Progress indication during batch processing

### 4. **Enhanced Visualization**
- Maintains all existing visualization features
- Overlay segmentations on original images
- Consistent mask coloring across slices

## Usage Examples

### Basic 2D Usage (unchanged)
```python
# Load your trained model
model = load_your_medsam_model()

# Create demo instance
demo = TextPromptDemo(model)

# Set 2D image
demo.set_image(image_2d)  # Shape: (H, W)

# Show interactive interface
demo.show()
```

### New 3D Volume Usage
```python
# Load your trained model
model = load_your_medsam_model()

# Create demo instance
demo = TextPromptDemo(model)

# Set 3D volume
demo.set_image(volume_3d)  # Shape: (D, H, W)

# Show interactive interface with slider
demo.show()
```

### Programmatic Segmentation
```python
# Set volume
demo.set_image(volume_3d)

# Segment specific slice
segmentation = demo.infer("liver", slice_idx=10)

# Segment all slices for a prompt
for i in range(volume_3d.shape[0]):
    seg = demo.infer("liver", slice_idx=i)
    # Process segmentation...

# Get complete segmentation volume
seg_volume = demo.get_segmentation_volume("liver")

# Save segmentation volume
demo.save_segmentation_volume("liver", "liver_segmentation.npy")
```

## New Methods

### `set_image(image)`
Enhanced to handle both 2D and 3D inputs:
- **2D input**: `(H, W)` - behaves as before
- **3D input**: `(D, H, W)` - processes all slices and computes embeddings

### `infer(text, slice_idx=None)`
Enhanced inference method:
- `text`: Text prompt for segmentation
- `slice_idx`: Specific slice index for volumes (optional)
- Returns segmentation mask for the specified or current slice

### `get_segmentation_volume(prompt)`
Returns the complete 3D segmentation volume for a given prompt.
- **Returns**: `numpy.array` of shape `(D, H, W)` or `None`

### `save_segmentation_volume(prompt, filepath)`
Saves the 3D segmentation volume to a file.
- Supports `.npy` and `.npz` formats
- Includes prompt metadata for `.npz` files

### `get_segmentation_statistics(prompt)`
Returns statistics about segmentation coverage:
```python
{
    'total_slices': int,
    'segmented_slices': int, 
    'coverage_percentage': float,
    'total_voxels_analyzed': int,
    'segmented_voxels': int,
    'segmentation_percentage': float
}
```

## Interactive Interface Features

### 3D Volume Interface
- **Text Input**: Enter segmentation prompts
- **Slice Slider**: Navigate through volume slices
- **Segment All Button**: Process entire volume with current prompt
- **Info Display**: Shows volume dimensions and current slice

### Navigation
- Use the slider to move between slices
- Segmentations are cached and displayed when revisiting slices
- Error messages show for invalid prompts

### Batch Processing
- Click "Segment All Slices" to process the entire volume
- Progress is indicated by button state changes
- All segmentations are cached for future navigation

## Implementation Details

### Memory Management
- Image embeddings are computed once per slice and cached
- Segmentations are stored in memory for fast access
- For very large volumes, consider processing in chunks

### Performance Considerations
- Initial setup time increases with volume size (embedding computation)
- Interactive navigation is fast due to cached embeddings
- Batch processing time scales linearly with volume depth

### Data Structures
```python
self.image_embeddings: List[torch.Tensor]  # For 3D volumes
self.segmentations: Dict[str, Dict[int, np.array]]  # {prompt: {slice_idx: mask}}
self.is_volume: bool  # True for 3D, False for 2D
self.current_slice: int  # Current slice index for 3D
```

## Dependencies
The enhanced version requires additional widget imports:
```python
from ipywidgets import IntSlider, VBox, HBox
from IPython.display import display, clear_output
```

## Backward Compatibility
All existing 2D functionality remains unchanged. The class automatically detects input dimensionality and switches between 2D and 3D modes.

## Error Handling
- Validates input dimensions
- Handles missing segmentations gracefully  
- Provides informative error messages for invalid prompts
- Manages edge cases in volume navigation

## Future Enhancements
Potential improvements could include:
- 3D visualization using plotly or similar
- Export options for various medical image formats
- Multi-prompt comparison views
- Advanced volume statistics and analysis tools
