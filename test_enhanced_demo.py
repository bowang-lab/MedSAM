#!/usr/bin/env python3
"""
Test script for the enhanced BboxPromptDemo class with 2D/3D support
"""

import numpy as np
import sys
import os

# Add the utils directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from demo import BboxPromptDemo

def create_test_data():
    """Create synthetic test data for demonstration"""
    
    # Create a 2D test image (256x256)
    test_2d = np.random.rand(256, 256) * 255
    # Add some structure (circle in the middle)
    y, x = np.ogrid[:256, :256]
    center_y, center_x = 128, 128
    mask = (x - center_x)**2 + (y - center_y)**2 <= 50**2
    test_2d[mask] = 200
    test_2d = test_2d.astype(np.uint8)
    
    # Create a 3D test volume (10 slices, 256x256)
    test_3d = np.random.rand(10, 256, 256) * 255
    # Add some structure across slices
    for i in range(10):
        y, x = np.ogrid[:256, :256]
        center_y, center_x = 128, 128
        radius = 30 + i * 2  # Growing circle across slices
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        test_3d[i][mask] = 200
    test_3d = test_3d.astype(np.uint8)
    
    return test_2d, test_3d

def test_2d_demo():
    """Test the demo with 2D data"""
    print("Testing 2D Demo...")
    
    # Create mock model (you would load your actual MedSAM model here)
    class MockModel:
        def __init__(self):
            self.device = 'cpu'
        
        def eval(self):
            pass
        
        class ImageEncoder:
            def __call__(self, x):
                # Mock embedding - same shape as expected by MedSAM
                return np.random.rand(1, 256, 64, 64)
        
        class PromptEncoder:
            def __call__(self, points=None, boxes=None, masks=None, tokens=None):
                sparse = np.random.rand(1, 2, 256)
                dense = np.random.rand(1, 256, 64, 64)
                return sparse, dense
            
            def get_dense_pe(self):
                return np.random.rand(1, 256, 64, 64)
        
        class MaskDecoder:
            def __call__(self, **kwargs):
                logits = np.random.rand(1, 1, 256, 256)
                return logits, None
        
        image_encoder = ImageEncoder()
        prompt_encoder = PromptEncoder()
        mask_decoder = MaskDecoder()
    
    model = MockModel()
    
    # Create demo instance
    demo = BboxPromptDemo(model)
    
    # Create test data
    test_2d, _ = create_test_data()
    
    print("2D Demo initialized. Call demo.show(test_2d) to start interactive demo.")
    return demo, test_2d

def test_3d_demo():
    """Test the demo with 3D data"""
    print("Testing 3D Demo...")
    
    # Create mock model (same as above)
    class MockModel:
        def __init__(self):
            self.device = 'cpu'
        
        def eval(self):
            pass
        
        class ImageEncoder:
            def __call__(self, x):
                return np.random.rand(1, 256, 64, 64)
        
        class PromptEncoder:
            def __call__(self, points=None, boxes=None, masks=None, tokens=None):
                sparse = np.random.rand(1, 2, 256)
                dense = np.random.rand(1, 256, 64, 64)
                return sparse, dense
            
            def get_dense_pe(self):
                return np.random.rand(1, 256, 64, 64)
        
        class MaskDecoder:
            def __call__(self, **kwargs):
                logits = np.random.rand(1, 1, 256, 256)
                return logits, None
        
        image_encoder = ImageEncoder()
        prompt_encoder = PromptEncoder()
        mask_decoder = MaskDecoder()
    
    model = MockModel()
    
    # Create demo instance
    demo = BboxPromptDemo(model)
    
    # Create test data
    _, test_3d = create_test_data()
    
    print("3D Demo initialized. Call demo.show(test_3d) to start interactive demo.")
    print("Features available:")
    print("- Use slider to navigate between slices")
    print("- Toggle 'Global BBox' to use same bbox for all slices")
    print("- Click 'Apply Global BBox' to apply stored bbox to all slices")
    print("- Use demo.go_to_slice(n) to navigate programmatically")
    print("- Use demo.export_3d_segmentation() to save 3D results")
    
    return demo, test_3d

def demo_usage_examples():
    """Show usage examples"""
    print("\n" + "="*50)
    print("ENHANCED BboxPromptDemo USAGE EXAMPLES")
    print("="*50)
    
    print("\n1. Basic 2D usage:")
    print("   demo = BboxPromptDemo(model)")
    print("   demo.show(image_2d)  # image_2d shape: (H, W)")
    
    print("\n2. Basic 3D usage:")
    print("   demo = BboxPromptDemo(model)")
    print("   demo.show(image_3d)  # image_3d shape: (D, H, W)")
    
    print("\n3. 3D with global bbox (same bbox for all slices):")
    print("   demo = BboxPromptDemo(model)")
    print("   demo.set_image_data(image_3d)")
    print("   demo.set_global_bbox_mode(True)")
    print("   demo.show()  # Draw bbox, then click 'Apply Global BBox'")
    
    print("\n4. Text prompt support (requires dataroot):")
    print("   demo = BboxPromptDemo(model, use_text_prompts=True, dataroot='/path/to/data')")
    print("   demo.show_text_prompt_interface()")
    
    print("\n5. Programmatic slice navigation:")
    print("   demo.go_to_slice(5)  # Go to slice 5")
    print("   demo.get_all_segmentations()  # Get all slice segmentations")
    print("   demo.export_3d_segmentation('output.npy')  # Save 3D volume")
    
    print("\n6. Load from image path (backward compatibility):")
    print("   demo.show(image_path='/path/to/image.png')")
    
    print("\nNew Features:")
    print("- Direct 2D/3D numpy array input")
    print("- Slice-by-slice navigation for 3D volumes") 
    print("- Global bbox mode for applying same bbox to all slices")
    print("- Individual bbox mode for slice-specific segmentation")
    print("- 3D segmentation export")
    print("- Backward compatibility with image paths")

if __name__ == "__main__":
    print("Enhanced BboxPromptDemo Test Script")
    print("This script demonstrates the new 2D/3D capabilities.")
    print("\nNote: This is a test script with mock models.")
    print("For actual usage, replace MockModel with your trained MedSAM model.")
    
    demo_usage_examples()
    
    # Uncomment to test with mock data:
    # demo_2d, test_2d = test_2d_demo()
    # demo_3d, test_3d = test_3d_demo()
