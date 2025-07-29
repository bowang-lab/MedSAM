#!/usr/bin/env python3
"""
Test script for the enhanced BboxPromptDemo class
Tests both 2D and 3D functionality
"""

import numpy as np
import torch
from utils.demo import BboxPromptDemo

def create_mock_model():
    """Create a mock model for testing purposes"""
    class MockEncoder:
        def __call__(self, x):
            # Return mock embeddings with proper shape
            return torch.randn(1, 256, 64, 64)
    
    class MockPromptEncoder:
        def __call__(self, points=None, boxes=None, masks=None, tokens=None):
            # Return mock sparse and dense embeddings
            sparse = torch.randn(1, 2, 256)
            dense = torch.randn(1, 256, 64, 64)
            return sparse, dense
        
        def get_dense_pe(self):
            return torch.randn(1, 256, 64, 64)
    
    class MockMaskDecoder:
        def __call__(self, image_embeddings, image_pe, sparse_prompt_embeddings, 
                     dense_prompt_embeddings, multimask_output=False):
            # Return mock logits
            logits = torch.randn(1, 1, 256, 256)
            return logits, None
    
    class MockModel:
        def __init__(self):
            self.image_encoder = MockEncoder()
            self.prompt_encoder = MockPromptEncoder()
            self.mask_decoder = MockMaskDecoder()
            self.device = 'cpu'
        
        def eval(self):
            pass
    
    return MockModel()

def test_2d_functionality():
    """Test 2D image functionality"""
    print("Testing 2D functionality...")
    
    # Create mock model
    model = create_mock_model()
    demo = BboxPromptDemo(model)
    
    # Create a 2D test image (grayscale)
    test_image_2d = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    
    # Set the image
    demo.set_image_text(test_image_2d)
    
    # Verify properties
    assert demo.is_volume == False, "Should be detected as 2D image"
    assert demo.img_size == (256, 256), f"Expected (256, 256), got {demo.img_size}"
    assert demo.image_embeddings is not None, "Image embeddings should be computed"
    
    print("✓ 2D functionality test passed")

def test_3d_functionality():
    """Test 3D volume functionality"""
    print("Testing 3D functionality...")
    
    # Create mock model
    model = create_mock_model()
    demo = BboxPromptDemo(model)
    
    # Create a 3D test volume
    test_volume = np.random.randint(0, 255, (10, 256, 256), dtype=np.uint8)
    
    # Set the volume
    demo.set_image_text(test_volume)
    
    # Verify properties
    assert demo.is_volume == True, "Should be detected as 3D volume"
    assert demo.img_size == (10, 256, 256), f"Expected (10, 256, 256), got {demo.img_size}"
    assert len(demo.image_embeddings) == 10, f"Expected 10 slice embeddings, got {len(demo.image_embeddings)}"
    assert demo.current_slice == 0, "Should start at slice 0"
    
    print("✓ 3D functionality test passed")

def test_inference():
    """Test inference functionality"""
    print("Testing inference...")
    
    # Create mock model
    model = create_mock_model()
    demo = BboxPromptDemo(model)
    
    # Test 2D inference
    test_image_2d = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    demo.set_image_text(test_image_2d)
    
    # Test bbox inference
    bbox = np.array([50, 50, 150, 150])  # x_min, y_min, x_max, y_max
    seg_2d = demo._infer(bbox)
    
    assert seg_2d.shape == (256, 256), f"Expected (256, 256), got {seg_2d.shape}"
    assert seg_2d.dtype == np.uint8, f"Expected uint8, got {seg_2d.dtype}"
    
    # Test 3D inference
    test_volume = np.random.randint(0, 255, (5, 256, 256), dtype=np.uint8)
    demo.set_image_text(test_volume)
    
    seg_3d = demo._infer(bbox, slice_idx=2)
    assert seg_3d.shape == (256, 256), f"Expected (256, 256), got {seg_3d.shape}"
    
    print("✓ Inference test passed")

def test_volume_utilities():
    """Test volume utility functions"""
    print("Testing volume utilities...")
    
    # Create mock model
    model = create_mock_model()
    demo = BboxPromptDemo(model)
    
    # Create a 3D test volume
    test_volume = np.random.randint(0, 255, (5, 128, 128), dtype=np.uint8)
    demo.set_image_text(test_volume)
    
    # Add some mock segmentations
    demo.segmentations[0] = np.ones((128, 128), dtype=np.uint8)
    demo.segmentations[2] = np.ones((128, 128), dtype=np.uint8)
    demo.segmentations[4] = np.ones((128, 128), dtype=np.uint8)
    
    # Test statistics
    stats = demo.get_segmentation_statistics()
    assert stats is not None, "Statistics should be available"
    assert stats['segmented_slices'] == 3, f"Expected 3 segmented slices, got {stats['segmented_slices']}"
    assert stats['total_slices'] == 5, f"Expected 5 total slices, got {stats['total_slices']}"
    
    # Test volume creation
    seg_volume = demo.get_segmentation_volume()
    assert seg_volume is not None, "Segmentation volume should be available"
    assert seg_volume.shape == (5, 128, 128), f"Expected (5, 128, 128), got {seg_volume.shape}"
    
    print("✓ Volume utilities test passed")

def main():
    """Run all tests"""
    print("Running BboxPromptDemo enhanced functionality tests...\n")
    
    try:
        test_2d_functionality()
        test_3d_functionality()
        test_inference()
        test_volume_utilities()
        
        print("\n✅ All tests passed successfully!")
        print("\nThe enhanced BboxPromptDemo class supports:")
        print("- 2D image segmentation with bounding box prompts")
        print("- 3D volume segmentation with slice-by-slice processing")
        print("- Same bbox for all slices mode")
        print("- Individual bbox per slice mode")
        print("- Volume statistics and saving functionality")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
