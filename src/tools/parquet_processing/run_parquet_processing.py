from src.tools.parquet_processing.parquet_processor import ParquetProcessor

# Initialize
processor = ParquetProcessor()

# 1. Merge existing datasets (HPC dataset takes priority)
processor.merge(
    "src/image_data/images_hpc.parquet",
    "src/image_data/yolo_ds_images_dev_hpc.parquet"
)

# 2. Filter out low confidence predictions
processor.filter_by_confidence(threshold=0.5)

# 3. Keep only training data
# processor.filter_by_split(['train'])

# 4. Check stats
processor.summarize()

# 5. Save result (without massive mask bytes if running locally)
processor.save("src/image_data/cleaned_merged.parquet", include_mask_bytes=False)