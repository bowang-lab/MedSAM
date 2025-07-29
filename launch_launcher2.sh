#!/bin/bash
# This script launches sbatch jobs, with each job running 3 command combinations.
# Total number of jobs to be launched: 42

# --- Job 1: F20204_Liver_Imaging_T1_ShMoLLI_DICOM_H5v3 (1-3) ---
sbatch -J F20204_Liver_1 launcher_dgxtiny_textprompt.sh --args "--input_dir /group/glastonbury/soumick/dataset/ukbbnii/minisets/F20204_Liver_Imaging_T1_ShMoLLI_DICOM_H5v3/exported_images/NIFTI/ --output_dir /group/glastonbury/soumick/dataset/ukbbnii/minisets/F20204_Liver_Imaging_T1_ShMoLLI_DICOM_H5v3/segmentations/MedSAM_textprompt/ --norm_mode minmax --use_otsu --no-keep_largest_only; \
--input_dir /group/glastonbury/soumick/dataset/ukbbnii/minisets/F20204_Liver_Imaging_T1_ShMoLLI_DICOM_H5v3/exported_images/NIFTI/ --output_dir /group/glastonbury/soumick/dataset/ukbbnii/minisets/F20204_Liver_Imaging_T1_ShMoLLI_DICOM_H5v3/segmentations/MedSAM_textprompt/ --norm_mode minmax --use_otsu --keep_largest_only --fill_holes; \
--input_dir /group/glastonbury/soumick/dataset/ukbbnii/minisets/F20204_Liver_Imaging_T1_ShMoLLI_DICOM_H5v3/exported_images/NIFTI/ --output_dir /group/glastonbury/soumick/dataset/ukbbnii/minisets/F20204_Liver_Imaging_T1_ShMoLLI_DICOM_H5v3/segmentations/MedSAM_textprompt/ --norm_mode minmax --use_otsu --keep_largest_only --no-fill_holes"
