#!/bin/bash
#
# A script to generate and launch a series of sbatch jobs with various parameter combinations.
# Version 2: Implements dataset-specific job naming and conditional arguments.
#

# --- Step 1: Define Parameters and Configuration ---

# Set to 'true' to use --use_otsu_text, or 'false' for --no-use_otsu_text
readonly USE_OTSU_TEXT_BOOL=true

# Fixed normalisation parameters
readonly NORM_MODE="ct"
readonly WINDOW_MIN_PERCENTILE=1
readonly WINDOW_MAX_PERCENTILE=99

# Define dataset-specific values for --text_model_slice
declare -A TEXT_MODEL_SLICE_VALS
TEXT_MODEL_SLICE_VALS["F20204_Liver_Imaging_T1_ShMoLLI_DICOM_H5v3"]=15
TEXT_MODEL_SLICE_VALS["F20254_Liver_imaging_IDEAL_protocol_DICOM_H5v3"]=20
TEXT_MODEL_SLICE_VALS["F20259_Pancreas_Images_ShMoLLI_DICOM_H5v3"]=12
TEXT_MODEL_SLICE_VALS["F20260_Pancreas_Images_gradient_echo_DICOM_H5v3"]=18

# Base directory paths
readonly BASE_INPUT_DIR="/group/glastonbury/soumick/dataset/ukbbnii/minisets"
readonly BASE_OUTPUT_DIR="/group/glastonbury/soumick/dataset/ukbbnii/minisets"

# Dataset identifiers
readonly DATASETS=(
    "F20204_Liver_Imaging_T1_ShMoLLI_DICOM_H5v3"
    "F20254_Liver_imaging_IDEAL_protocol_DICOM_H5v3"
    "F20259_Pancreas_Images_ShMoLLI_DICOM_H5v3"
    "F20260_Pancreas_Images_gradient_echo_DICOM_H5v3"
)

# The launcher script to be called by sbatch
readonly LAUNCHER_SCRIPT="launcher_dgxtiny_2stageprompt.sh"
# Number of commands to group into a single sbatch job
readonly GROUP_SIZE=3

# --- Step 2: Generate and Launch Jobs for Each Dataset ---

echo "🚀 Preparing to generate and launch sbatch jobs..."
echo "------------------------------------------------------------"

# Determine the Otsu text flag once
if [ "$USE_OTSU_TEXT_BOOL" = true ]; then
    otsu_text_flag="--use_otsu_text"
else
    otsu_text_flag="--no-use_otsu_text"
fi

# Process each dataset individually
for dataset in "${DATASETS[@]}"; do
    echo "Processing dataset: ${dataset}"
    
    # Array to store commands specifically for the current dataset
    dataset_commands=()
    
    input_dir="${BASE_INPUT_DIR}/${dataset}/exported_images/NIFTI/"
    output_dir="${BASE_OUTPUT_DIR}/${dataset}/segmentations/"
    specific_slice=${TEXT_MODEL_SLICE_VALS[$dataset]}

    # --- Generate all combinations for the current dataset ---
    for slice_val in -1 "$specific_slice"; do
        for model_mode in "point" "bbox"; do
            for use_otsu_flag in "--use_otsu" "--no-use_otsu"; do
                for keep_largest_flag in "--keep_largest_only" "--no-keep_largest_only"; do
                    base_cmd="--input_dir ${input_dir} --output_dir ${output_dir} --norm_mode ${NORM_MODE} --window_min_percentile ${WINDOW_MIN_PERCENTILE} --window_max_percentile ${WINDOW_MAX_PERCENTILE} ${otsu_text_flag} --text_model_slice ${slice_val} --second_model_mode ${model_mode} ${use_otsu_flag} ${keep_largest_flag}"
                    
                    if [ "$keep_largest_flag" == "--keep_largest_only" ]; then
                        for fill_holes_flag in "--fill_holes" "--no-fill_holes"; do
                            dataset_commands+=("${base_cmd} ${fill_holes_flag}")
                        done
                    else
                        dataset_commands+=("${base_cmd} --no-fill_holes")
                    fi
                done
            done
        done
    done

    echo "  Generated ${#dataset_commands[@]} combinations for this dataset."

    # --- Group commands and launch sbatch jobs for this dataset ---
    job_serial=1
    for (( i=0; i<${#dataset_commands[@]}; i+=GROUP_SIZE )); do
        group_slice=("${dataset_commands[@]:i:GROUP_SIZE}")
        command_group=$(printf ";%s" "${group_slice[@]}")
        command_group=${command_group:1}

        # Format the job name with a dataset-specific serial number
        job_name="${dataset}_${job_serial}"

        # Check if it's a pancreas dataset to add the --programme argument
        if [[ $dataset == *"Pancreas"* ]]; then
            echo "  Launching Pancreas Job: ${job_name}"
            sbatch -J "${job_name}" "${LAUNCHER_SCRIPT}" --programme minipipes/2stageprompt_pan.py --args "${command_group}"
        else
            echo "  Launching Job: ${job_name}"
            sbatch -J "${job_name}" "${LAUNCHER_SCRIPT}" --args "${command_group}"
        fi
        
        ((job_serial++))
    done
    echo "------------------------------------------------------------"
done

echo "✅ All sbatch jobs have been launched successfully."