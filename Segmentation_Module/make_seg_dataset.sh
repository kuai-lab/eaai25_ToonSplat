#!/bin/bash

# Navigate to the detectron2/demo directory
conda activate text2seg
cd detectron2/demo

# Run the first command
python demo_all.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --input /input_path --output /output_path  --opts MODEL.WEIGHTS ./model_final_2d9806.pkl

# Move back to the original directory
cd ../..

# Run the second command
python get_seg_all.py --input /dataset_path --output /output_dataset_path --bbox_dir /bbox_path --prompt "clothed human, human" --mask_num_thresh 2
conda deactivate