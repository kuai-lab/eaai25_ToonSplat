# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import imageio
import pickle
import pdb

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        type=str,
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


def process_folders_in_directory(input_folder, output_folder, cfg, demo):
    for folder in os.listdir(input_folder):
        # pdb.set_trace()
        folder_path = os.path.join(input_folder, folder)
        if os.path.isdir(folder_path):
            input_files = glob.glob(os.path.join(folder_path, "*.jpg"))
            for input_file in tqdm.tqdm(input_files, disable=not args.output):
                img = read_image(input_file, format="BGR")
                start_time = time.time()
                predictions, visualized_output = demo.run_on_image(img)
                
                try:
                    person_idx = (
                        predictions["instances"]
                        .get_fields()["pred_classes"] == 0
                    ).nonzero()[0].cpu().item()
                except:
                    # os.remove(input_file)
                    # print(f"{input_file} 파일이 삭제되었습니다.")
                    continue

                img_name = os.path.splitext(os.path.basename(input_file))[0]
                bb_dict[img_name] = predictions["instances"].pred_boxes[
                    person_idx
                ].tensor.to("cpu").numpy().tolist()[0]

                logger.info(
                    "{}: {} in {:.2f}s".format(
                        input_file,
                        "detected {} instances".format(len(predictions["instances"]))
                        if "instances" in predictions
                        else "finished",
                        time.time() - start_time,
                    )
                )


        output_folder_ = os.path.join(output_folder, folder)
        os.makedirs(output_folder_, exist_ok=True)
        with open(os.path.join(output_folder_, 'bbox_results.txt'), "wb") as fw:
            pickle.dump(bb_dict, fw)

# ...

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    bb_dict = {}
    process_folders_in_directory(args.input, args.output, cfg, demo)