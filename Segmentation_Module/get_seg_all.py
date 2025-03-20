import os
from text2seg.text2seg import Text2Seg
import argparse
import pdb

def main(args: argparse.Namespace) -> None:
    # Get a list of subdirectories in the input directory
    subdirectories = [d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d))]
    arg_input = args.input
    arg_bbox = args.bbox_dir
    arg_output = args.output

    for subdir in subdirectories:
        # Initialize Text2Seg object for each subdirectory
        test = Text2Seg()

        # Construct input and bbox paths for the current subdirectory
        sub_input_path = os.path.join(arg_input, subdir)
        sub_bbox_path = os.path.join(arg_bbox, subdir)

        # Create an output directory for masks for the current subdirectory
        sub_output_path = os.path.join(arg_output, subdir)
        os.makedirs(sub_output_path, exist_ok=True)

        # Construct a list of input images for the current subdirectory
        
        input_images = [os.path.join(sub_input_path, f) for f in os.listdir(sub_input_path) if f.endswith('.jpg')]
        
        # Process masks for the current subdirectory
        for input_image in input_images:
            args.input = input_image
            args.bbox_dir = sub_bbox_path
            args.output = sub_output_path

            # Call the predict function for the current input image
            test.predict_DETECTRON2_DINO_CLIP2(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Runs automatic mask generation on an input image or directory of images, "
            "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
            "as well as pycocotools if saving in RLE format."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input directory containing subdirectories of images.",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the directory where masks will be output for each subdirectory.",
    )

    parser.add_argument(
        "--bbox_dir",
        type=str,
        required=True,
        help="Path to the directory containing bbox data for each subdirectory.",
    )

    parser.add_argument(
    "--prompt",
    type=lambda prompt: [str(item) for item in prompt.split(',')],
    required=True,
    help="text prompt",
    default="clothed human, human"
    )

    parser.add_argument(
        "--convert-to-rle",
        action="store_true",
        help=(
            "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
            "Requires pycocotools."
        ),
    )

    parser.add_argument(
        "--mask_num_thresh",
        type=int,
        help="If the number of pixels of masks exceed threshold, that will effect output",
        default=2
    )
    args = parser.parse_args()
    main(args)
