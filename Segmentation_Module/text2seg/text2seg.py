"""

Text2seg project

"""

# import package
from PIL import Image
import requests
import numpy as np
import torch
import pickle
import cv2
from torchvision import transforms
from matplotlib import pyplot as plt
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

import os, sys
import argparse
import copy
import pdb
import json

# from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# load GroundingDINO
from groundingdino.util import box_ops
from groundingdino.util.inference import load_model, load_image, predict, annotate

import sys
sys.path.append('/ssd/text2seg/CLIP_Surgery')

# CLIP Surgery
import clip_surgery as clips

# CLIP original
import clip

# segment anything
from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator,sam_model_registry

from text2seg.utils import *

# set constant values

clip_prompt = {"building":["roof", 
               "building", 
               "construction", 
               "office", 
               "edifice", 
               "architecture", 
               "Establishment", 
               "Facility", 
               "House",
               "Mansion", 
               "Tower",
               "Skyscraper",
               "Monument",
               "Shrine",
               "Temple",
               "Palace",
               "iron roof",
               "cabin"
                          ],
               "road":["Street",
                       "road",
                       "Highway",
                       "Path",
                       "Route",
                       "Lane",
                       "Avenue",
                       "Boulevard",
                       "Way",
                       "Alley",
                       "Drive"
                      ],
               "water":["Liquid",
                       "water"
                      ],
               "barren":["barren",
                       "Desolate",
                       "Sterile",
                       "Unproductive",
                       "Infertile",
                       "Wasteland",
                      ],
               "forest":["Woodland",
                       "Jungle",
                       "Timberland",
                       "Wildwood",
                       "Bush"
                      ],
               "agricultural":["Farming",
                       "Agrarian",
                       "Ranching",
                       "Plantation",
                      ],
               "impervious surfaces":["Impermeable surfaces",
                                      "Non-porous surfaces",
                                      "Sealed surfaces",
                                      "Hard surfaces",
                                      "Concrete surfaces",
                                      "Pavement"
                                      
                      ],
               "low vegetation":["Ground cover",
                                 "Underbrush",
                                 "Shrubs",
                                 "Brush",
                                 "Herbs",
                                 "Grass",
                                 "Sod"
                      ],
               
               "tree":["Forest",
                       "Wood",
                       "tree",
                       "Timber",
                       "Grove",
                       "Sapling"
                      ],
               "car":["Automobile",
                       "Vehicle",
                      "Carriage",
                       "Sedan",
                       "Coupe",
                       "SUV",
                      "Truck",
                      ],
               "clutter":["background",
                       "clutter"
                      ],
               "human":["person",
                       "human"
                      ],
               "background":["background"],
               "dog":["puppy", "dog", "canine"],
               "puppy":["puppy", "dog", "canine"]
              }





class Text2Seg():
    def __init__(self):
        # set device if GPU is available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.load_sam()
        self.load_groundingDINO()
        self.load_CLIPS()

    def load_groundingDINO(self):
        # load GroundingDINO
        groundingDINO_chcheckpoint = "/ssd/text2seg/Pretrained/groundingdino_swint_ogc.pth"
        self.groundingDINO = load_model("/ssd/text2seg/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", groundingDINO_chcheckpoint)
        self.groundingDINO.to(self.device)
        self.groundingDINO.eval()

    def load_sam(self):
        #load SAM
        sam_checkpoint = '/ssd/text2seg/Pretrained/sam_vit_h_4b8939.pth'
        self.sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(self.device))
        sam_checkpoint = "/ssd/text2seg/Pretrained/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=self.device)        
        self.sam = self.sam.to(self.device)
        self.sam.eval()
    
    def load_CLIPS(self):
        # CLIPS
        self.clip_model, self.clip_preprocess = clips.load("ViT-B/16", device=self.device)
        self.clip_model.eval()


    

    @torch.no_grad()
    def retriev(self, elements, search_text): 
        # pdb.set_trace()
        preprocessed_images = [self.clip_preprocess(image).to(self.device) for image in elements]
        tokenized_text = clip.tokenize([search_text]).to(self.device)
        stacked_images = torch.stack(preprocessed_images)
        image_features = self.clip_model.encode_image(stacked_images)
        text_features = self.clip_model.encode_text(tokenized_text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        probs = 100. * image_features @ text_features.T
        return probs[:, 0].softmax(dim=0)



    def get_indices_of_values_above_threshold(self, values, threshold):
        return [i for i, v in enumerate(values) if v > threshold]
    
    
    def get_indices_of_values_topk(self, scores, k):
        try:
            values, indices = torch.topk(scores.squeeze().cpu(), k)
        except:
            return self.get_indices_of_values_topk(scores, k-1)
        
        return indices.tolist() #
    

    def is_a_inside_b(self, a, b):
        x1, y1, x2, y2 = a
        x1_2, y1_2, x2_2, y2_2 = b
        return True if x1 >= x1_2 and y1 >= y1_2 and x2 <= x2_2 and y2 <= y2_2 else False
    
    
    def IoU(self, box1, box2):
        # box = (x1, y1, x2, y2)
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # obtain x1, y1, x2, y2 of the intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # compute the width and height of the intersection
        w = max(0, x2 - x1 + 1)
        h = max(0, y2 - y1 + 1)

        inter = w * h
        iou = inter / (box1_area + box2_area - inter)
        return iou
    
    def filter_box_with_IOU(self, detectronBB, dinoBB, threshold=0.3):
        
        idx = []
        for i in range(len(dinoBB)):
            iou = self.IoU(detectronBB, dinoBB[i])
            if self.is_a_inside_b(dinoBB[i], detectronBB):
                threshold = 0.05
            if iou > threshold:
                idx.append(i)
        return torch.from_numpy(np.array(dinoBB)[idx])
    
    def filter_box_with_IOU_for_BB(self, detectronBB, dinoBB, upper_threshold = 0.2, lower_threshold = 0.005):
        
        idx = []
        for i in range(len(dinoBB)):
            iou = self.IoU(detectronBB, dinoBB[i])
            if iou > lower_threshold and iou < upper_threshold:
                idx.append(i)
        return torch.from_numpy(np.array(dinoBB)[idx])



    def predict_dino(self, image_path, text_prompt):
        """
        Use groundingDINO to predict the bounding boxes of the text prompt.
        Then use the bounding boxes to predict the masks of the text prompt.
        """
        image_source, image = load_image(image_path)
        # check whether the text prompt is string or list of strings
        if type(text_prompt) == str:
            text_prompt = [text_prompt]
        # predict
        boxes_lst = []
        for prompt in text_prompt:
            boxes, logits, phrases = predict(
                model=self.groundingDINO,
                image=image,
                caption=prompt,
                box_threshold=0.35,
                text_threshold=0.25,
            )
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            boxes_lst.extend(boxes)
        
        boxes_lst = torch.stack(boxes_lst)
        self.sam_predictor.set_image(image_source)
        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_lst) * torch.Tensor([W, H, W, H])
        
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy.to(self.device), image_source.shape[:2])
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
        )
        print(masks.shape)
        
        masks = ((masks.sum(dim=0)>0)[0]*1).cpu().numpy()
        print(masks.shape)
        annotated_frame_with_mask = draw_mask(masks, image_source)

        return masks, annotated_frame_with_mask, annotated_frame


    
    def predict_CLIPS(self, image_path, text_prompt):
        """
        Use the CLIPS model to predict the bounding boxes of the text prompt.
        Then use the bounding boxes to predict the masks of the text prompt.
        """
        image_source, image = load_image(image_path)
        # check whether the text prompt is string or list of strings
        if type(text_prompt) == str:
            text_prompt = [text_prompt]
        # predict
        preprocess =  Compose([Resize((224, 224), interpolation=BICUBIC),Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
        cv2_img = cv2.cvtColor(np.array(image_source), cv2.COLOR_RGB2BGR)
        image = preprocess(image).unsqueeze(0).to(self.device)   
        self.sam_predictor.set_image(image_source)
        with torch.no_grad():
            # CLIP architecture surgery acts on the image encoder
            image_features = self.clip_model.encode_image(image)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)

            # Prompt ensemble for text features with normalization
            text_features = clips.encode_text_with_prompt_ensemble(self.clip_model, text_prompt, self.device)

            # Extract redundant features from an empty string
            redundant_features = clips.encode_text_with_prompt_ensemble(self.clip_model, [""], self.device)

            # CLIP feature surgery with costum redundant features
            similarity = clips.clip_feature_surgery(image_features, text_features, redundant_features)[0]
            
            # Inference SAM with points from CLIP Surgery
            points, labels = clips.similarity_map_to_points(similarity[1:, 0], cv2_img.shape[:2], t=0.8)
            self.sam_predictor.set_image(image_source)
            masks, scores, _ = self.sam_predictor.predict(
                point_coords = np.array(points),
                point_labels = labels,
                multimask_output = False,
            )
            masks = np.array(masks)[0, :, :]
            annotated_frame_with_mask = draw_mask(masks, image_source)
            return masks, annotated_frame_with_mask
        

    def predict_CLIP(self, image_path, text_prompt, points_per_side):
        """
        Use the SAM to generate candidate segmentation masks.
        Then use the CLIP to determine if the segemented objects belong to the text prompt.
        """
        self.generic_mask_generator = SamAutomaticMaskGenerator(self.sam, points_per_side=points_per_side)
        image_source, image = load_image(image_path)
        # check whether the text prompt is string or list of strings
        if type(text_prompt) == str:
            text_prompt = [text_prompt]
        # use SAM to generate masks
        segmented_frame_masks = self.generic_mask_generator.generate(image_source)

        # Cut out all masks
        cropped_boxes = []
        for mask in segmented_frame_masks:
            print(mask["segmentation"].shape)
            cropped_boxes.append(segment_image(image_source, mask["segmentation"]).crop(convert_box_xywh_to_xyxy(mask["bbox"])))  
        
        indices_lst = []
        for prompt in text_prompt:
            scores = self.retriev(cropped_boxes, prompt)
            indices = self.get_indices_of_values_above_threshold(scores, 0.05)
            indices_lst.extend(indices)      
            
        segmentation_masks = []
        for seg_idx in np.unique(indices_lst):
            segmentation_mask_image = segmented_frame_masks[seg_idx]["segmentation"]
            segmentation_masks.append(segmentation_mask_image)
        segmentation_masks = 3*(np.array(segmentation_masks).sum(axis=0)>0)

        annotated_frame_with_mask = draw_mask(segmentation_masks, image_source)

        return segmentation_masks, annotated_frame_with_mask
    


    def predict_DINO_CLIP(self, image_path, text_prompt, points_per_side):
        """
        Use the SAM to generate candidate segmentation masks.
        Then use the CLIP to determine if the segemented objects belong to the text prompt.
        """
        image_source, image = load_image(image_path) # (h,w,c), torch(c,h,w)
        # check whether the text prompt is string or list of strings
        if type(text_prompt) == str:
            text_prompt = [text_prompt]
        # predict
        boxes_lst = []
        for prompt in text_prompt:
            boxes, logits, phrases = predict(
                model=self.groundingDINO,
                image=image,
                caption=prompt,
                box_threshold=0.35,
                text_threshold=0.25,
            )
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            boxes_lst.extend(boxes)
        
        boxes_lst = torch.stack(boxes_lst)
        self.sam_predictor.set_image(image_source)
        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_lst) * torch.Tensor([W, H, W, H])
        
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy.to(self.device), image_source.shape[:2])
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
        )
        # pdb.set_trace()
        masks = masks.cpu().numpy()



        # Cut out all masks
        cropped_boxes = []
        for i, mask in enumerate(masks):
            cropped_boxes.append(segment_image(image_source, np.squeeze(mask)).crop(boxes_xyxy.numpy()[i]))  
        
        indices_lst = []
        for prompt in text_prompt:
            scores = self.retriev(cropped_boxes, prompt)
            indices = self.get_indices_of_values_above_threshold(scores, 0.05)
            indices_lst.extend(indices)      
            
        segmentation_masks = []
        for seg_idx in np.unique(indices_lst):
            segmentation_mask_image = np.squeeze(masks[seg_idx])
            segmentation_masks.append(segmentation_mask_image)
        segmentation_masks = 3*(np.array(segmentation_masks).sum(axis=0)>0)

        annotated_frame_with_mask = draw_mask(segmentation_masks, image_source)

        return segmentation_masks, annotated_frame_with_mask
    


    def predict_DETECTRON2_CLIP(self, args):
        """
        Use the SAM to generate candidate segmentation masks.
        Then use the CLIP to determine if the segemented objects belong to the text prompt.
        """
        text_prompt = args.prompt

        predictor = SamPredictor(self.sam)
        output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"

        if type(text_prompt) == str:
            text_prompt = [text_prompt]

        if not os.path.isdir(args.input):
            targets = [args.input]
        else:
            targets = [
                f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
            ]
            targets = [os.path.join(args.input, f) for f in targets]

        os.makedirs(args.output, exist_ok=True)
        
        bbox_path = os.path.join(args.bbox_dir,'bbox_results.txt')
        with open(bbox_path, 'rb') as fr:
            bbox_list = pickle.load(fr)
        # print(bbox_list)

        
        for t in targets:
            print(f"Processing '{t}'...")
            image = cv2.imread(t)
            if image is None:
                print(f"Could not load '{t}' as an image, skipping...")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            bbox = bbox_list[os.path.basename(t).split('.')[0]]


            input_point = np.array([[(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2-30]]).astype(np.int16)
            input_label = np.array([1]) # foreground
            input_box = np.array(bbox)
            
            # masks = generator.generate(image)
            predictor.set_image(image)
            masks, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_box,
                multimask_output=True,
            )



            cropped_boxes = []
            for i, mask in enumerate(masks):
                cropped_boxes.append(segment_image(image, np.squeeze(mask)).crop(bbox))  
            
            indices_lst = []
            for prompt in text_prompt:
                scores = self.retriev(cropped_boxes, prompt)
                # print(scores)
                indices = self.get_indices_of_values_above_threshold(scores, 0.05)
                indices_lst.extend(indices)      
                
            segmentation_masks = []
            for seg_idx in np.unique(indices_lst):
                segmentation_mask_image = np.squeeze(masks[seg_idx])
                segmentation_masks.append(segmentation_mask_image)
            segmentation_masks = 3*(np.array(segmentation_masks).sum(axis=0)>0)

            annotated_frame_with_mask = draw_mask(segmentation_masks, image)
            # plt.imshow(annotated_frame_with_mask)
            # plt.show()



            base = os.path.basename(t)
            base = os.path.splitext(base)[0]



            save_base = args.output
            # print('---------------')
            if output_mode == "binary_mask":
                maskname = f"{base}.png"
                imgname = f"{base}_img.png"
                cv2.imwrite(os.path.join(save_base, maskname), segmentation_masks * 255)
                cv2.imwrite(os.path.join(save_base, imgname), cv2.cvtColor(annotated_frame_with_mask, cv2.COLOR_BGR2RGB))

            else:
                print('else')
                save_file = save_base + ".json"
                with open(save_file, "w") as f:
                    json.dump(segmentation_masks, f)
            ######################################################################################################




            # save_base = os.path.join(args.output, base)
            save_base = args.output
            # print('---------------')
            if output_mode == "binary_mask":
                # os.makedirs(save_base, exist_ok=False)
                i=0
                for mask in masks:
                    i+=1
                    # if i !=3: continue
                    filename = f"_____{base}_{i}.png"
                    cv2.imwrite(os.path.join(save_base, filename), mask * 255)
                    
                # print('save complete')
                # write_masks_to_folder(masks, save_base)
            else:
                print('else')
                save_file = save_base + ".json"
                with open(save_file, "w") as f:
                    json.dump(masks, f)
        print("Done!")
        ############################################################################################




    def predict_DETECTRON2_DINO_CLIP(self, args):
        """
        Use the SAM to generate candidate segmentation masks.
        Then use the CLIP to determine if the segemented objects belong to the text prompt.
        """
        text_prompt = args.prompt

        predictor = SamPredictor(self.sam)
        output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"

        if type(text_prompt) == str:
            text_prompt = [text_prompt]

        if not os.path.isdir(args.input):
            targets = [args.input]
        else:
            targets = [
                f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
            ]
            targets = [os.path.join(args.input, f) for f in targets]

        os.makedirs(args.output, exist_ok=True)
        
        bbox_path = os.path.join(args.bbox_dir,'bbox_results.txt')
        with open(bbox_path, 'rb') as fr:
            bbox_list = pickle.load(fr)
        # print(bbox_list)

        
        for t in targets:
            print(f"Processing '{t}'...")
            image = cv2.imread(t)
            if image is None:
                print(f"Could not load '{t}' as an image, skipping...")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            bbox = bbox_list[os.path.basename(t).split('.')[0]]

            ###########################################

            input_point = np.array([[(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2-30]]).astype(np.int16)
            input_label = np.array([1]) # foreground
            input_box = np.array(bbox)
            
            # masks = generator.generate(image)
            predictor.set_image(image)
            masks, _, _ = predictor.predict(
                point_coords=input_point, # input_point
                point_labels=input_label,
                box=input_box, # input_box
                multimask_output=False,
            )

            #########################################################################
            # DINO + SAM
            image_source, image_pil = load_image(t)
            # print(t)

            boxes_lst = []
            for prompt in text_prompt:
                boxes, logits, phrases = predict(
                    model=self.groundingDINO,
                    image= image_pil,
                    caption=prompt,
                    box_threshold=0.4,
                    text_threshold=0.25,
                )
                annotated_frame = annotate(image_source=image_source , boxes=boxes, logits=logits, phrases=phrases)
                boxes_lst.extend(boxes)
            
            boxes_lst = torch.stack(boxes_lst)
            self.sam_predictor.set_image(image_source)
            H, W, _ = image.shape
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_lst) * torch.Tensor([W, H, W, H])
            
            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy.to(self.device), image_source.shape[:2])
            masks_DINO, _, _ = self.sam_predictor.predict_torch(
                point_coords = None,
                point_labels = input_label,
                boxes = transformed_boxes,
                multimask_output = False,
            )
            # pdb.set_trace()
            masks_DINO = masks_DINO.cpu().numpy()



            cropped_boxes = []
            for i, mask in enumerate(masks_DINO):
                cropped_boxes.append(segment_image(image_source, np.squeeze(mask)).crop(boxes_xyxy.numpy()[i]))  
            
            indices_lst = []
            for prompt in text_prompt:
                scores = self.retriev(cropped_boxes, prompt)
                # print(scores)
                indices = self.get_indices_of_values_above_threshold(scores, 0.05)
                indices_lst.extend(indices)      
                
            segmentation_masks = []
            for seg_idx in np.unique(indices_lst):
                segmentation_mask_image = np.squeeze(masks_DINO[seg_idx])
                segmentation_masks.append(segmentation_mask_image)
            segmentation_masks.append(np.squeeze(masks))
            segmentation_masks = 3*(np.array(segmentation_masks).sum(axis=0)>0)

            annotated_frame_with_mask = draw_mask(segmentation_masks, image)
            # plt.imshow(annotated_frame_with_mask)
            # plt.show()



            base = os.path.basename(t)
            base = os.path.splitext(base)[0]



            ######################################################################################################

            save_base = args.output
            # print('---------------')
            if output_mode == "binary_mask":
                maskname = f"{base}.png"
                imgname = f"{base}_img.png"
                cv2.imwrite(os.path.join(save_base, maskname), segmentation_masks * 255)
                cv2.imwrite(os.path.join(save_base, imgname), cv2.cvtColor(annotated_frame_with_mask, cv2.COLOR_BGR2RGB))

            else:
                print('else')
                save_file = save_base + ".json"
                with open(save_file, "w") as f:
                    json.dump(segmentation_masks, f)
            ######################################################################################################



        ############################################################################################

            # save_base = os.path.join(args.output, base)
            save_base = args.output
            # print('---------------')
            if output_mode == "binary_mask":
                # os.makedirs(save_base, exist_ok=False)
                i=0
                for mask in masks:
                    i+=1
                    # if i !=3: continue
                    filename = f"_____{base}_{i}.png"
                    cv2.imwrite(os.path.join(save_base, filename), mask * 255)
                    
                # print('save complete')
                # write_masks_to_folder(masks, save_base)
            else:
                print('else')
                save_file = save_base + ".json"
                with open(save_file, "w") as f:
                    json.dump(masks, f)
        print("Done!")
        ############################################################################################

    



    def predict_DETECTRON2_DINO_CLIP2(self, args):
        
        """
        Use the SAM to generate candidate segmentation masks.
        Then use the CLIP to determine if the segemented objects belong to the text prompt.
        """
        text_prompt = args.prompt

        predictor = SamPredictor(self.sam)
        output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"

        if type(text_prompt) == str:
            text_prompt = [text_prompt]

        if not os.path.isdir(args.input):
            targets = [args.input]
        else:
            targets = [
                f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
            ]
            targets = [os.path.join(args.input, f) for f in targets]

        os.makedirs(args.output, exist_ok=True)
        bbox_path = os.path.join(args.bbox_dir,'bbox_results.txt')
        with open(bbox_path, 'rb') as fr:
            bbox_list = pickle.load(fr)
        
        for t in targets:
            ########################################################
            ########################################################
            print(f"Processing '{t}'...")
            image = cv2.imread(t)
            if image is None:
                print(f"Could not load '{t}' as an image, skipping...")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            try:
                bbox = bbox_list['0' + os.path.basename(t).split('.')[0]]
            except:
                try:
                    bbox = bbox_list[os.path.basename(t).split('.')[0]]
                except:

                    print('bb가 잘못된 경우 전체 이미지 사용')
                    bbox = [0,0, image.shape[1], image.shape[0]]


            input_point = np.array([[(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2-30]]).astype(np.int16)
            input_label = np.array([1]) # foreground
            input_box = np.array(bbox)
            
            # masks = generator.generate(image)
            predictor.set_image(image)
            masks, _, _ = predictor.predict(
                point_coords=input_point, # input_point
                point_labels=input_label,
                box=input_box, # input_box
                multimask_output=False,
            )

            #########################################################################
            # DINO + SAM
            image_source, image_pil = load_image(t)
            # print(t)
            try:
                boxes_lst = []
                for prompt in text_prompt:
                    boxes, logits, phrases = predict(
                        model=self.groundingDINO,
                        image= image_pil,
                        caption=prompt,
                        box_threshold=0.2,
                        text_threshold=0.25,
                    )
                    annotated_frame = annotate(image_source=image_source , boxes=boxes, logits=logits, phrases=phrases)
                    boxes_lst.extend(boxes)
                boxes_lst = torch.stack(boxes_lst)
                self.sam_predictor.set_image(image_source)
                H, W, _ = image.shape
                boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_lst) * torch.Tensor([W, H, W, H])

                boxes_xyxy = self.filter_box_with_IOU(bbox, boxes_xyxy.tolist(), threshold=0.3)
                # print(boxes_xyxy_)
                
                transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy.to(self.device), image_source.shape[:2])
                # pdb.set_trace()
                batch_size = transformed_boxes.shape[0]
                point_coord_input = torch.unsqueeze(torch.from_numpy(input_point),0).to(self.device)
                point_coord_input = point_coord_input.repeat(batch_size, 1, 1)

                point_labels_input = torch.unsqueeze(torch.from_numpy(input_label),0).to(self.device)
                point_labels_input = point_labels_input.repeat(batch_size, 1)
                masks_DINO, _, _ = self.sam_predictor.predict_torch(
                    point_coords = point_coord_input, # BxNx2
                    point_labels = point_labels_input, # BxN
                    boxes = transformed_boxes, # Bx4
                    multimask_output = False,
                )
                masks_DINO = masks_DINO.cpu().numpy()
                
                


                cropped_boxes = []
                for i, mask in enumerate(masks_DINO):
                    cropped_boxes.append(segment_image(image_source, np.squeeze(mask)).crop(boxes_xyxy.numpy()[i]))  
                
                indices_lst = []

                
                for prompt in text_prompt:
                    scores = self.retriev(cropped_boxes, prompt)

                    indices = self.get_indices_of_values_topk(scores, 2)
                    if isinstance(indices, int):
                        indices = [indices]
                    indices_lst.extend(indices)      
                # print(np.unique(indices_lst))
                segmentation_masks = []
                
                
                for seg_idx in np.unique(indices_lst):
                    # plt.imshow(np.squeeze(masks_DINO[seg_idx]))
                    # plt.show()
                    segmentation_mask_image = np.squeeze(masks_DINO[seg_idx])
                    segmentation_masks.append(segmentation_mask_image)
                

                for mask in masks:
                    segmentation_masks.append(mask)
                ####################################################################

                segmentation_masks = 3*(np.array(segmentation_masks).sum(axis=0)>=args.mask_num_thresh)
                
                
            except Exception as e:
                print(e)
                segmentation_masks = 3*(np.squeeze(masks))
            annotated_frame_with_mask = draw_mask(segmentation_masks, image)
            # plt.imshow(annotated_frame_with_mask)
            # plt.show()



            base = os.path.basename(t)
            base = os.path.splitext(base)[0]




            save_base = args.output
            # print('---------------')
            if output_mode == "binary_mask":
                maskname = f"{base}.png"
                imgname = f"{base}_img.png"
                cv2.imwrite(os.path.join(save_base, maskname), segmentation_masks * 255)
                cv2.imwrite(os.path.join(save_base, imgname), cv2.cvtColor(annotated_frame_with_mask, cv2.COLOR_BGR2RGB))

            else:
                print('else')
                save_file = save_base + ".json"
                with open(save_file, "w") as f:
                    json.dump(segmentation_masks, f)
            ######################################################################################################



    def predict_garments_mask(self, args):
        """
        Use the SAM to generate candidate segmentation masks.
        Then use the CLIP to determine if the segemented objects belong to the text prompt.
        """
        text_prompt = args.prompt

        predictor = SamPredictor(self.sam)
        output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"

        if type(text_prompt) == str:
            text_prompt = [text_prompt]

        if not os.path.isdir(args.input):
            targets = [args.input]
        else:
            targets = [
                f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
            ]
            targets = [os.path.join(args.input, f) for f in targets]

        os.makedirs(args.output, exist_ok=True)
        
        bbox_path = os.path.join(args.bbox_dir,'bbox_results.txt')
        with open(bbox_path, 'rb') as fr:
            bbox_list = pickle.load(fr)
        # print(bbox_list)
        
        
        for t in targets:

            print(f"Processing '{t}'...")
            image = cv2.imread(t)
            if image is None:
                print(f"Could not load '{t}' as an image, skipping...")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            try:
                bbox = bbox_list['0' + os.path.basename(t).split('.')[0]]
            except:
                try:
                    bbox = bbox_list[os.path.basename(t).split('.')[0]]
                except:
                    print('bb가 잘못된 경우 전체 이미지 사용')
                    bbox = [0,0, image.shape[1], image.shape[0]]


            image_source, image_pil = load_image(t)

            boxes_lst = []
            for i, prompt in enumerate(text_prompt):
                boxes, logits, phrases = predict(
                    model=self.groundingDINO,
                    image= image_pil,
                    caption=prompt,
                    box_threshold=0.25,
                    text_threshold=0.25,
                )
                annotated_frame = annotate(image_source=image_source , boxes=boxes, logits=logits, phrases=phrases)
                cv2.imwrite("./tmp/annotated_image"+str(i)+'.png', annotated_frame)
                boxes_lst.extend(boxes)
            boxes_lst = torch.stack(boxes_lst)
            self.sam_predictor.set_image(image_source)
            H, W, _ = image.shape
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_lst) * torch.Tensor([W, H, W, H])

            boxes_xyxy = self.filter_box_with_IOU(bbox, boxes_xyxy.tolist(), threshold=0.3)
            
            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy.to(self.device), image_source.shape[:2])
            batch_size = transformed_boxes.shape[0]
            transformed_boxes_ = transformed_boxes.to('cpu').T
            input_point = torch.unsqueeze(torch.from_numpy(np.array([[((transformed_boxes_[0][:]+transformed_boxes_[2][:])/2).tolist(), ((transformed_boxes_[1][:]+transformed_boxes_[3][:])/2).tolist()]]).astype(np.int16).squeeze().T), 1).to(self.device)

            masks_DINO, _, _ = self.sam_predictor.predict_torch(
                point_coords = input_point, # BxNx2
                point_labels = torch.tensor([1]).repeat(batch_size, 1).to(self.device), # BxN
                boxes = transformed_boxes, # Bx4
                multimask_output = False,
            )
            masks_DINO = masks_DINO.cpu().numpy()

            segmentation_masks = []
            for seg_idx in range(masks_DINO.shape[0]):
                # plt.imshow(np.squeeze(masks_DINO[seg_idx]))
                # plt.show()
                segmentation_mask_image = np.squeeze(masks_DINO[seg_idx])
                segmentation_masks.append(segmentation_mask_image)

            # np.array(segmentation_masks).sum(axis=0)>1      ## (h,w)
            # 3*(np.array(segmentation_masks).sum(axis=0)>1)  ## (h,w,h,w,h,w)
            segmentation_masks = 3*(np.array(segmentation_masks).sum(axis=0)>=args.mask_num_thresh)
                
                

            annotated_frame_with_mask = draw_mask(segmentation_masks, image)
            # plt.imshow(annotated_frame_with_mask)
            # plt.show()



            base = os.path.basename(t)
            base = os.path.splitext(base)[0]


            save_base = args.output
            # print('---------------')
            if output_mode == "binary_mask":
                maskname = f"{base}.png"
                imgname = f"{base}_img.png"
                cv2.imwrite(os.path.join(save_base, maskname), segmentation_masks * 255)
                cv2.imwrite(os.path.join(save_base, imgname), cv2.cvtColor(annotated_frame_with_mask, cv2.COLOR_BGR2RGB))

            else:
                print('else')
                save_file = save_base + ".json"
                with open(save_file, "w") as f:
                    json.dump(segmentation_masks, f)
            ######################################################################################################
    
    def predict_garments_bb(self, args):
        """
        Use the SAM to generate candidate segmentation masks.
        Then use the CLIP to determine if the segemented objects belong to the text prompt.
        """
        text_prompt = args.prompt

        predictor = SamPredictor(self.sam)
        output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"

        if type(text_prompt) == str:
            text_prompt = [text_prompt]

        if not os.path.isdir(args.input):
            targets = [args.input]
        else:
            targets = [
                f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
            ]
            targets = sorted([os.path.join(args.input, f) for f in targets])
            
        if not os.path.isdir(args.mask):
            targets_mask = [args.mask]
        else:
            targets_mask = [
                f for f in os.listdir(args.mask) if not os.path.isdir(os.path.join(args.mask, f))
            ]
            targets_mask = sorted([os.path.join(args.mask, f) for f in targets_mask])

        os.makedirs(args.output, exist_ok=True)
        
        bbox_path = os.path.join(args.bbox_dir,'bbox_results.txt')
        with open(bbox_path, 'rb') as fr:
            bbox_list = pickle.load(fr)
        # print(bbox_list)
        
        
        for t, mask in zip(targets, targets_mask):
            # pdb.set_trace()
            ########################################################
            if t[-6:-4] != '32':
                continue
            ########################################################
            print(f"Processing '{t}'...")
            image = cv2.imread(t)
            mask_image = cv2.imread(mask)
            if image is None or mask_image is None:
                print(f"Could not load '{t}' as an image, skipping...")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            try:
                bbox = bbox_list['0' + os.path.basename(t).split('.')[0]]
            except:
                try:
                    bbox = bbox_list[os.path.basename(t).split('.')[0]]
                except:
                    print('bb가 잘못된 경우 전체 이미지 사용')
                    bbox = [0,0, image.shape[1], image.shape[0]]


            image_source, image_pil = load_image(t)

            boxes_lst = []
            for i, prompt in enumerate(text_prompt):
                boxes, logits, phrases = predict(
                    model=self.groundingDINO,
                    image= image_pil,
                    caption=prompt,
                    box_threshold=0.25,
                    text_threshold=0.25,
                )
                annotated_frame = annotate(image_source=image_source , boxes=boxes, logits=logits, phrases=phrases)
                # cv2.imwrite("./tmp/annotated_image"+str(i)+'.png', annotated_frame)
                boxes_lst.extend(boxes)
            boxes_lst = torch.stack(boxes_lst)
            self.sam_predictor.set_image(image_source)
            H, W, _ = image.shape
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_lst) * torch.Tensor([W, H, W, H])
            
            # print(boxes_xyxy)

            boxes_xyxy = self.filter_box_with_IOU_for_BB(bbox, boxes_xyxy.tolist())
            
            
            boxes_xyxy = boxes_xyxy.numpy().tolist()
            garment_mask = np.zeros((H,W))
            for bbox in boxes_xyxy:
                mask = np.zeros((H,W))


                x_min = int(bbox[0])
                y_min = int(bbox[1])
                x_max = int(bbox[2])
                y_max = int(bbox[3])


                mask[y_min:y_max, x_min:x_max] = 1
                garment_mask += mask
                
            garment_mask = garment_mask>=args.mask_num_thresh
            # pdb.set_trace()
            
            
        
            garment_mask_ = np.repeat(garment_mask[:, :, np.newaxis], 3, axis=2)
            mask_image = mask_image * garment_mask_
            
            annotated_frame_with_mask = draw_mask(mask_image[:,:,0]/255, image)
            mask_image = cv2.cvtColor(mask_image[:,:,0], cv2.COLOR_BGR2RGB)
            
            base = os.path.basename(t)
            base = os.path.splitext(base)[0]



            ######################################################################################################

            save_base = args.output
            # print('---------------')
            if output_mode == "binary_mask":
                maskname = f"{base}.png"
                imgname = f"{base}_img.png"
                maskimgname = f"{base}_garments.png"
                # cv2.imwrite(os.path.join(save_base, maskname), garment_mask * 255)
                cv2.imwrite(os.path.join(save_base, imgname), cv2.cvtColor(annotated_frame_with_mask, cv2.COLOR_BGR2RGB))
                cv2.imwrite(os.path.join(save_base, maskimgname), cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB))
            
     


    def predict_SAM(self, args):
        """
        Use the SAM to generate candidate segmentation masks.
        Then use the CLIP to determine if the segemented objects belong to the text prompt.
        """
        text_prompt = args.prompt

        predictor = SamPredictor(self.sam)
        output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"

        if type(text_prompt) == str:
            text_prompt = [text_prompt]

        if not os.path.isdir(args.input):
            targets = [args.input]
        else:
            targets = [
                f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
            ]
            targets = [os.path.join(args.input, f) for f in targets]

        os.makedirs(args.output, exist_ok=True)
        
        bbox_path = os.path.join(args.bbox_dir,'bbox_results.txt')
        with open(bbox_path, 'rb') as fr:
            bbox_list = pickle.load(fr)
        # print(bbox_list)

        
        for t in targets:
            print(f"Processing '{t}'...")
            image = cv2.imread(t)
            if image is None:
                print(f"Could not load '{t}' as an image, skipping...")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            try:
                bbox = bbox_list[os.path.basename(t).split('.')[0]]
            except:
                ###########################################

                bbox = [0,0, 1280, 720]
                print('bb가 잘못된 경우 전체 이미지 사용')
                # bbox = bbox

            input_point = np.array([[(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2-30]]).astype(np.int16)
            input_label = np.array([1]) # foreground
            input_box = np.array(bbox)
            
            # masks = generator.generate(image)
            predictor.set_image(image)
            masks, _, _ = predictor.predict(
                point_coords=input_point, # input_point
                point_labels=input_label,
                box=input_box, # input_box
                multimask_output=False,
            )

            
            annotated_frame_with_mask = draw_mask(masks, image)
            # plt.imshow(annotated_frame_with_mask)
            # plt.show()



            base = os.path.basename(t)
            base = os.path.splitext(base)[0]
            segmentation_masks = 3*(np.squeeze(masks))




            save_base = args.output
            # print('---------------')
            if output_mode == "binary_mask":
                maskname = f"{base}.png"
                imgname = f"{base}_img.png"
                cv2.imwrite(os.path.join(save_base, maskname), segmentation_masks * 255)
                cv2.imwrite(os.path.join(save_base, imgname), cv2.cvtColor(annotated_frame_with_mask, cv2.COLOR_BGR2RGB))

            else:
                print('else')
                save_file = save_base + ".json"
                with open(save_file, "w") as f:
                    json.dump(masks, f)