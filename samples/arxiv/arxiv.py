"""
Mask R-CNN
Configurations and data loading code for ARXIV.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained ARXIV weights
    python3 arxiv.py train --dataset=/path/to/arxiv/ --model=arxiv

    # Train a new model starting from ImageNet weights. Also auto download ARXIV dataset
    python3 arxiv.py train --dataset=/path/to/arxiv/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 arxiv.py train --dataset=/path/to/arxiv/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 arxiv.py train --dataset=/path/to/arxiv/ --model=last

    # Run ARXIV evaluatoin on the last model you trained
    python3 arxiv.py evaluate --dataset=/path/to/arxiv/ --model=last
"""


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
import os
import json
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152\n",
os.environ["CUDA_VISIBLE_DEVICES"]="0"#,1"#2,3,4,5,6,7"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imageaug)

import zipfile
import urllib.request
import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils


from pycocotools import mask as maskUtils

# Path to trained weights file
ARXIV_MODEL_PATH = os.path.join(ROOT_DIR, 'data', 'models', "mask_rcnn_arxiv.h5")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'data', 'models', "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class ArxivConfig(Config):
    """Configuration for training on ARXIV.
    Derives from the base Config class and overrides values specific
    to the ARXIV dataset.
    """
    # Give the configuration a recognizable name
    NAME = "arxiv"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  #table, row, col, content_line, caption, cell 

    MAX_GT_INSTANCES = 400

    #MAX_WORKERS = 7
    #MAX_WORKERS = 1

    DETECTION_MAX_INSTANCES = 400
    

    IMAGE_MIN_DIM = 600
    IMAGE_MAX_DIM = 1024

############################################################
#  Dataset
############################################################

class ArxivDataset(utils.Dataset):
    def load_arxiv(self, dataset_dir, subset, class_ids=None,
                  class_map=None):
        """Load a subset of the ARXIV dataset.
        dataset_dir: The root directory of the ARXIV dataset.
        subset: What to load (train, val, minival, valminusminival)
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        """

        dataset_mapping_json = 'tables_'+subset+'.json'
        image_root_path = os.path.join(dataset_dir, 'images')
        jsons_root_path = os.path.join(dataset_dir, 'jsons')


        allowed_class_ids = [1,2,3,4,5,18]
        allowed_class_names = {1:'table',2:'table_caption',3:'table_cell',4:'table_row',5:'table_col',18:'content_line'}

        with open(dataset_mapping_json, 'r') as fp:
            docs_and_pages = json.load(fp)


        # Load all classes or a subset?
        if not class_ids:
            class_ids = sorted(allowed_class_ids)
       
        self.all_ann_json_paths = [] 
        self.all_img_json_paths = []
        self.all_image_paths = []
        self.page_nrs = []
        for doc_id, pages in docs_and_pages:
            for page in pages:
                self.all_ann_json_paths.append(os.path.join(jsons_root_path, doc_id, 'table_anns.json'))
                self.all_img_json_paths.append(os.path.join(jsons_root_path, doc_id, 'imgs.json'))
                img_name = doc_id + "_page{0:0=3d}_feature.png".format(page)
                self.all_image_paths.append(os.path.join(image_root_path, doc_id, img_name))
                self.page_nrs.append(page)
        image_ids = range(len(self.all_image_paths))
        #TODO: debugging, remove later:`
        #image_ids = image_ids[0:(int)(0.2*len(image_ids))]

        # Add classes
        for i in class_ids:
            self.add_class("arxiv", i, allowed_class_names[i])

        # Add images
        for i in image_ids:
            if i%500 == 0:
                logger.info("Loaded {} of {} images".format(i, len(image_ids)))
            page = self.page_nrs[i]
            with open(self.all_img_json_paths[i], 'r') as fp:
                json_dict = json.load(fp)
                img_dict = [x for x in json_dict if x['page'] == page][0]
            self.add_image(
                "arxiv", image_id=i,
                path=self.all_image_paths[i],
                width=img_dict['width'],
                height=img_dict["height"])
                #annotations=ann_list)


    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a ARXIV image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "arxiv":
            return super(ArxivDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []

        page = self.page_nrs[image_id]
        with open(self.all_ann_json_paths[image_id], 'r') as fp:
            json_dict = json.load(fp)
            #logger.info("page: {}, json dict {} has {} anns for image id: {}".format(page, self.all_ann_json_paths[image_id], len(json_dict), image_id))
            
            annotations = [x for x in json_dict if x['page'] == page]
        #logger.info("fetched {} anns for image id: {}".format(len(annotations), image_id))

        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "arxiv.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)
        #logger.info("finished creating {} instance masks for image id: {}".format(len(instance_masks), image_id))

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(ArxivDataset, self).load_mask(image_id)

#    def image_reference(self, image_id):
#        """Return a link to the image in the ARXIV Website."""
#        info = self.image_info[image_id]
#        if info["source"] == "arxiv":
#            return "http://arxivdataset.org/#explore?id={}".format(info["id"])
#        else:
#            super(ArxivDataset, self).image_reference(image_id)

    # The following two functions are from pyarxivtools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        #segm = ann['segmentation']
        #segm = ann['segmentation']
        segm = ann['bbox']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            segm_array = np.asarray([segm])

            rles = maskUtils.frPyObjects(segm_array, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


#############################################################
##  ARXIV Evaluation
#############################################################
#
#def build_arxiv_results(dataset, image_ids, rois, class_ids, scores, masks):
#    """Arrange resutls to match ARXIV specs in http://arxivdataset.org/#format
#    """
#    # If no results, return an empty list
#    if rois is None:
#        return []
#
#    results = []
#    for image_id in image_ids:
#        # Loop through detections
#        for i in range(rois.shape[0]):
#            class_id = class_ids[i]
#            score = scores[i]
#            bbox = np.around(rois[i], 1)
#            mask = masks[:, :, i]
#
#            result = {
#                "image_id": image_id,
#                "category_id": dataset.get_source_class_id(class_id, "arxiv"),
#                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
#                "score": score,
#                "segmentation": maskUtils.encode(np.asfortranarray(mask))
#            }
#            results.append(result)
#    return results
#
#
#def evaluate_arxiv(model, dataset, arxiv, eval_type="bbox", limit=0, image_ids=None):
#    """Runs official ARXIV evaluation.
#    dataset: A Dataset object with valiadtion data
#    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
#    limit: if not 0, it's the number of images to use for evaluation
#    """
#    # Pick ARXIV images from the dataset
#    image_ids = image_ids or dataset.image_ids
#
#    # Limit to a subset
#    if limit:
#        image_ids = image_ids[:limit]
#
#    # Get corresponding ARXIV image IDs.
#    arxiv_image_ids = [dataset.image_info[id]["id"] for id in image_ids]
#
#    t_prediction = 0
#    t_start = time.time()
#
#    results = []
#    for i, image_id in enumerate(image_ids):
#        # Load image
#        image = dataset.load_image(image_id)
#
#        # Run detection
#        t = time.time()
#        r = model.detect([image], verbose=0)[0]
#        t_prediction += (time.time() - t)
#
#        # Convert results to ARXIV format
#        # Cast masks to uint8 because ARXIV tools errors out on bool
#        image_results = build_arxiv_results(dataset, arxiv_image_ids[i:i + 1],
#                                           r["rois"], r["class_ids"],
#                                           r["scores"],
#                                           r["masks"].astype(np.uint8))
#        results.extend(image_results)
#
#    # Load results. This modifies results with additional attributes.
#    arxiv_results = arxiv.loadRes(results)
#
#    # Evaluate
#    arxivEval = ARXIVeval(arxiv, arxiv_results, eval_type)
#    arxivEval.params.imgIds = arxiv_image_ids
#    arxivEval.evaluate()
#    arxivEval.accumulate()
#    arxivEval.summarize()
#
#    print("Prediction time: {}. Average {}/image".format(
#        t_prediction, t_prediction / len(image_ids)))
#    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on ARXIV.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' on ARXIV")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/arixv/",
                        help='Directory of the ARXIV dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'arxiv'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = ArxivConfig()
    else:
        class InferenceConfig(ArxivConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()[1]
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    #model.load_weights(model_path, by_name=True)
    model.load_weights(model_path, by_name=True, exclude=["mrcnn_class_logits", \
        "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        logger.info("Creating train dataset")
        dataset_train = ArxivDataset()
        dataset_train.load_arxiv(args.dataset, "train")
        dataset_train.prepare()

        # Validation dataset
        logger.info("Creating test dataset")
        dataset_val = ArxivDataset()
        dataset_val.load_arxiv(args.dataset, "test")
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Flipud(0.2)

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+',
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all',
                    augmentation=augmentation)

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
