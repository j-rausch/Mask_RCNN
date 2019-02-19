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
import copy
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
import skimage

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
    NUM_CLASSES = 1 + 33 #all except content line

#  #table, row, col, content_line, caption, cell 

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

        categories = [ 
            {'id': 1, 'name': 'table', 'supercategory': 'section'},
            {'id': 2, 'name': 'table_caption', 'supercategory': 'table'},
            {'id': 3, 'name': 'table_cell', 'supercategory': 'table'},
            {'id': 4, 'name': 'table_row', 'supercategory': 'table'},
            {'id': 5, 'name': 'table_col', 'supercategory': 'table'},
            {'id': 6, 'name': 'figure', 'supercategory': 'section'},
            {'id': 7, 'name': 'figure_caption', 'supercategory': 'figure'},
            {'id': 8, 'name': 'figure_graphic', 'supercategory': 'figure'},
            {'id': 9, 'name': 'title', 'supercategory': 'meta_section'},
            {'id': 10, 'name': 'section', 'supercategory': 'section'},
            {'id': 11, 'name': 'header', 'supercategory': 'section'},
            {'id': 12, 'name': 'subsection', 'supercategory': 'document'},
            {'id': 13, 'name': 'subsubsection', 'supercategory': 'document'},
            {'id': 14, 'name': 'paragraph', 'supercategory': 'document'},
            {'id': 15, 'name': 'subparagraph', 'supercategory': 'document'},
            {'id': 16, 'name': 'part', 'supercategory': 'document'},
            {'id': 17, 'name': 'content', 'supercategory': 'document'},
            {'id': 18, 'name': 'content_line', 'supercategory': 'content_block'},
            {'id': 19, 'name': 'content_block', 'supercategory': 'section'},
            {'id': 20, 'name': 'authors', 'supercategory': 'meta_section'},
            {'id': 21, 'name': 'author', 'supercategory': 'author'},
            {'id': 22, 'name': 'affiliations', 'supercategory': 'meta_section'},
            {'id': 23, 'name': 'affiliation', 'supercategory': 'affiliations'},
            {'id': 24, 'name': 'abstract', 'supercategory': 'meta_section'},
            {'id': 25, 'name': 'date', 'supercategory': 'meta_section'},
            {'id': 26, 'name': 'bibliography', 'supercategory': 'document'},
            {'id': 27, 'name': 'bibliography_item', 'supercategory': 'bibliography'},
            {'id': 28, 'name': 'equation', 'supercategory': 'document'},
            {'id': 29, 'name': 'equation_element', 'supercategory': 'equation'},
            {'id': 30, 'name': 'equation_label', 'supercategory': 'equation'},
            {'id': 31, 'name': 'meta_section', 'supercategory': 'document'},
            {'id': 32, 'name': 'enumerate', 'supercategory': 'document'},
            {'id': 33, 'name': 'itemize', 'supercategory': 'document'},
            {'id': 34, 'name': 'page_nr', 'supercategory': 'document'}
            ]
	
        all_ids = []
        all_names = []
        self.allowed_class_ids = []
        self.allowed_class_names = []
        for cat in categories:
            all_names.append(cat['name'])
            all_ids.append(cat['id'])
            if	cat['name'] != 'content_line':
                self.allowed_class_ids.append(cat['id'])
                self.allowed_class_names.append(cat['name'])


#        allowed_class_ids = [x for x in all_ids in 
#        allowed_class_names = {1:'table',2:'table_caption',3:'table_cell',4:'table_row',5:'table_col',18:'content_line'}

        with open(dataset_mapping_json, 'r') as fp:
            docs_and_pages = json.load(fp)


        # Load all classes or a subset?
        if not class_ids:
            class_ids = sorted(self.allowed_class_ids)
       
        self.all_ann_json_paths = [] 
        self.all_img_json_paths = []
        self.all_img_dicts = []
        self.all_image_paths = []
        self.page_nrs = []
        progress_count = 0
        for doc_id, _ in docs_and_pages:
            all_doc_images = os.listdir(os.path.join(image_root_path, doc_id))
            #print('all doc images: {}'.format(all_doc_images))
            all_doc_pages = [int(x.split('_page')[1][0:3]) for x in all_doc_images]
            #print('all doc pages: {}'.format(all_doc_pages))
            for page in all_doc_pages:
                ann_json_path = os.path.join(jsons_root_path, doc_id, 'anns.json')
                img_json_path = os.path.join(jsons_root_path, doc_id, 'imgs.json')
                with open(img_json_path, 'r') as fp:
                    if progress_count%500 == 0:
                        logger.info("Loaded {} images".format(progress_count))
                    progress_count += 1
                    json_dict = json.load(fp)
                    all_img_dicts = [x for x in json_dict if x['page'] == page]
                    if len(all_img_dicts) == 0: #faulty json, missing page entry
                        logger.info('skipping document {} page {}: faulty img json'.format(doc_id, page))
                        continue
                    img_dict = all_img_dicts[0]

                self.all_ann_json_paths.append(ann_json_path)
                self.all_img_json_paths.append(img_json_path)
                self.all_img_dicts.append(img_dict)
                img_name = doc_id + "_page{0:0=3d}_feature.png".format(page)
                img_path = os.path.join(image_root_path, doc_id, img_name)
                self.all_image_paths.append(os.path.join(image_root_path, doc_id, img_name))
                self.page_nrs.append(page)
        image_ids = range(len(self.all_image_paths))
        #TODO: debugging, remove later:`
        #image_ids = image_ids[0:(int)(0.2*len(image_ids))]

        # Add classes
        for i in class_ids:
            self.add_class("arxiv", i, self.allowed_class_names[self.allowed_class_ids.index(i)])

        # Add images
        for i in image_ids:
#            if i%500 == 0:
#                logger.info("Loaded {} of {} images".format(i, len(image_ids)))
            #page = self.page_nrs[i]
            img_dict = self.all_img_dicts[i]
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
            
            annotations = [x for x in json_dict if x['page'] == page and x['category_name'] in self.allowed_class_names]
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



def build_arxiv_results(dataset, image_ids, rois, class_ids, scores, masks):
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "arxiv"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


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
    elif args.command == "eval":
        config = ArxivConfig()
        config.IMAGES_PER_GPU =1 
        config.GPU_COUNT=1
        config.BATCH_SIZE=1
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
#
    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)
#    model.load_weights(model_path, by_name=True, exclude=["mrcnn_class_logits", \
#        "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

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
        #augmentation = imgaug.augmenters.Flipud(0.5)
        augmentation = imgaug.augmenters.SomeOf((0,3),
                    [imgaug.augmenters.Fliplr(0.25),
                    imgaug.augmenters.Flipud(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 3.0))
                    ])

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=180,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=200,
                    layers='4+',
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=240,
                    layers='all',
                    augmentation=augmentation)
    elif args.command == 'eval':
        print('doing zurich evaluation..')
        dataset_eval_root = args.dataset
        dataset_eval_json_path = os.path.join(dataset_eval_root, 'test_images_zurich.json')
        with open(dataset_eval_json_path, 'r') as fp:
            dataset_eval_list = json.load(fp)
       
        eval_output_path = os.path.join(dataset_eval_root, 'eval' )
        eval_output_json_path = os.path.join(dataset_eval_root, 'eval_results_'+os.path.splitext(os.path.basename(model_path))[0]+'.json')    
        eval_results_list = []
        print('json path: {}'.format(eval_output_json_path))
        debug_limit = 2
        counter = 0
        for test_doc in dataset_eval_list:
            test_doc_id = test_doc['id']
            test_doc_image_folder = os.path.join(dataset_eval_root, test_doc['images_dir'])
            test_doc_images = [x for x in os.listdir(test_doc_image_folder) if x.endswith('.png')]
            output_images_path = os.path.join(eval_output_path, str(test_doc_id))
    
            #print('test images: {}'.format(test_doc_images))
            #print('output_images path: {}'.format(output_images_path))
            current_results_entry = copy.deepcopy(test_doc)
            
            for doc_image in test_doc_images:
                #print('current image: {}'.format(doc_image))\
                current_page = int(doc_image.split('.png')[0][-3:])

                #print('current results entry page: {}'.format(current_page))
                image_path = os.path.join(test_doc_image_folder, doc_image)
                img = skimage.io.imread(image_path) 
                if img.ndim != 3:
                    img = skimage.color.gray2rgb(image)
                if img.shape[-1] == 4:
                    img = image[..., :3]
                image, window, scale, padding, crop = utils.resize_image(
                    img,
                    min_dim=config.IMAGE_MIN_DIM,
                    min_scale=config.IMAGE_MIN_SCALE,
                    max_dim=config.IMAGE_MAX_DIM,
                    mode=config.IMAGE_RESIZE_MODE)
                #Run object detection
                print('getting results')
                results = model.detect([image], verbose=1)
                print('saving results to dict')
                r = results[0]
                
                if not 'results' in current_results_entry:
                    #current_results_entry['results']['page'] = current_page
                    current_results_entry['results'] = {current_page: {'page': current_page, 'rois':r['rois'].tolist(), 'class_ids':r['class_ids'].tolist(), 'scores':r['scores'].tolist()}}
                else:
                    current_results_entry['results'][current_page] = {'page': current_page, 'rois':r['rois'].tolist(), 'class_ids':r['class_ids'].tolist(), 'scores':r['scores'].tolist()}

            eval_results_list.append(current_results_entry)

            counter += 1
            print('processed doc nr: {}'.format(counter))
#            if counter > debug_limit:
#                break
        print('saving json')
        with open(eval_output_json_path, 'w') as fp:
            json.dump(eval_results_list, fp, indent=1)
 
            
    elif args.command == 'evalarxiv':
        logger.info("Creating test dataset")
        dataset_val = ArxivDataset()
        dataset_val.load_arxiv(args.dataset, "test")
        dataset_val.prepare()
        limit = 100

        image_ids = dataset.image_ids

        # Limit to a subset
        if limit:
            image_ids = image_ids[:limit]

        # Get corresponding COCO image IDs.
        arxiv_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

        results = []
        for i, image_id in enumerate(image_ids):
            # Load image
            image = dataset.load_image(image_id)

            # Run detection
            t = time.time()
            r = model.detect([image], verbose=0)[0]
            t_prediction += (time.time() - t)

            # Convert results to COCO format
            # Cast masks to uint8 because COCO tools errors out on bool
            image_results = build_arxiv_results(dataset, arxiv_image_ids[i:i + 1],
                                               r["rois"], r["class_ids"],
                                               r["scores"],
                                               r["masks"].astype(np.uint8))
            results.extend(image_results)
        


        eval_output_json_path = os.path.join('eval_results_'+os.path.splitext(os.path.basename(model_path))[0]+'.json')    
        print('json path: {}'.format(eval_output_json_path))

        with open(eval_output_json_path, 'w') as fp:
            json.dump(results, fp, indent=1)

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
