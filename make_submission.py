import json
import os

import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo
from pycocotools.coco import COCO

from utils import binary_mask_to_rle

cocoGt = COCO('./data/test.json')
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')  # path to the model we just trained
predictor = DefaultPredictor(cfg)

coco_dt = []

for imgid in cocoGt.imgs:
    image = cv2.imread(os.path.join('./data/test_images', cocoGt.loadImgs(ids=imgid)[0]['file_name']))
    pred = predictor(image)['instances']

    # masks, categories, scores = predictor(image)  # run inference of your model
    scores = pred.get('scores').to('cpu').numpy()
    categories = pred.get('pred_classes').to('cpu').numpy()
    masks = pred.get('pred_masks').to('cpu').numpy()
    n_instances = len(scores)
    if len(categories) > 0:  # If any objects are detected in this image
        for i in range(n_instances):  # Loop all instances
            # save information of the instance in a dictionary then append on coco_dt list
            pred = {'image_id': imgid, 'category_id': int(categories[i]) + 1,
                    'segmentation': binary_mask_to_rle(masks[i, :, :]), 'score': float(scores[i])}
            coco_dt.append(pred)

with open('submission.json', 'w') as f:
    json.dump(coco_dt, f)

