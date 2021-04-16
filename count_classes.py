import argparse
import os,sys,inspect
import numpy as np
import random 
import time
import math
import _pickle as pickle
random.seed = 0

import cv2
from PIL import Image
import torch

import matplotlib.pyplot  as plt

from config.data_paths import data_paths

# detector_path = os.path.join(os.getcwd(),"models","pytorch_retinanet_detector")
# sys.path.insert(0,detector_path)

detector_path = os.path.join(os.getcwd(),"models","py_ret_det_multigpu")
sys.path.insert(0,detector_path)

detrac_util_path = os.path.join(os.getcwd(),"util_detrac")
sys.path.insert(0,detrac_util_path)
eval_path = os.path.join(os.getcwd(),"util_eval","py_motmetrics")
sys.path.insert(0,eval_path)

from models.py_ret_det_multigpu.retinanet.model import resnet50 

from util_detrac.detrac_detection_dataset import class_dict
from util_eval import mot_eval as mot
from tracker_fsld_112 import Localization_Tracker


def get_track_dict(TRAIN):
    # get list of all files in directory and corresponding path to track and labels
    if TRAIN:
        track_dir = data_paths["train_im"]
        label_dir = data_paths["train_lab"]
    else:
        track_dir = data_paths["test_im"]
        label_dir = data_paths["test_lab"]
    track_list = [os.path.join(track_dir,item) for item in os.listdir(track_dir)]  
    label_list = [os.path.join(label_dir,item) for item in os.listdir(label_dir)]
    track_dict = {}
    for item in track_list:
        id = int(item.split("MVI_")[-1])
        track_dict[id] = {"frames": item,
                          "labels": None}
    for item in label_list:
        if not TRAIN:
            id = int(item.split("MVI_")[-1].split(".xml")[0])
        else:
            id = int(item.split("MVI_")[-1].split("_v3.xml")[0])
        
        track_dict[id]['labels'] = item
    return track_dict

if __name__ == "__main__":
    
        
         det_steps = [1,8]
         TRAIN = False
         confs = [0]
         GPU_ID = 0
         mode = "iou"
         SHOW = False
         
         truncation_count = 0
         class_dict = {
               'Sedan':0,
               'Hatchback':1,
               'Suv':2,
               'Van':3,
               'Police':4,
               'Taxi':5,
               'Bus':6,
               'Truck-Box-Large':7,
               'MiniVan':8,
               'Truck-Box-Med':9,
               'Truck-Util':10,
               'Truck-Pickup':11,
               'Truck-Flatbed':12
               }
           
         class_count = np.zeros([13])
           
         # get track_dict
         track_dict = get_track_dict(TRAIN)         
         tracks = [key for key in track_dict]
         tracks.sort()  
         #tracks.reverse()
         #override tracks with a shorter list
         #tracks = [39761,40141,40213,40241,40963,40992,63521]
         #tracks = [40863,40864,40892,40763,39501,39511,40761,40903]
         
         # for each track and for specified det_step, track and evaluate
         running_metrics = {}
         count = 0
         for id in tracks:
             if id  in [40712,40774,40773,40772,40771,40711,40792,40775,39361,40901]:
                 continue
             # get ground truth labels
             gts,metadata = mot.parse_labels(track_dict[id]["labels"])
             
             for frame in gts:
                 for det in frame:
                     cls = det["class_num"]
                     if det["truncation"] > 0.5:
                        truncation_count += 1
                     class_count[cls] += 1
             print("Finished {}".format(id))
        
         total_objs = np.sum(class_count)
         for key in class_dict:
            print("{}: {}".format(key,class_count[class_dict[key]]))
            
         print("{} total objects".format(total_objs))
    