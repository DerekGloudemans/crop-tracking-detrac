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
    
     #add argparse block here so we can optinally run from command line
     try:
        parser = argparse.ArgumentParser()
        parser.add_argument("det_steps",help = "list of values separated by commas")
        parser.add_argument("-confs", help = "list separated by commas",default = "0.8")
        parser.add_argument("-skip_steps",help = "list separated by commas",type = str, default = "1")
        parser.add_argument("-gpu",help = "gpu idx from 0-3", type = int,default = 0)
        parser.add_argument("-mode",help = "iou or linear", type = str,default = "iou")
        parser.add_argument("--test",action = "store_true")
        parser.add_argument("--show",action = "store_true")
        parser.add_argument("-save",help = "relative directory to save in", type = str,default = "results/temp")

        args = parser.parse_args()
        
        det_steps = args.det_steps.split(",")
        det_steps = [int(item) for item in det_steps]
        confs = args.confs.split(",")
        confs = [float(item) for item in confs]
        skip_steps = [int(item) for item in args.skip_steps.split(",")]
        TRAIN = not args.test
        GPU_ID = args.gpu
        mode = args.mode
        SHOW = args.show
        out_dir = args.save
        
     except:
         det_steps = [8]
         skip_steps = [1]
         TRAIN = True
         confs = [0.8]
         GPU_ID = 0
         mode = "linear"
         SHOW = True
         out_dir = "results/temp"
         
     tracker_name = "sort" if mode == "linear" else "iou"
     print("\nStarting tracking run with the following settings:")
     print("Training set: {}".format(TRAIN))
     print("Det steps:    {}".format(det_steps))
     print("Skip steps:   {}".format(skip_steps))
     print("Confs:        {}".format(confs))
     print("GPU ID:       {}".format(GPU_ID))
     print("Tracker:      {}".format(tracker_name))
        
     #for TRAIN in [True, False]:   
     for det_step in det_steps:
        d = det_step
        for det_conf_cutoff in confs:
         for skip_step in skip_steps:
             
           det_step = d * skip_step
           # parameters
           overlap = 0.5
           iou_cutoff = 0.75       # tracklets overlapping by this amount will be pruned
           #det_step = 9           # frames between full detection steps
           #skip_step = 1           # frames between update steps (either detection or localization)
           ber = 1.2               # amount by which to expand a priori tracklet to crop object
           init_frames = 1
           #det_conf_cutoff = 0.8   # number of detection steps in a row to perform
           matching_cutoff = 100 if mode == "linear" else 0.7  # detections farther from tracklets than this will not be matched 
           #SHOW = False          # show tracking in progress?
           LOCALIZE = True         # if False, will skip frames between detection steps
           OUTVID = os.path.join(os.getcwd(),"demo","example_outputs")
           #mode = "linear"
           
           #
           loc_cp = "./config/localizer_112.pt"
           det_cp = "./config/detector.pt"

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
               'Truck-Flatbed':12,
               
               0:'Sedan',
               1:'Hatchback',
               2:'Suv',
               3:'Van',
               4:'Police',
               5:'Taxi',
               6:'Bus',
               7:'Truck-Box-Large',
               8:'MiniVan',
               9:'Truck-Box-Med',
               10:'Truck-Util',
               11:'Truck-Pickup',
               12:'Truck-Flatbed',
               }
           
           # get filter
           filter_state_path = "./config/filter_params_tuned_112.cpkl"

           with open(filter_state_path ,"rb") as f:
                    kf_params = pickle.load(f)
                    kf_params["R"] /= 100
                    
           # get localizer
           localizer = resnet50(num_classes=13,device_id = GPU_ID)
           cp = torch.load(loc_cp)
           localizer.load_state_dict(cp) 
           
           # no localization update!
           if not LOCALIZE:
               localizer = None
           
           # get detector
           detector = resnet50(num_classes=13,device_id = GPU_ID)
           try:
               detector.load_state_dict(torch.load(det_cp))
           except:    
               temp = torch.load(det_cp)["model_state_dict"]
               new = {}
               for key in temp:
                   new_key = key.split("module.")[-1]
                   new[new_key] = temp[key]
               detector.load_state_dict(new)
           
           #detector = Mock_Detector(data_paths["detections"],detector = det)
           
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
   
               track_dir = track_dict[id]["frames"]
               save_file = os.path.join(out_dir,"results_{}_{}_{}_{}.cpkl".format(id,det_step,skip_step,int(det_conf_cutoff*10)))
               
                    
               if os.path.exists(save_file):
                   continue
               
               with open(save_file,"wb") as f:
                    pickle.dump(0,f) # empty file to prevent other workers from grabbing this file as well
                    
               # track it!
               tracker = Localization_Tracker(track_dir,
                                              detector,
                                              localizer,
                                              kf_params,
                                              class_dict,
                                              det_step = det_step,
                                              skip_step = skip_step,
                                              init_frames = init_frames,
                                              fsld_max = 1,
                                              det_conf_cutoff = det_conf_cutoff,
                                              matching_cutoff = matching_cutoff,
                                              iou_cutoff = iou_cutoff,
                                              ber = ber,
                                              PLOT = SHOW,
                                              downsample = 1,
                                              distance_mode = mode,
                                              cs = 112,
                                              device_id = GPU_ID)
               
               tracker.track()
               preds, Hz, time_metrics = tracker.get_results()
               
               # get ground truth labels
               gts,metadata = mot.parse_labels(track_dict[id]["labels"])
               ignored_regions = metadata['ignored_regions']
       
               # match and evaluate
               metrics,acc = mot.evaluate_mot(preds,gts,ignored_regions,threshold = 0.3,ignore_threshold = overlap)
               metrics = metrics.to_dict()
               metrics["framerate"] = {0:Hz}
               
               print("\rTrack {} - {}: MOTA: {}  Framerate: {}".format(count,id,np.round(metrics["mota"][0],3),np.round(metrics["framerate"][0],1)))
               
               with open(save_file,"wb") as f:
                   pickle.dump((preds,metrics,time_metrics,Hz),f)
               
               # add results to aggregate results
               try:
                   for key in metrics:
                       running_metrics[key] += metrics[key][0]
               except:
                   for key in metrics:
                       running_metrics[key] = metrics[key][0]
               
               count += 1
                   
           # average results  
           print("\n\nAverage Metrics with {} tracks with det_step {} and skip_step {} and conf {}".format(len(tracks),det_step,skip_step,det_conf_cutoff))
           for key in running_metrics:
               running_metrics[key] /= count
               print("   {}: {}".format(key,running_metrics[key]))
    