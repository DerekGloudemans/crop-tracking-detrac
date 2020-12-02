#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 12:41:35 2020

@author: worklab
"""

import os
import sys,inspect
import numpy as np
import random 
import math
import time
random.seed = 0

import cv2
from PIL import Image
import torch
from torch import nn, optim
from torch.utils import data
from torchvision import transforms,models
from torchvision.transforms import functional as F
import matplotlib.pyplot  as plt
import collections


# add all packages and directories to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0,parent_dir)

from config.data_paths import directories
for item in directories:
    sys.path.insert(0,item)

from _data_utils.detrac.detrac_detection_dataset import Detection_Dataset, class_dict, collate
from _detectors.pytorch_retinanet.retinanet import model, csv_eval
from config.data_paths import data_paths



class Mock_Detector():
    
    def __init__(self,directory,detector = "ACF"):
        """
        directory - string
            directory where all detections are stored. Within directory, each detector should
            have a train and a test folder (ex. ACF-train/, ACF-test/)
        detctor - string
            Indicates which set of precomputed detections should be used. 
            Must be one of: ACF, DPM, R-CNN, CompACT, ground-truth
        """
        
        sub_directory = os.path.join(directory,detector)
        
        self.detector = detector
        
        self.detector_times = {"ACF":1/0.67,
                               "DPM":1/0.17,
                               "R-CNN":1/0.1,
                               "CompACT":1/0.22,
                               "ground-truth":1/30.0
                               }
        
        # store all labels
        # dictionary is keyed by track number
        # each item is a dictionary (one key per frame) containing a 2D tensor in required form
        self.all_data = {}
        
        
        
        for track_detections in os.listdir(sub_directory):
            
            # allocate storage for up to 10000 frames (one list per frame)
            track_data = [[] for i in range(10000)]
            
            # get data from file
            id = int(track_detections.split("_")[1])
            with open(os.path.join(sub_directory,track_detections),"r") as f:
                content = [i.strip() for i in f.readlines()]
            
            # transfer to list of lists
            for obj in content:
                obj = obj.split(",")
                obj = [float(num) for num in obj]
                frame = int(obj[0])
                bbox = torch.tensor([obj[2],obj[3],obj[4],obj[5],obj[1],obj[6]])
                
                
                
                track_data[frame].append(bbox)
            
            # parse list into single tensor and add
            track_dict = {}
            for i,item in enumerate(track_data):
                if len(item)> 0:
                    bboxes = torch.stack(item)
                    track_dict[i] = bboxes
                else:
                    track_dict[i] = torch.empty(1)
            
            # add track_dict to all_data
            self.all_data[id] = track_dict.copy()
            
    def __call__(self,track_id,frame):
        """
        Simulates the detector by returning the detector's output

        Parameters
        ----------
        track_id : int
            track identification number.
        frame : int
            frame number, must be > 0 

        Returns
        -------
        scores - n_detections x 1 tensor of confidence scores in range [0,1]
        detections - n_detections x 4 tensor of bboxes in xmin,ymin,xmax,ymax form
        labels - n_detections x 1 tensor of class numbers for detections
        time - float, average time taken by detector
        """
        
        frame_data = self.all_data[track_id][frame]
        
        try: # get detections
            scores = frame_data[:,5]
            labels = frame_data[:,4]
            bboxes = frame_data[:,:4]
            
            # convert xywh -> xyxy
            new_bboxes = torch.zeros(bboxes.shape)
            new_bboxes[:,0] = bboxes[:,0] 
            new_bboxes[:,1] = bboxes[:,1] 
            new_bboxes[:,2] = bboxes[:,0] + bboxes[:,2]
            new_bboxes[:,3] = bboxes[:,1] + bboxes[:,3]
            bboxes = new_bboxes
            
        except: # no detections or frame DNE
            scores = []
            labels = []
            bboxes = []
            
        det_time = self.detector_times[self.detector]
        
        return scores,labels,bboxes,det_time
            
    def to(self,device):
        """
        Implemented to prevent errors in code execution, det = det.to(device) does nothing
        """
        return self
    
    def eval(self):
        """
        Implemented to prevent errors in code execution, det.eval() does nothing
        """
        pass