"""
This file provides a dataset class for working with the UA-detrac tracking dataset.
Provides:
    - plotting of 2D bounding boxes
    - training/testing loader mode (random images from across all tracks) using __getitem__()
    - track mode - returns a single image, in order, using __next__()
"""

import os,sys
import numpy as np
import random 
import math
random.seed = 0

import cv2
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


sys.path.insert(0,os.getcwd)
from detrac_plot import pil_to_cv, plot_bboxes_2d


class Track_Dataset(data.Dataset):
    """
    Creates an object for referencing the UA-Detrac 2D object tracking dataset
    and returning single object images for localization. Note that this dataset
    does not automatically separate training and validation data, so you'll 
    need to partition data manually by separate directories
    """
    
    def __init__(self, image_dir,label_dir,n = 8):
        """ initializes object
        image dir - (string) - a directory containing a subdirectory for each track sequence
        label dir - (string) - a directory containing a label file per sequence
        """
        self.n = n
        self.frame_objs = {}
        # parse labels and store in dict keyed by track name
        label_list = []
        im_list = []
        for item in os.listdir(label_dir):
            name = item.split("_v3.xml")[0].split("MVI_")[-1]
            if int(name) in  [20012,20034,63525,63544,63552,63553,63554,63561,63562,63563]:
                #print("Removed Validation tracks, gotta maintain data separation!")
                continue

            detections = self.parse_labels(os.path.join(label_dir,item))[0]
            
            objects = {}
            
            
            for num, frame in enumerate(detections):
                num = num + 1 
                
                # save path to frame
                path = os.path.join(image_dir,"MVI_" + name,'img' + str(num).zfill(5) + '.jpg')
                
                for item in frame:
                    id = item['id']
                    bbox = item['bbox']
                    
                    if False:
                    # shift to xysr
                        new_bbox = np.zeros(4)
                        new_bbox[0] = (bbox[2] + bbox[0])/2.0
                        new_bbox[1] = (bbox[3] + bbox[1])/2.0
                        new_bbox[2] = (bbox[2] - bbox[0])
                        new_bbox[3] = (bbox[3] - bbox[1])/new_bbox[2]
                        bbox = new_bbox
                    
                    try:
                        self.frame_objs[path].append(bbox)
                    except:
                        self.frame_objs[path] = [bbox]
                    
                    if id in objects.keys():

                        objects[id]['box'].append(bbox)
                        objects[id]["im"].append(path)
                    else:
                        objects[id] = {}
                        objects[id]['box'] = [bbox]
                        objects[id]['im'] = [path]
                        
            # get rid of object ids and just keep list of bboxes, important to do this every track because ids repeat
            for id in objects:
                if len(objects[id]['box']) > self.n:
                    label_list.append(np.array(objects[id]['box']))
                    im_list.append(objects[id]['im'])
               
        self.label_list = label_list
        self.im_list = im_list
        
    
        # parse_labels returns a list (one frame per index) of lists, where 
        # each item in the sublist is one object
        # so we need to go through and keep a running record of all objects, indexed by id
            
        with_speed = []
        for bboxes in self.label_list:     
            speeds = np.zeros(bboxes.shape)  
            speeds[:len(speeds)-1,:] = bboxes[1:,:] - bboxes[:len(bboxes)-1,:]
            speeds[-1,:] = speeds[-2,:]

            try:
                speeds = savgol_filter(speeds,5,2,axis = 0)
            except:
                print(speeds.shape)
                print(bboxes.shape)
            #plt.plot(speeds[:,0])
            #plt.legend(["Unsmoothed","Smoothed"])
            #plt.show()
            combined = np.concatenate((bboxes,speeds),axis = 1)
            with_speed.append(combined)
        self.label_list = with_speed
        
        


    def __len__(self):
        """ returns total number of frames in all tracks"""
        return len (self.label_list)

    def __getitem__(self, index):
        
        data = self.label_list[index]
        
        # if track is too short, just use the next index instead
        while len(data) <= self.n:
            index = (index + 1) % len(self.label_list)
            data = self.label_list[index]
        
        start = np.random.randint(0,len(data)-self.n)
        data = data[start:start+self.n,:]
        
        ims = self.im_list[index]
        ims = ims[start:start+self.n]
        
        return data, ims
        
    def parse_labels(self,label_file):
        """
        Returns a set of metadata (1 per track) and a list of labels (1 item per
        frame, where an item is a list of dictionaries (one dictionary per object
        with fields id, class, truncation, orientation, and bbox
        """
        
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
            12:'Truck-Flatbed'
            }
        
        
        tree = ET.parse(label_file)
        root = tree.getroot()
        
        # get sequence attributes
        seq_name = root.attrib['name']
        
        # get list of all frame elements
        #frames = root.getchildren()
        frames = list(root)
        # first child is sequence attributes
        seq_attrs = frames[0].attrib
        
        # second child is ignored regions
        ignored_regions = []
        for region in frames[1]:
            coords = region.attrib
            box = np.array([float(coords['left']),
                            float(coords['top']),
                            float(coords['left']) + float(coords['width']),
                            float(coords['top'])  + float(coords['height'])])
            ignored_regions.append(box)
        frames = frames[2:]
        
        # rest are bboxes
        all_boxes = []
        frame_counter = 1
        for frame in frames:
            while frame_counter < int(frame.attrib['num']):
                # this means that there were some frames with no detections
                all_boxes.append([])
                frame_counter += 1
            
            frame_counter += 1
            frame_boxes = []
            #boxids = frame.getchildren()[0].getchildren()
            boxids = list(list(frame)[0])
            for boxid in boxids:
                #data = boxid.getchildren()
                data = list(boxid)
                coords = data[0].attrib
                stats = data[1].attrib
                bbox = np.array([float(coords['left']),
                                float(coords['top']),
                                float(coords['left']) + float(coords['width']),
                                float(coords['top'])  + float(coords['height'])])
                det_dict = {
                        'id':int(boxid.attrib['id']),
                        'class':stats['vehicle_type'],
                        'class_num':class_dict[stats['vehicle_type']],
                        'color':stats['color'],
                        'orientation':float(stats['orientation']),
                        'truncation':float(stats['truncation_ratio']),
                        'bbox':bbox
                        }
                
                frame_boxes.append(det_dict)
            all_boxes.append(frame_boxes)
        
        sequence_metadata = {
                'sequence':seq_name,
                'seq_attributes':seq_attrs,
                'ignored_regions':ignored_regions
                }
        return all_boxes, sequence_metadata
        

if __name__ == "__main__":
    #### Test script here
    try:
        test
    except:
        try:
            label_dir = "C:\\Users\\derek\\Desktop\\UA Detrac\\DETRAC-Train-Annotations-XML-v3"
            test = Track_Dataset(label_dir)
    
        except:
            image_dir = "/home/worklab/Desktop/detrac/DETRAC-all-data"
            label_dir = "/home/worklab/Desktop/detrac/DETRAC-Train-Annotations-XML-v3"
            test = Track_Dataset(image_dir,label_dir)
    idx = np.random.randint(0,len(test))
    
    cv2.destroyAllWindows()