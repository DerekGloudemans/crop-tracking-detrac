#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:33:55 2020

@author: worklab
"""
import _pickle as pickle
import os
import numpy as np

directory = "/home/worklab/Documents/code/detection-by-tracking/results_no_mask"
identifier = 3 # det step or speed threshold

for file in os.listdir(directory):
    sequence = file.split("_")[1]
    det_step = int(file.split("_")[2].split(".cpkl")[0] )
    
    if det_step != identifier:
     continue
 
    else:
        # parse detections to create n_frames x n_objs x 4 array where each row is one frame and each column is one object
        # x,y,w,h
        
        with open(os.path.join(directory,file),'rb') as f:
            positions,metrics = pickle.load(f)
        
        max_id = -1
        for frame in positions:
            for obj in frame:
                if obj["id"] > max_id:
                    max_id = obj["id"]
                    
        # positions is a list with one entry per frame
        # each entry is a list with one entry per object (objects in dict format)
        
        # create output np array
        output = np.zeros([len(positions),max_id+1, 4])
        
        
        # iterate through positions
        for i in range(len(positions)):
            for obj in positions[i]:
                id = obj["id"]
                bbox = obj["bbox"]
                x = (bbox[2] + bbox[0])/2
                y = (bbox[1] + bbox[3])/2
                w =  bbox[2] - bbox[0]
                h =  bbox[3] = bbox[1]
                
                output[i,id,0] = x
                output[i,id,1] = y
                output[i,id,2] = w
                output[i,id,3] = h
        
        
        x_file = "detrac_format/MVI_{}_LX.txt".format(sequence)
        y_file = "detrac_format/MVI_{}_LY.txt".format(sequence)
        w_file = "detrac_format/MVI_{}_W.txt".format(sequence)
        h_file = "detrac_format/MVI_{}_H.txt".format(sequence)
        s_file = "detrac_format/MVI_{}_speed.txt".format(sequence)
        # write x file
        with open(x_file,"w") as f:
            # write one line
            for i in range(len(output)):
                
                # write one value
                for j in range(len(output[0])):
                    
                    f.write(str(round(output[i,j,0],3)))
                    if j < max_id:
                        f.write(" , ")
                if i < len(output) - 1:
                    f.write('\n')
        # write y file
        with open(y_file,"w") as f:
            # write one line
            for i in range(len(output)):
                
                # write one value
                for j in range(len(output[0])):
                    
                    f.write(str(round(output[i,j,1],3)))
                    if j < max_id:
                        f.write(" , ")
                if i < len(output) - 1:
                    f.write('\n')
        
        # write w file
        with open(w_file,"w") as f:
            # write one line
            for i in range(len(output)):
                
                # write one value
                for j in range(len(output[0])):
                    
                    f.write(str(round(output[i,j,2],3)))
                    if j < max_id:
                        f.write(" , ")
                if i < len(output) - 1:
                    f.write('\n')
        
        # write h file        
        with open(h_file,"w") as f:
            # write one line
            for i in range(len(output)):
                
                # write one value
                for j in range(len(output[0])):
                    
                    f.write(str(round(output[i,j,3],3)))
                    if j < max_id:
                        f.write(" , ")
                if i < len(output) - 1:
                    f.write('\n')
                    
        # write speed file
        with open(h_file,"w") as f:
            f.write(str(round(metrics["framerate"][0],3)))