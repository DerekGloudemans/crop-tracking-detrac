#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:20:41 2020

@author: worklab
"""

import torch
import numpy as np
import random
import _pickle as pickle
import matplotlib.pyplot as plt
import cv2 
import os,sys,inspect
import time

#from detrac_files.detrac_tracking_dataset import Track_Dataset

from torch.utils.data import DataLoader



from PIL import Image
from torchvision.transforms import functional as F
from torchvision.ops import roi_align


# add relevant packages and directories to path
detector_path = os.path.join(os.getcwd(),"models","pytorch_retinanet_detector")
sys.path.insert(0,detector_path)
localizer_path = os.path.join(os.getcwd(),"models","pytorch_retinanet_localizer")
sys.path.insert(0,localizer_path)
detrac_util_path = os.path.join(os.getcwd(),"util_detrac")
sys.path.insert(0,detrac_util_path)


# data
from util_detrac.detrac_tracking_dataset import Track_Dataset
from util_detrac.detrac_detection_dataset import Detection_Dataset,collate

# filter and CNNs
from util_track.kf import Torch_KF
from models.pytorch_retinanet_detector.retinanet.model import resnet50 
from models.pytorch_retinanet_localizer.retinanet.model import resnet34




plt.style.use("seaborn")
random.seed  = 0

def test_outputs(bboxes,crops):
    """
    Description
    -----------
    Generates a plot of the bounding box predictions output by the localizer so
    performance of this component can be visualized
    
    Parameters
    ----------
    bboxes - tensor [n,4] 
        bounding boxes output for each crop by localizer network
    crops - tensor [n,3,width,height] (here width and height are both 224)
    """
    
    # define figure subplot grid
    batch_size = len(crops)
    row_size = min(batch_size,8)
    
    for i in range(0,len(crops)):    
        # get image
        im   = crops[i].data.cpu().numpy().transpose((1,2,0))
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        im   = std * im + mean
        im   = np.clip(im, 0, 1)
        
        # get predictions
        bbox = bboxes[i].data.cpu().numpy()
        
        wer = 3
        imsize = 224
        
        # transform bbox coords back into im pixel coords
        bbox = (bbox* imsize*wer - imsize*(wer-1)/2).astype(int)
        # plot pred bbox
        im = cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0.1,0.6,0.9),2)
        im = im.get()

        plt.imshow(im)
        plt.pause(1)
        
        
def iou(a,b):
    """
    Description
    -----------
    Calculates intersection over union for all sets of boxes in a and b

    Parameters
    ----------
    a : a torch of size [batch_size,4] of bounding boxes.
    b : a torch of size [batch_size,4] of bounding boxes.

    Returns
    -------
    mean_iou - float between [0,1] with average iou for a and b
    """

    area_a = a[:,2] * a[:,2] * a[:,3]
    area_b = b[:,2] * b[:,2] * b[:,3]
    
    minx = torch.max(a[:,0]-a[:,2]/2, b[:,0]-b[:,2]/2)
    maxx = torch.min(a[:,0]+a[:,2]/2, b[:,0]+b[:,2]/2)
    miny = torch.max(a[:,1]-a[:,2]*a[:,3]/2, b[:,1]-b[:,2]*b[:,3]/2)
    maxy = torch.min(a[:,1]+a[:,2]*a[:,3]/2, b[:,1]+b[:,2]*b[:,3]/2)
    zeros = torch.zeros(minx.shape,dtype=float)
    
    intersection = torch.max(zeros, maxx-minx) * torch.max(zeros,maxy-miny)
    union = area_a + area_b - intersection
    iou = torch.div(intersection,union)
    mean_iou = torch.mean(iou)
    
    return mean_iou

def md_iou(a,b):
    """
    a,b - [batch_size x num_anchors x 4]
    """
    
    area_a = (a[:,:,2]-a[:,:,0]) * (a[:,:,3]-a[:,:,1])
    area_b = (b[:,:,2]-b[:,:,0]) * (b[:,:,3]-b[:,:,1])
    
    minx = torch.max(a[:,:,0], b[:,:,0])
    maxx = torch.min(a[:,:,2], b[:,:,2])
    miny = torch.max(a[:,:,1], b[:,:,1])
    maxy = torch.min(a[:,:,3], b[:,:,3])
    zeros = torch.zeros(minx.shape,dtype=float)
    
    intersection = torch.max(zeros, maxx-minx) * torch.max(zeros,maxy-miny)
    union = area_a + area_b - intersection
    iou = torch.div(intersection,union)
    
    #print("MD iou: {}".format(iou.max(dim = 1)[0].mean()))
    return iou

def plot_states(ap_states,
                ap_covs,
                loc_meas,
                apst_states,
                apst_covs,
                gts,
                save_num = None
                ):
    
   
    
    #convert list into numpy array
    ap_states =   torch.stack(ap_states)
    ap_covs =     torch.stack(ap_covs)
    loc_meas =    torch.stack(loc_meas)
    apst_states = torch.stack(apst_states)
    apst_covs =   torch.stack(apst_covs)
    gts =         torch.stack(gts)
    titles = ["X min", "Y min", "X max", "Y max", "X min dot", "Y min dot", "S min dot", "R min dot"]
    
    # format covariances - want each 
    covs = torch.empty(ap_covs.shape[0]+apst_covs.shape[0],ap_covs.shape[1])
    covs[1::2,:]  = ap_covs
    covs[::2,:] = apst_covs
    means = torch.empty(ap_states.shape[0]+apst_states.shape[0],ap_states.shape[1])
    means[1::2,:] = ap_states
    means[::2,:] = apst_states
    
    xvals = [i/2 for i in range(len(covs))]
    ap_xvals = [i for i in range(1,len(ap_states)+1)]
    
    fig, axs = plt.subplots(int(len(gts[0]))//2,2,constrained_layout=True,figsize = (20,20))
    
    for i in range(len(ap_states[0])):
        legend = ["apriori","aposteriori","true","covariance"]
        
        # plot apriori state
        axs[i//2,i%2].plot(ap_xvals,ap_states[:,i],"--",linewidth = 3, color = (0.3,0.2,0.5))
        
        # plot measurement
        if i in range(len(loc_meas[0])):
            axs[i//2,i%2].plot(ap_xvals,loc_meas[:,i],".", markersize=15,color = (0.3,0.2,0.5))
            legend = ["apriori","measurement","aposteriori","true","covariance"]
        
        # if i == 6:
        #     axs[i//2,i%2].set_ylim([-30,30])
            
        # if i == 3:
        #    axs[i//2,i%2].set_ylim([0,2]) 
        
        # if i == 7:
        #      axs[i//2,i%2].set_ylim([-0.3,0.3]) 
             
        # plot aposteriori state
        axs[i//2,i%2].plot(apst_states[:,i],"-", color = (0.4,0.1,0.5))
        
        # plot ground truth
        axs[i//2,i%2].plot(gts[:,i],"-", color = (0.2,0.7,0.3))
        
        # plot covariance
        covs = covs.sqrt()
        axs[i//2,i%2].fill_between(xvals,means[:,i]-covs[:,i],covs[:,i]+means[:,i],color = (0.4,0.1,0.5,0.2))
        axs[i//2,i%2].fill_between(xvals,means[:,i]-3*covs[:,i],3*covs[:,i]+means[:,i],color = (0.4,0.1,0.5,0.05))
        

        # set plot settings
        axs[i//2,i%2].legend(legend)
        axs[i//2,i%2].set_title(titles[i])
        # axs[i//2,i%2].set_xlim([0,len(loc_meas)])
        
    plt.pause(3)
    if save_num is not None:
        plt.savefig("states_{}.png".format(save_num))
        

def fit_Q(loader,kf_params, n_iterations = 20000, save_file = "temp.cpkl", speed_init = "smooth",state_size = 8):
    """
    Fits model error covariance matrix for KF using one-step prediction rollout
    
    Parameters
    ----------
    dataloader : generator that loads [batch_size,n_frames,state_size] tensors
    kf_params : dictionary used to initialize Torch_KF
    n_iterations : (int) number of iterations for Q estimation, default is 20000
    speed_init : (string) "zero" or "smooth" specifies initialization of speed in loaded data
    state_size : (int) number of state variables to use, rest are truncated from incoming states
    """
    
    error_vectors = []
    scores = []
    for iteration in range(n_iterations):
        # grab batch
        batch, ims = next(iter(loader))
        
        # initialize tracker
        tracker = Torch_KF("cpu",INIT = kf_params)
    
        obj_ids = [i for i in range(len(batch))]
        
        # don't want to always use first frame in track to initialize, so randomly pick an index
        first = np.random.randint(0,batch.shape[1]-2)
        
        if speed_init == "smooth":
            tracker.add(batch[:,first,:state_size],obj_ids)
        else:
            tracker.add(batch[:,first,:4],obj_ids)
        # roll out a frame
        tracker.predict()
        
        # get predicted object locations
        objs = tracker.objs()
        objs = [objs[key] for key in objs]
        pred = torch.from_numpy(np.array(objs)).double()
        
        # get ground truths
        gt = batch[:,first+1,:state_size]
        error = gt - pred
        error_vectors.append(error)
        
        # get ious
        scores.append(iou(gt,pred))
        
        print("Finished iteration {}".format(iteration))
        
    # summary metrics    
    error_vectors = torch.cat(error_vectors,dim = 0)
    mean = torch.mean(error_vectors, dim = 0)
    
    covariance = torch.zeros((state_size,state_size))
    for vec in error_vectors:
        covariance += torch.mm((vec - mean).unsqueeze(1), (vec-mean).unsqueeze(1).transpose(0,1))
    
    covariance = covariance / error_vectors.shape[0]
    kf_params["mu_Q"] = mean
    kf_params["Q"] = covariance
    
    with open(save_file,"wb") as f:
          pickle.dump(kf_params,f)
    
    print("---------- Model 1-step errors ----------")
    print("Average 1-step IOU: {}".format(sum(scores)/len(scores)))
    print("Mean 1-step state error: {}".format(mean))
    print("1-step covariance: {}".format(covariance))

    ################# end fit_Q function definition ##########################

def fit_localizer_R(loader,
                    kf_params,
                    device,
                    localizer,
                    n_iterations = 500,
                    bers = [1.5],
                    skew_ratio = 1, 
                    save_file = "temp.cpkl",
                    wer = 1.25):
    
    # save best across all ber values
    best_covariance = None
    best_mean = None
    best_score = 0
    best_ber = 0
    
    for ber in bers:
        print("Computing covariance matrix for ber = {}".format(ber))
        skewed_iou = []
        localizer_iou = []
        meas_errors = []
        
        start = time.time()
        for iteration in range(n_iterations):
            batch,ims = next(iter(loader))
            frame_idx = 0
            gt = batch[:,frame_idx,:4]
            b = batch.shape[0]
            
            # get starting error
            degradation = np.array([2,2,4,0.01]) *skew_ratio # should roughly equal localizer error covariance
            degradation = np.array([10,10,10,10])
            skew = np.random.normal(0,degradation,(len(batch),4))
            gt_skew = gt + skew
            skewed_iou.append(iou(gt_skew,gt))
            
            
            with torch.no_grad():
                # ims are collated by frame,then batch index
                relevant_ims = ims[frame_idx]
                frames =[]
                for idx,item in enumerate(relevant_ims):
                    with Image.open(item) as im:
                           im = F.to_tensor(im)
                           frame = F.normalize(im,mean=[0.3721, 0.3880, 0.3763],
                                 std=[0.0555, 0.0584, 0.0658])
                           #correct smaller frames
                           # if frame.shape[1] < 375:
                           #    new_frame = torch.zeros([3,375,frame.shape[2]])
                           #    new_frame[:,:frame.shape[1],:] = frame
                           #    frame = new_frame
                           # if frame.shape[2] < 1242:
                           #    new_frame = torch.zeros([3,375,1242])
                           #    new_frame[:,:,:frame.shape[2]] = frame
                           #    frame = new_frame   
                           
                           MASK = False
                           if MASK:
                               
                               other_objs = dataset.frame_objs[item]                          
                               # create copy of frame
                               frame_copy = frame.clone()                           
                               # mask each other object in frame
                               for obj in other_objs:
                                   xmin = (obj[0] - obj[2] / 2.0).astype(int)
                                   ymin = (obj[1] - obj[2]*obj[3] / 2.0).astype(int)
                                   xmax = (obj[0] + obj[2] / 2.0).astype(int)
                                   ymax = (obj[1] + obj[2]*obj[3] / 2.0).astype(int)       
                                   
                                   region = obj
                                   shape = frame[:,ymin:ymax,xmin:xmax].shape
                                   r =  torch.normal(0.485,0.229,[shape[1],shape[2]])
                                   g =  torch.normal(0.456,0.224,[shape[1],shape[2]])
                                   b =  torch.normal(0.406,0.225,[shape[1],shape[2]])
                                   rgb = torch.stack([r,g,b])
                                   frame[:,ymin:ymax,xmin:xmax] = rgb
                                   
                               # restore gt_skew pixels
                               o = gt_skew[idx]
                               xmin = (o[0] - o[2] / 2.0).int()
                               ymin = (o[1] - o[2]*obj[3] / 2.0).int()
                               xmax = (o[0] + o[2] / 2.0).int()
                               ymax = (o[1] + o[2]*obj[3] / 2.0).int()
                               frame[:,ymin:ymax,xmin:xmax] = frame_copy[:,ymin:ymax,xmin:xmax]
                               #plt.imshow(frame.transpose(2,0).transpose(0,1))
                               #plt.pause(5)
                               
                           frames.append(frame)
                frames = torch.stack(frames).to(device)
                
                # crop image
                boxes = gt_skew.clone()
                
                #convert xyxy into xysr
                temp = boxes.clone()
                temp[:,0] = (boxes[:,0] + boxes[:,2])/2.0
                temp[:,1] = (boxes[:,1] + boxes[:,3])/2.0
                temp[:,2] =  boxes[:,2] - boxes[:,0]
                temp[:,3] = (boxes[:,3] - boxes[:,1])/temp[:,2]
                boxes = temp
            
                
                # convert xysr boxes into xmin xmax ymin ymax
                # first row of zeros is batch index (batch is size 0) for ROI align
                new_boxes = np.zeros([len(boxes),5]) 
        
                # use either s or s x r for both dimensions, whichever is larger,so crop is square
                #box_scales = np.max(np.stack((boxes[:,2],boxes[:,2]*boxes[:,3]),axis = 1),axis = 1)
                box_scales = np.min(np.stack((boxes[:,2],boxes[:,2]*boxes[:,3]),axis = 1),axis = 1) #/2.0
                    
                #expand box slightly
                box_scales = box_scales * ber# box expansion ratio
                
                new_boxes[:,1] = boxes[:,0] - box_scales/2
                new_boxes[:,3] = boxes[:,0] + box_scales/2 
                new_boxes[:,2] = boxes[:,1] - box_scales/2 
                new_boxes[:,4] = boxes[:,1] + box_scales/2 
                for i in range(len(new_boxes)):
                    new_boxes[i,0] = i # set image index for each
                    
                torch_boxes = torch.from_numpy(new_boxes).float().to(device)
                
                # crop using roi align
                crops = roi_align(frames,torch_boxes,(224,224))
                
                # batch_idx,batch,batch_confs,batch_class = localizer(crops,LOCALIZE = True)
                
                # # gather max confidence bounding boxes for each index
                # detections = torch.zeros([b,4])
                # for i in range(0,b):
                #     where = (batch_idx == i).nonzero()[0]
                #     detections[i,:] = batch[where,:]
                
                reg_boxes, classes = localizer(crops,LOCALIZE = True)
                torch.cuda.synchronize()
                reg_boxes = reg_boxes.data.cpu()
                #confs = confs.data.cpu()
                classes = classes.data.cpu()
                confs,_ = torch.max(classes, dim = 2)
                
                # use original bboxes to weight best bboxes 
                n_anchors = reg_boxes.shape[1]
                a_priori = gt.clone()
                gt_skew = gt.clone()                
                a_priori[:,0] = (gt_skew[:,0] - gt_skew[:,2]/2.0)               - new_boxes[:,1]
                a_priori[:,1] = (gt_skew[:,1] - gt_skew[:,2]*gt_skew[:,3]/2.0 ) - new_boxes[:,2]
                a_priori[:,2] = (gt_skew[:,0] + gt_skew[:,2]/2.0 )              - new_boxes[:,1]
                a_priori[:,3] = (gt_skew[:,1] + gt_skew[:,2]*gt_skew[:,3]/2.0 ) - new_boxes[:,2]
                bs = torch.from_numpy(box_scales).unsqueeze(1).repeat(1,4)
                a_priori = a_priori * 224/bs
                a_priori = a_priori.unsqueeze(1).repeat(1,n_anchors,1)
                
                #reg_boxes = reg_boxes.unsqueeze(0).repeat(b,1,1)
                
                # evaluate each box based on xy similarity
                # x_diff = torch.abs(a_priori[:,0] + a_priori[:,2] - (reg_boxes[:,0] + reg_boxes[:,2]) )
                # y_diff = torch.abs(a_priori[:,1] + a_priori[:,3] - (reg_boxes[:,1] + reg_boxes[:,3]) )
                # # evaluate each box on width and ratio similarity
                # w_diff = torch.abs(a_priori[:,2] - a_priori[:,0] - (reg_boxes[:,2] - reg_boxes[:,0]) )
                # h_diff = torch.abs(a_priori[:,3] - a_priori[:,1] - (reg_boxes[:,3] - reg_boxes[:,1]) )
                
                keep_num = 5
                
                # evaluate each box on conf
                alpha = 0.6
                beta  = 0 
                gamma = 0
                delta = 1
                iou_score = md_iou(a_priori.double(),reg_boxes.double())
                score = alpha*confs + delta * iou_score
                
                _,sorted_idxs = torch.sort(score,dim = 1)
                #keep_num = 4
                best5 = sorted_idxs[:,-keep_num:]

                det_list = []
                for k in range(b):
                    det5 = reg_boxes[k,best5[k],:]
                    avg = det5.mean(dim = 0)
                    det_list.append(avg)
                detections = torch.stack(det_list)
        
                # 5b. convert to global image coordinates 
                    
                # these detections are relative to crops - convert to global image coords
                
                # detections = (reg_out* 224*wer - 224*(wer-1)/2)
                # detections = detections.data.cpu()
                
                # plot outputs
                if False and iteration % 100 == 0:
                    batch_size = 32
                    row_size = 8
                    fig, axs = plt.subplots((batch_size+row_size-1)//row_size, row_size, constrained_layout=True)
                
                    for i in range(0,batch_size):
                        
                        # get image
                        im   = crops[i].cpu().numpy().transpose((1,2,0))
                        mean = np.array([0.485, 0.456, 0.406])
                        std  = np.array([0.229, 0.224, 0.225])
                        im   = std * im + mean
                        im   = np.clip(im, 0, 1)
                        
                        bbox = detections[i]
                        
                        
                        imsize = 224
                        
                        # convert xysr to xyxy
                        reg_true = torch.zeros([len(gt),4])
                        reg_true[:,0] =  gt[:,0] - gt[:,2]/2.0
                        reg_true[:,1] =  gt[:,1] - gt[:,2]*gt[:,3]/2.0
                        reg_true[:,2] =  gt[:,0] + gt[:,2]/2.0
                        reg_true[:,3] =  gt[:,1] + gt[:,2]*gt[:,3]/2.0
                        
                        reg_true[:,0] = reg_true[:,0] - new_boxes[:,1]
                        reg_true[:,1] = reg_true[:,1] - new_boxes[:,2]
                        reg_true[:,2] = reg_true[:,2] - new_boxes[:,1]
                        reg_true[:,3] = reg_true[:,3] - new_boxes[:,2]
                        reg = reg_true[i] * 224/box_scales[i]
                        
                        # plot pred bbox
                        im = cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0.9,0.2,0.2),2)
                       
                        # plot ground truth bbox
                        im = cv2.rectangle(im,(reg[0],reg[1]),(reg[2],reg[3]),(0.0,0.8,0.0),2)
                
                        im = im.get()
                        
                        axs[i//row_size,i%row_size].imshow(im)
                        axs[i//row_size,i%row_size].set_xticks([])
                        axs[i//row_size,i%row_size].set_yticks([])
                        plt.pause(.001)    
                    plt.close()
                    
                    
                # # add in original box offsets and scale outputs by original box scales
                detections[:,0] = detections[:,0]*box_scales/224 + new_boxes[:,1]
                detections[:,2] = detections[:,2]*box_scales/224 + new_boxes[:,1]
                detections[:,1] = detections[:,1]*box_scales/224 + new_boxes[:,2]
                detections[:,3] = detections[:,3]*box_scales/224 + new_boxes[:,2]
        
                # # convert into xysr form 
                output = np.zeros([len(detections),4])
                output[:,0] = (detections[:,0] + detections[:,2]) / 2.0
                output[:,1] = (detections[:,1] + detections[:,3]) / 2.0
                output[:,2] = (detections[:,2] - detections[:,0])
                output[:,3] = (detections[:,3] - detections[:,1]) / output[:,2]
                pred = torch.from_numpy(output)
                
                pred = detections
    
                # evaluate localizer
                localizer_iou.append(iou(pred.double(),gt.double()))
                #print("Eval IOU: {}".format(localizer_iou[-1]))
                error = (gt[:,:4]-pred)
                meas_errors.append(error)
            
            if iteration % 100 == 0:
                print("Finished iteration {}".format(iteration))
            
        meas_errors = torch.stack(meas_errors)
        meas_errors = meas_errors.view(-1,4)
        mean = torch.mean(meas_errors, dim = 0)    
        covariance = torch.zeros((4,4))
        for vec in meas_errors:
            covariance += torch.mm((vec - mean).unsqueeze(1), (vec-mean).unsqueeze(1).transpose(0,1))
            
        covariance = covariance / meas_errors.shape[0]
        
        score = sum(localizer_iou)/len(localizer_iou)
        print("---------- Localizer 1-step errors with ber {}----------".format(ber))
        print("Average starting IOU: {}".format(sum(skewed_iou)/len(skewed_iou)))
        print("Average 1-step IOU: {}".format(score))
        print("Mean 1-step state error: {}".format(mean))
        print("1-step covariance: {}".format(covariance))
        
        total = time.time() - start
        print("{} crops processed in {} sec ({} cps)".format(len(localizer_iou),total,len(localizer_iou)/total))
        
        if score > best_score:
            best_score = score# save best across all ber values
            best_covariance = covariance
            best_mean = mean
            best_ber = ber
       
    print("---------- Localizer 1-step errors with best ber ({})----------".format(best_ber))
    print("Average starting IOU: {}".format(sum(skewed_iou)/len(skewed_iou)))
    print("Average 1-step IOU: {}".format(best_score))
    print("Mean 1-step state error: {}".format(best_mean))
    print("1-step covariance: {}".format(best_covariance))
   
    kf_params["mu_R"] = best_mean
    kf_params["R"] = best_covariance
    
    with open(save_file,"wb") as f:
        pickle.dump(kf_params,f)
            
    return meas_errors


def fit_detector_R(loader,
                   kf_params, 
                   device, 
                   detector,
                   save_file = "temp.cpkl",
                   n_iterations = 2000):
    errors = []
    ious = []
    
    for iteration in range(n_iterations):
        print("On iteration {} of {}".format(iteration,n_iterations))
        
        ims, gts,ignored = next(iter(loader))
        ims = ims.to(device)
        
        gts = gts.squeeze(0)
        
        with torch.no_grad():   
            scores,labels,boxes = detector(ims)
           
            
            #match output to ground truth detections
            if len(boxes) == 0 or len(gts) == 0:
                continue
            
            del scores,labels
            boxes = boxes.cpu()
            torch.cuda.empty_cache()
            
            # pred = torch.zeros(boxes.shape)
            # pred[:,0] = (boxes[:,0] + boxes[:,2]) / 2.0
            # pred[:,1] = (boxes[:,1] + boxes[:,3]) / 2.0
            # pred[:,2] = boxes[:,2] - boxes[:,0] 
            # pred[:,3] = (boxes[:,3] - boxes[:,1]) / (boxes[:,2] - boxes[:,0])
            
            # gtxysr = torch.zeros(gts.shape)
            # gtxysr[:,0] = (gts[:,0] + gts[:,2]) / 2.0
            # gtxysr[:,1] = (gts[:,1] + gts[:,3]) / 2.0
            # gtxysr[:,2] = gts[:,2] - gts[:,0]
            # gtxysr[:,3] = (gts[:,3] - gts[:,1]) / (gts[:,2] - gts[:,0])
            # gts = gtxysr
            pred = boxes
            
            idxs = torch.zeros(len(gts)) # idx item 0 holds the pred index that best matches the 0th ground truth, etc.
            keepers = []
            # get iou for each 
            for i, obj in enumerate(gts):
                max_iou = 0 
                for j,pred_obj in enumerate(pred):
                    iou_score = iou(pred_obj.unsqueeze(0).double(),obj.unsqueeze(0).double())
                    if iou_score > max_iou:
                        idxs[i] = j
                        max_iou = iou_score
                        
                if max_iou < 0.25: #object was missed so don't count it
                    pass
                else:
                    keepers.append(i)
                    ious.append(max_iou)
            matched_preds = pred[idxs.long(),:]

            matched_preds = matched_preds[keepers]
            gts = gts[keepers]
            
            error = gts[:,:4] - matched_preds
            for row in error:
                errors.append(row)
        
            del gts,boxes,pred,idxs,matched_preds
        torch.cuda.empty_cache()
        
    errors = torch.stack(errors)
    errors = errors.view(-1,4)
    mean = torch.mean(errors, dim = 0)    
    covariance = torch.zeros((4,4))
    for vec in errors:
        covariance += torch.mm((vec - mean).unsqueeze(1), (vec-mean).unsqueeze(1).transpose(0,1))
    covariance = covariance / errors.shape[0]
    
    score = sum(ious)/len(ious)
    
    print("Detector average IOU: {}" .format(score))
    print("Mean error: {}".format(mean))
    print(" covariance: {}".format(covariance))
    
    kf_params["R2"] = covariance
    kf_params["mu_R2"] = mean
    
    with open(save_file,"wb") as f:
        pickle.dump(kf_params,f)
    
def filter_rollouts(loader,
                    kf_params,
                    localizer,
                    device,
                    n_iterations = 100,
                    ber = 2.0, 
                    skew_ratio = 0,
                    PLOT = True,
                    speed_init = "none",
                    state_size = 7,
                    keep_nums = [1],
                    wer = 1.25
                    ):
    
     for keep_num in keep_nums:
         ap_errors = []
         skewed_iou = []        # how far off each skewed measurement is during init
         starting_iou = []     # after initialization, how far off are we
         a_priori_iou = {}      # after prediction step, how far off is the state
         localizer_iou = {}     # how far off is the localizer
         a_posteriori_iou = {}  # after updating, how far off is the state
         
         for i in range(n_pre,n_pre+n_post):
             a_priori_iou[i] = []
             localizer_iou[i] = []
             a_posteriori_iou[i] = []
         
         model_errors = []
         meas_errors = []
         
         degradation = np.array([2,2,4,0.01]) *skew_ratio # should roughly equal localizer error covariance
         
         for iteration in range(n_iterations):
             
             batch,ims = next(iter(loader))
             
             b = batch.shape[0]
             
             # initialize tracker
             tracker = Torch_KF("cpu",INIT = kf_params, ADD_MEAN_Q = True, ADD_MEAN_R = False)
         
             obj_ids = [i for i in range(len(batch))]
             
             if speed_init == "smooth":
                 tracker.add(batch[:,0,:state_size],obj_ids)
             else:
                 tracker.add(batch[:,0,:4],obj_ids)
             
             # initialize storage
             ap_states = []
             ap_covs = []
             loc_meas = []
             apst_states = []
             apst_covs = []
             gts = []
             
             apst_states.append(tracker.X[0].clone())
             apst_covs.append(torch.diag(tracker.P[0]).clone())
             gts.append(batch[0,n_pre-1,:].clone())
             
             # initialize tracker
             for frame in range(1,n_pre):
                 tracker.predict()
                 
                 # here, rather than updating with ground truth we degrade ground truth by some amount
                 measurement = batch[:,frame,:4]
                 skew = np.random.normal(0,degradation,(len(batch),4))
                 measurement_skewed = measurement + skew
                 
                 skewed_iou.append(iou(measurement,measurement_skewed))
                 tracker.update2(measurement_skewed,obj_ids)
                 
             
             # cumulative error so far
             objs = tracker.objs()
             objs = [objs[key] for key in objs]
             starting = torch.from_numpy(np.array(objs)).double()
             starting_iou.append(iou(starting,batch[:,n_pre-1,:]))
             
             
             for frame_idx in range(n_pre,n_pre + n_post):
                 gt = batch[:,frame_idx,:]
         
                 # get a priori error
                 tracker.predict()
                 objs = tracker.objs()
                 objs = [objs[key] for key in objs]
                 a_priori = torch.from_numpy(np.array(objs)).double()
                 a_priori_iou[frame_idx].append(iou(a_priori,gt))
             
                 if frame_idx in [1,2,3,4]:
                     ap_error = a_priori[:,:4] - gt[:,:4]
                     ap_errors.append(ap_error)
                
                 ap_states.append(tracker.X[0].clone())
                 ap_covs.append(torch.diag(tracker.P[0]).clone())
    
                 # at this point, we have gt, the correct bounding boxes for this frame, 
                 # and the tracker states, the estimate of the state for this frame
                 # expand state estimate and get localizer prediction
                 # shift back into global coordinates
                 
                 # ims are collated by frame,then batch index
                 relevant_ims = ims[frame_idx]
                 frames =[]
                 for item in relevant_ims:
                     with Image.open(item) as im:
                            im = F.to_tensor(im)
                            frame = F.normalize(im,mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
                            # #correct smaller frames
                            # if frame.shape[1] < 540:
                            #    new_frame = torch.zeros([3,375,frame.shape[2]])
                            #    new_frame[:,:frame.shape[1],:] = frame
                            #    frame = new_frame
                            # if frame.shape[2] < 1242:
                            #    new_frame = torch.zeros([3,375,1242])
                            #    new_frame[:,:,:frame.shape[2]] = frame
                            #    frame = new_frame   
                            MASK = False
                            if MASK:
                               
                               other_objs = dataset.frame_objs[item]                          
                               # create copy of frame
                               frame_copy = frame.clone()                           
                               # mask each other object in frame
                               for obj in other_objs:
                                   xmin = (obj[0] - obj[2] / 2.0).astype(int)
                                   ymin = (obj[1] - obj[2]*obj[3] / 2.0).astype(int)
                                   xmax = (obj[0] + obj[2] / 2.0).astype(int)
                                   ymax = (obj[1] + obj[2]*obj[3] / 2.0).astype(int)       
                                   
                                   region = obj
                                   shape = frame[:,ymin:ymax,xmin:xmax].shape
                                   r =  torch.normal(0.485,0.229,[shape[1],shape[2]])
                                   g =  torch.normal(0.456,0.224,[shape[1],shape[2]])
                                   b =  torch.normal(0.406,0.225,[shape[1],shape[2]])
                                   rgb = torch.stack([r,g,b])
                                   frame[:,ymin:ymax,xmin:xmax] = rgb
                                   
                               # restore gt_skew pixels
                               o = gt_skew[idx]
                               xmin = (o[0] - o[2] / 2.0).int()
                               ymin = (o[1] - o[2]*obj[3] / 2.0).int()
                               xmax = (o[0] + o[2] / 2.0).int()
                               ymax = (o[1] + o[2]*obj[3] / 2.0).int()
                               frame[:,ymin:ymax,xmin:xmax] = frame_copy[:,ymin:ymax,xmin:xmax]
                               #plt.imshow(frame.transpose(2,0).transpose(0,1))
                               #plt.pause(5)
                               
                            frames.append(frame)
                 frames = torch.stack(frames).to(device)
                 
                 # crop image
                 boxes = a_priori
                 
                 #convert xyxy into xysr
                 temp = boxes.clone()
                 temp[:,0] = (boxes[:,0] + boxes[:,2])/2.0
                 temp[:,1] = (boxes[:,1] + boxes[:,3])/2.0
                 temp[:,2] =  boxes[:,2] - boxes[:,0]
                 temp[:,3] = (boxes[:,3] - boxes[:,1])/temp[:,2]
                 boxes = temp
            
                 # convert xysr boxes into xmin xmax ymin ymax
                 # first row of zeros is batch index (batch is size 0) for ROI align
                 new_boxes = np.zeros([len(boxes),5]) 
         
                 # use either s or s x r for both dimensions, whichever is larger,so crop is square
                 #box_scales = np.max(np.stack((boxes[:,2],boxes[:,2]*boxes[:,3]),axis = 1),axis = 1)
                 box_scales = np.min(np.stack((boxes[:,2],boxes[:,2]*boxes[:,3]),axis = 1),axis = 1) #/2.0
                     
                 #expand box slightly
                 #ber = 2.15
                 box_scales = box_scales * ber# box expansion ratio
                 
                 new_boxes[:,1] = boxes[:,0] - box_scales/2
                 new_boxes[:,3] = boxes[:,0] + box_scales/2 
                 new_boxes[:,2] = boxes[:,1] - box_scales/2 
                 new_boxes[:,4] = boxes[:,1] + box_scales/2 
                 for i in range(len(new_boxes)):
                     new_boxes[i,0] = i # set image index for each
                     
                 torch_boxes = torch.from_numpy(new_boxes).float().to(device)
                 
                 # crop using roi align
                 crops = roi_align(frames,torch_boxes,(224,224))
                 
                 # _,reg_out = localizer(crops)
                 # torch.cuda.synchronize()
                 # detections = (reg_out* 224*wer - 224*(wer-1)/2)
                 # detections = detections.data.cpu()
         
                 if True:
                    reg_boxes, classes = localizer(crops,LOCALIZE = True)
                    torch.cuda.synchronize()
                    reg_boxes = reg_boxes.data.cpu()
                    #confs = confs.data.cpu()
                    classes = classes.data.cpu()
                    confs,_ = torch.max(classes, dim = 2)
                    
                    # use original bboxes to weight best bboxes 
                    n_anchors = reg_boxes.shape[1]
                    a_priori = a_priori[:,:4]
                    gt_skew = a_priori.clone() #gt.clone()                
                    a_priori[:,0] = (gt_skew[:,0] - gt_skew[:,2]/2.0)               - new_boxes[:,1]
                    a_priori[:,1] = (gt_skew[:,1] - gt_skew[:,2]*gt_skew[:,3]/2.0 ) - new_boxes[:,2]
                    a_priori[:,2] = (gt_skew[:,0] + gt_skew[:,2]/2.0 )              - new_boxes[:,1]
                    a_priori[:,3] = (gt_skew[:,1] + gt_skew[:,2]*gt_skew[:,3]/2.0 ) - new_boxes[:,2]
                    bs = torch.from_numpy(box_scales).unsqueeze(1).repeat(1,4)
                    a_priori = a_priori * 224/bs
                    a_priori = a_priori.unsqueeze(1).repeat(1,n_anchors,1)
                    
                    #reg_boxes = reg_boxes.unsqueeze(0).repeat(b,1,1)
                    
                    # evaluate each box based on xy similarity
                    # x_diff = torch.abs(a_priori[:,0] + a_priori[:,2] - (reg_boxes[:,0] + reg_boxes[:,2]) )
                    # y_diff = torch.abs(a_priori[:,1] + a_priori[:,3] - (reg_boxes[:,1] + reg_boxes[:,3]) )
                    # # evaluate each box on width and ratio similarity
                    # w_diff = torch.abs(a_priori[:,2] - a_priori[:,0] - (reg_boxes[:,2] - reg_boxes[:,0]) )
                    # h_diff = torch.abs(a_priori[:,3] - a_priori[:,1] - (reg_boxes[:,3] - reg_boxes[:,1]) )
                    
                    
                    # evaluate each box on conf
                    alpha = 0.6
                    beta  = 0 
                    gamma = 0
                    delta = 1
                    iou_score = md_iou(a_priori.double(),reg_boxes.double())
                    score = alpha*confs + delta * iou_score
                    
                    # _,sorted_idxs = torch.sort(score,dim = 1)
                    # #keep_num = 4
                    # best5 = sorted_idxs[:,-keep_num:]
    
                    # det_list = []
                    # for k in range(b):
                    #     det5 = reg_boxes[k,best5[k],:]
                    #     avg = det5.mean(dim = 0)
                    #     det_list.append(avg)
                    # detections = torch.stack(det_list)
                    
                    best_scores ,keep = torch.max(score,dim = 1)
                    idx = torch.arange(reg_boxes.shape[0])
                    detections = reg_boxes[idx,keep,:]
                    score = iou_score[idx,keep]
                    confs = confs[idx,keep]
                    classes = classes[idx,keep]
                    
                    
            
                 # 5b. convert to global image coordinates 
                     
                 # these detections are relative to crops - convert to global image coords
                 
    
                 
                 # add in original box offsets and scale outputs by original box scales
                 detections[:,0] = detections[:,0]*box_scales/224 + new_boxes[:,1]
                 detections[:,2] = detections[:,2]*box_scales/224 + new_boxes[:,1]
                 detections[:,1] = detections[:,1]*box_scales/224 + new_boxes[:,2]
                 detections[:,3] = detections[:,3]*box_scales/224 + new_boxes[:,2]
         
         
                 # convert into xysr form 
                 output = np.zeros([len(detections),4])
                 output[:,0] = (detections[:,0] + detections[:,2]) / 2.0
                 output[:,1] = (detections[:,1] + detections[:,3]) / 2.0
                 output[:,2] = (detections[:,2] - detections[:,0])
                 output[:,3] = (detections[:,3] - detections[:,1]) / output[:,2]
                 pred = torch.from_numpy(output)
                 
                 pred = detections
                 
                 # evaluate localizer
                 localizer_iou[frame_idx].append(iou(pred.double(),gt.double()))
                 error = (gt[:,:4]-pred)
                 meas_errors.append(error)
                 
                 loc_meas.append(pred[0].clone())
                 
                 # evaluate a posteriori estimate
                 tracker.update(pred,obj_ids)
                 objs = tracker.objs()
                 objs = [objs[key] for key in objs]
                 a_posteriori = torch.from_numpy(np.array(objs)).double()
                 a_posteriori_iou[frame_idx].append(iou(a_posteriori,gt))
                 
                 
                 apst_states.append(tracker.X[0].clone())
                 apst_covs.append(torch.diag(tracker.P[0]).clone())
                 
                 gts.append(gt[0].clone())
             
             if PLOT and iteration < 10:
                 plot_states(ap_states,
                             ap_covs,
                             loc_meas,
                             apst_states,
                             apst_covs,
                             gts,
                             save_num = iteration)
                 #break
             elif PLOT:
                 break
                 
             if iteration % 50 == 0:
                 print("Finished iteration {}".format(iteration))
    
         errors = torch.stack(ap_errors)
         errors = errors.view(-1,4)
         mean = torch.mean(errors, dim = 0)    
         covariance = torch.zeros((4,4))
         for vec in errors:
             covariance += torch.mm((vec - mean).unsqueeze(1), (vec-mean).unsqueeze(1).transpose(0,1))
         covariance = covariance / errors.shape[0]     
            
         print("------------------Results {}: --------------------".format(keep_num))
        # print("Skewed initialization IOUs: {}".format(sum(skewed_iou)/len(skewed_iou)))
         print("Starting state IOUs: {}".format(sum(starting_iou)/len(starting_iou)))
      
         iou_score = 0
         for key in a_priori_iou.keys():
             print("Frame {}".format(key))
             print("A priori state IOUs: {}".format(sum(a_priori_iou[key])/len(a_priori_iou[key])))
             print("Localizer state IOUs: {}".format(sum(localizer_iou[key])/len(localizer_iou[key])))
             print("A posteriori state IOUs: {}".format(sum(a_posteriori_iou[key])/len(a_posteriori_iou[key])))
             iou_score += sum(a_posteriori_iou[key])/len(a_posteriori_iou[key])
    
         print("A posteriori estimate error mean: {}".format(mean))
         print("A posteriori estimate error covariance: \n{}".format(covariance))
                                                
     iou_score /= key # last key = number of frames in rollout
    
     
     return iou_score 
 
#%% MAIN CODE BLOCK
if __name__ == "__main__":

    # define parameters
    b         = 50 # batch size
    n_pre     = 1      # number of frames used to initially tune tracker
    n_post    = 15     # number of frame irollouts used to evaluate tracker
    
    # create tracking dataset
    try:
        loader
    except: 
        train_im_dir  = "/home/worklab/Desktop/detrac/DETRAC-all-data"
        train_lab_dir = "/home/worklab/Desktop/detrac/DETRAC-Train-Annotations-XML-v3"
        dataset = Track_Dataset(train_im_dir,train_lab_dir,n = (n_pre + n_post+1))

        # create training params
        params = {'batch_size' : b,
                  'shuffle'    : True,
                  'num_workers': 0,
                  'drop_last'  : True
                  }
        
        # returns batch_size x (n_pre + n_post) x state_size tensor
        loader = DataLoader(dataset, **params)

    INIT = os.path.join(os.getcwd(),"config","filter_params_untuned.cpkl")
   

    #%% fit model error covariance
    with open(INIT,"rb") as f:
        kf_params = pickle.load(f)
    
    
    
    fit_Q(loader,
          kf_params,
          n_iterations = 20000,
          save_file = INIT,
          speed_init = "smooth",
          state_size = 8)    
    
    
     #%% fit measurement error 
    
    with open(INIT,"rb") as f:
        kf_params = pickle.load(f)
        
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    cp = os.path.join(os.getcwd(),"config","localizer_state_dict.pt")
    retinanet = resnet34(13)
    retinanet.load_state_dict(torch.load(cp))
    retinanet = retinanet.to(device)
    retinanet.eval()
    retinanet.training = False
    

    
    
    vecs = fit_localizer_R(loader,
                    kf_params, 
                    device, 
                    retinanet,
                    bers = [2.4],
                    save_file = INIT,
                    n_iterations = 200,
                    wer = 1.25)
        
    
    #%%        Rollouts
    with open(INIT,"rb") as f:
        kf_params = pickle.load(f)
        kf_params["R"] = kf_params["R"] / 20

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    cp = os.path.join(os.getcwd(),"config","localizer_state_dict.pt")
    retinanet = resnet34(13)
    retinanet.load_state_dict(torch.load(cp))
    retinanet = retinanet.to(device)
    retinanet.eval()
    retinanet.training = False
    
    filter_rollouts(loader,
                    kf_params,
                    retinanet,
                    device,
                    n_iterations = 1,
                    ber = 2.4,
                    skew_ratio = 0,
                    PLOT = True,
                    keep_nums = [1],
                    wer = 1.25)


    #%% Fit R2 for detector
    with open(INIT,"rb") as f:
        kf_params = pickle.load(f)
    # 
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    # Create the detector
    cp = os.path.join(os.getcwd(),"config","detector_state_dict.pt")
    retinanet = resnet50(13)
    retinanet.load_state_dict(torch.load(cp))
    retinanet = retinanet.to(device)
    retinanet.eval()
    retinanet.training = False
    
    try:
        det_loader
    except:
        det_dataset = Detection_Dataset(train_im_dir, train_lab_dir)
        params = {'batch_size' : 1,
                      'shuffle'    : True,
                      'num_workers': 0,
                      'drop_last'  : True,
                      "collate_fn" : collate
                      }
        det_loader = DataLoader(det_dataset, **params)

    fit_detector_R(det_loader,
                   kf_params, 
                   device, 
                   retinanet,
                   save_file = INIT,
                   n_iterations = 1000)