"""
Description:
-----------
    Text Here
    
Parameters:
----------
    Text Here
    
Returns:
-------
    Text Here
"""
    
import motmetrics

import numpy as np
import xml.etree.ElementTree as ET
import _pickle as pickle
import matplotlib.pyplot as plt

def parse_labels(label_file):
     """
     Description
     -----------
     Returns a set of metadata (1 per track) and a list of labels (1 item per
     frame, where an item is a list of dictionaries (one dictionary per object
     with fields id, class, truncation, orientation, and bbox
     
     Parameters
     ----------
     label_file : str
         path where label xml files are stored for UA Detrac Dataset
     
     Returns
     all_boxes : list of list of dicts
        one dict per object, one list per frame
     sequence_metadata - dict
        info about sequence and ignored regions
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
         
         'car':0,
         'sedan':0,
         'hatchback':1,
         'suv':2,
         'van':3,
         'police':4,
         'taxi':5,
         'bus':6,
         'truck-box-large':7,
         'minivan':8,
         'truck-box-med':9,
         'truck-util':10,
         'truck-pickup':11,
         'truck-flatbed':12,
         'others':12,
         
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
                     'orientation':float(stats['orientation']),
                     'truncation':float(stats['truncation_ratio']),
                     'bbox':bbox
                     }
             try:
                 det_dict['color'] = stats['color']
             except:
                 pass
             
             frame_boxes.append(det_dict)
         all_boxes.append(frame_boxes)
     
     sequence_metadata = {
             'sequence':seq_name,
             'seq_attributes':seq_attrs,
             'ignored_regions':ignored_regions
             }
     return all_boxes, sequence_metadata

def test_regions(regions,x,y):
    """
    Description
    -----------
    Determines whether point (x,y) falls within any of regions
    
    Parameters
    ----------
    x : float
        x-coordinate
    y : float
        y-coordinate
    regions : list of np.array [4]
        xmin,ymin,xmax,ymax for each region
    
    Returns
    -------
    True if point is within any of the regions, False otherwise
    """
    
    for region in regions:
        if x > region[0] and x < region[2] and y > region[1] and y < region[3]:
            return True
    return False

def test_regions_fraction(regions,bbox,cutoff = 0.5):
    """
    Description
    -----------
    Determines whether a bbox overlaps with any of regions by above threshold
    
    Parameters
    ----------
    regions : list of np.array [4]
        xmin,ymin,xmax,ymax for each region
    bbox : np array [4]
        xmin ymin xmax ymax
    cutoff : float in range [0.1], optional
        overlap cutoff to consider boox in the ignore region. The default is 0.5.

    Returns
    -------
    True if bbox overlaps with any region by threshold or more, False otherwise.
    """
    
    for region in regions:
        # calculate iou
        area = (bbox[2] - bbox[0])     * (bbox[3] - bbox[1])
        
        xmin = max(region[0],bbox[0])
        xmax = min(region[2],bbox[2])
        ymin = max(region[1],bbox[1])
        ymax = min(region[3],bbox[3])
        intersection = max(0,(xmax - xmin)) * max(0,(ymax - ymin))
        overlap = intersection  / area
        
        if overlap > cutoff:
            return True
    
    return False

def evaluate_mot(preds,gts,ignored_regions = [],threshold = 100,ignore_threshold = 0.5):
    """
    Description:
    -----------
        Evaluates a set of multiple object tracks and returns a variety of MOT
        metrics for these tracks.
        
    Parameters:
    ----------
        preds : the predicted object locations and ids
        gts   : the ground-truth object locations and ids
        ** both preds and gts are expected to be of the form list[list[dict]]:
            - one outer list item per frame
            - one inner list item per object
            - dict must contain fields id, bbox (x0,y0,x1,y1), and class
        ignored_regions : list of np array [4]       
            regions from which to remove predictions
    Returns:
    -------
        metrics : list of MOT metrics for the tracks
    """
    
    acc = motmetrics.MOTAccumulator(auto_id = True)
    try:
        assert len(preds) == len(gts) , "Length of predictions and ground truths are not equal: {},{}".format(len(preds),len(gts))
    except AssertionError:
        while len(preds) > len(gts):
            gts.append([])

    for frame in range(len(gts)):
        # get gts in desired format
        gt = gts[frame]
        gt_ids = [] # object ids for each object in this frame
        for obj in gt:
            
            # gx = (obj["bbox"][0] + obj['bbox'][2]) /2.0
            # gy = (obj["bbox"][1] + obj['bbox'][3]) /2.0
            # exclude = test_regions(ignored_regions,gx,gy)
            
            # if not exclude:
                gt_ids.append(obj["id"])
        gt_ids = np.array(gt_ids)
        
        
        # get preds in desired format
        pred = preds[frame]
        pred_ids = [] # object ids for each object in this frame
        pred_idxs = [] # the index for each object in the frame, not id
        for i,obj in enumerate(pred):
            
            #pred object center
            #px = (obj["bbox"][0] + obj['bbox'][2]) /2.0
            #py = (obj["bbox"][1] + obj['bbox'][3]) /2.0
            #exclude = test_regions(ignored_regions,px,py)
            
            exclude = test_regions_fraction(ignored_regions,obj['bbox'], cutoff = ignore_threshold)
            
            if not exclude:
                pred_ids.append(obj["id"])
                pred_idxs.append(i)
        pred_ids = np.array(pred_ids)
        pred_idxs = np.array(pred_idxs)
        
        # get distance matrix in desired format
        
        if False: # use distance for matching
            dist = np.zeros([len(gt_ids),len(pred_ids)])
            for i in range(len(gt_ids)):
                for j in range(len(pred_ids)):
                    # ground truth object center
                    gx = (gt[i]["bbox"][0] + gt[i]['bbox'][2]) /2.0
                    gy = (gt[i]["bbox"][1] + gt[i]['bbox'][3]) /2.0
                    
                    # pred object center
                    px = (pred[j]["bbox"][0] + pred[j]['bbox'][2]) /2.0
                    py = (pred[j]["bbox"][1] + pred[j]['bbox'][3]) /2.0
                    
                    d = np.sqrt((px-gx)**2 + (py-gy)**2)
                    dist[i,j] = d
        
        else: # use iou for matching
            dist = np.ones([len(gt_ids),len(pred_ids)])
            ious = np.zeros([len(gt_ids),len(pred_ids)])
            for i in range(len(gt_ids)):
                for j in range(len(pred_ids)):
                    k = pred_idxs[j]
                    minx = max(gt[i]["bbox"][0],pred[k]["bbox"][0])
                    maxx = min(gt[i]["bbox"][2],pred[k]["bbox"][2])
                    miny = max(gt[i]["bbox"][1],pred[k]["bbox"][1])
                    maxy = min(gt[i]["bbox"][3],pred[k]["bbox"][3])
                    
                    intersection = max(0,maxx-minx) * max(0,maxy-miny)
                    a1 = (gt[i]["bbox"][2] - gt[i]['bbox'][0]) * (gt[i]["bbox"][3] - gt[i]['bbox'][1])
                    a2 = (pred[k]["bbox"][2] - pred[k]['bbox'][0]) * (pred[k]["bbox"][3] - pred[k]['bbox'][1])
                    
                    union = a1+a2-intersection
                    iou = intersection / union
                    dist[i,j] = 1-iou
                    ious[i,j] = iou
                    if dist[i,j] > threshold:
                        dist[i,j] = np.nan
            
        # if detection isn't close to any object (> threshold), remove
        # this is a cludgey fix since the detrac dataset doesn't have all of the vehicles labeled
        # get columnwise min
        if False:
            mins = np.min(dist,axis = 0)
            idxs = np.where(mins < threshold)
            
            pred_ids = pred_ids[idxs]
            dist = dist[:,idxs]
        
        # update accumulator
        acc.update(gt_ids,pred_ids,dist)
        
        print("\rEvaluating frame {} of {}".format(frame,len(gts)), end = '\r', flush = True)

        
    metric_module = motmetrics.metrics.create()
    summary = metric_module.compute(acc,metrics = ["num_frames",
                                                   "num_unique_objects",
                                                   "num_objects",
                                                   "mota",
                                                   "motp",
                                                   "precision",
                                                   "recall",
                                                   "num_switches",
                                                   "mostly_tracked",
                                                   "partially_tracked",
                                                   "mostly_lost",
                                                   "num_fragmentations",
                                                   "num_false_positives",
                                                   "num_misses",
                                                   "num_switches"])
    
    return summary,acc
        

if __name__ == "__main__":
    biggest_array = np.zeros([9,18])
    
    for num in [63525,63552]:
        all_step_metrics = {}
        for det_step in [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,23,26,29]:
    
            label_dir = "/home/worklab/Desktop/detrac/DETRAC-Train-Annotations-XML-v3"
            gt_file = "/home/worklab/Desktop/detrac/DETRAC-Train-Annotations-XML-v3/MVI_{}_v3.xml".format(num)
            pred_file = "/home/worklab/Documents/code/detection-by-tracking/preds_temp/preds_MVI_{}_{}.cpkl".format(num,det_step)
            # get gt labels
            gts,metadata = parse_labels(gt_file)
            ignored_regions = metadata['ignored_regions']
            
            # get pred labels
            try:
                with open(pred_file,"rb") as f:
                    preds,frame_rate = pickle.load(f)
            except:
                with open(pred_file,"rb") as f:
                    preds,frame_rate,time_met = pickle.load(f)
                
            metrics,acc = evaluate_mot(preds,gts,ignored_regions,threshold = 40)
            metrics = metrics.to_dict()
            metrics["framerate"] = frame_rate
            all_step_metrics[det_step] = metrics
            
        
        
        # aggregate and plot all 
        n_objs = metrics["num_unique_objects"][0]
        det_step = []
        mota = []
        motp = []
        mostly_tracked = []
        mostly_lost = []
        num_fragmentations = []
        num_switches = []
        num_fp = []
        num_fn = []
        framerate = []
        
        for d in all_step_metrics:
            det_step.append(d)
            mota.append(all_step_metrics[d]["mota"][0])
            motp.append(1.0 - ( all_step_metrics[d]["motp"][0]/ 1100))
            mostly_tracked.append(all_step_metrics[d]["mostly_tracked"][0]/n_objs)
            mostly_lost.append(all_step_metrics[d]["mostly_lost"][0]/n_objs)
            num_fragmentations.append(all_step_metrics[d]["num_fragmentations"][0])
            num_switches.append(all_step_metrics[d]["num_switches"][0])
            num_fp.append(all_step_metrics[d]["num_misses"][0])
            num_fn.append(all_step_metrics[d]["num_switches"][0])
            framerate.append(all_step_metrics[d]["framerate"])
        
        metrics_np= np.array([mota,motp,mostly_tracked,mostly_lost,num_fragmentations,num_switches,num_fp,num_fn,framerate])
        biggest_array+= metrics_np
        if False:
            plt.figure()
            plt.plot(det_step,mota)
            plt.plot(det_step,motp) #1100 = 1-(960**2 + 540**2 )**0.5, so normalize by image size
            plt.plot(det_step,mostly_tracked)
            plt.plot(det_step,mostly_lost)
            plt.plot(det_step,framerate)
            
            plt.legend(["MOTA","MOTP","MT","ML","100 Hz"])
        
    biggest_array = biggest_array / 2.0