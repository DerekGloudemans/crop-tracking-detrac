"""
Converts track output pickle file metrics into a csv file with aggregate metrics
"""
import _pickle as pickle
import csv
import os

directories = {
        "MOTA_SORT":"C:\\Users\\derek\\Desktop\\Results 2021 Detrac LBT\\DETRAC OUTPUTS 2021\\temp_outputs_sort",
        "MOTA_KIOU":"C:\\Users\\derek\\Desktop\\Results 2021 Detrac LBT\\DETRAC OUTPUTS 2021\\temp_outputs_conf08",
        "MOTA_SKIP":"C:\\Users\\derek\\Desktop\\Results 2021 Detrac LBT\\DETRAC OUTPUTS 2021\\temp_outputs_skip",
        "MOTA_SKIP1_LOC":"C:\\Users\\derek\\Desktop\\Results 2021 Detrac LBT\\DETRAC OUTPUTS 2021\\temp_outputs_skip1",
        "MOTA_SKIP2_LOC":"C:\\Users\\derek\\Desktop\\Results 2021 Detrac LBT\\DETRAC OUTPUTS 2021\\temp_outputs_skip2",
        "MOTA_SKIP3_LOC":"C:\\Users\\derek\\Desktop\\Results 2021 Detrac LBT\\DETRAC OUTPUTS 2021\\temp_outputs_skip3",
        "MOTA_SKIP4_LOC":"C:\\Users\\derek\\Desktop\\Results 2021 Detrac LBT\\DETRAC OUTPUTS 2021\\temp_outputs_skip4",
        "MOTA_DOWNSAMPLE":"C:\\Users\\derek\\Desktop\\Results 2021 Detrac LBT\\DETRAC OUTPUTS 2021\\temp_outputs_test_downsampling",
        "PRMOTA_TRAIN":"C:\\Users\\derek\\Desktop\\Results 2021 Detrac LBT\\DETRAC OUTPUTS 2021\\temp_outputs_train",
        "PRMOTA_TEST":"C:\\Users\\derek\\Desktop\\Results 2021 Detrac LBT\\DETRAC OUTPUTS 2021\\temp_outputs_test_PRMOTA_1_9"
            }

directories = {
        "MOTA_SORT"       :"/home/worklab/Data/cv/DETRAC OUTPUTS 2021/temp_outputs_sort",
        "MOTA_KIOU"       :"/home/worklab/Data/cv/DETRAC OUTPUTS 2021/temp_outputs_conf08",
        "MOTA_SKIP"       :"/home/worklab/Data/cv/DETRAC OUTPUTS 2021/temp_outputs_skip",
        "MOTA_SKIP1_LOC"  :"/home/worklab/Data/cv/DETRAC OUTPUTS 2021/temp_outputs_skip1",
        "MOTA_SKIP2_LOC"  :"/home/worklab/Data/cv/DETRAC OUTPUTS 2021/temp_outputs_skip2",
        "MOTA_SKIP3_LOC"  :"/home/worklab/Data/cv/DETRAC OUTPUTS 2021/temp_outputs_skip3",
        "MOTA_SKIP4_LOC"  :"/home/worklab/Data/cv/DETRAC OUTPUTS 2021/temp_outputs_skip4",
        "MOTA_DOWNSAMPLE" :"/home/worklab/Data/cv/DETRAC OUTPUTS 2021/temp_outputs_test_downsampling",
        "PRMOTA_TRAIN"    :"/home/worklab/Data/cv/DETRAC OUTPUTS 2021/temp_outputs_train",
        "PRMOTA_TEST"     :"/home/worklab/Data/cv/DETRAC OUTPUTS 2021/temp_outputs_test_PRMOTA_1_9"
            }

directories = {
    "_TRAIN_IOU_SKIP0"       : "/home/worklab/Documents/derek/detrac-lbt/results/rerun_skip_test",
    "_TRAIN_IOU_SKIP1"       : "/home/worklab/Documents/derek/detrac-lbt/results/rerun_skip_test",
    "_TRAIN_IOU_SKIP2"       : "/home/worklab/Documents/derek/detrac-lbt/results/rerun_skip_test",
    "_TRAIN_IOU_SKIP3"       : "/home/worklab/Documents/derek/detrac-lbt/results/rerun_skip_test",
    "_TRAIN_IOU_SKIP4"       : "/home/worklab/Documents/derek/detrac-lbt/results/rerun_skip_test",
    "_TRAIN_IOU_SKIP5"       : "/home/worklab/Documents/derek/detrac-lbt/results/rerun_skip_test",
    }

#directories = {"PRTEST_IOU"    : "/home/worklab/Documents/derek/detrac-lbt/results/PRMOTA_retest"}
#directories = {"SKIP_IOU"      :"/home/worklab/Documents/derek/detrac-lbt/results/PR_MOTA_train_iou"}        
directories = {"PRMOTA_TRAIN_SORT":"/home/worklab/Documents/derek/detrac-lbt/results/PR_MOTA_train_sort"}
directories = {"PRMOTA_TEST_SORT":"/home/worklab/Documents/derek/detrac-lbt/results/PR_MOTA_test_sort"}

directories = {
    "_TRAIN_sort_SKIP0"       : "/home/worklab/Documents/derek/detrac-lbt/results/TEST_skipping_sort",
    "_TRAIN_sort_SKIP1"       : "/home/worklab/Documents/derek/detrac-lbt/results/TEST_skipping_sort",
    "_TRAIN_sort_SKIP2"       : "/home/worklab/Documents/derek/detrac-lbt/results/TEST_skipping_sort",
    "_TRAIN_sort_SKIP3"       : "/home/worklab/Documents/derek/detrac-lbt/results/TEST_skipping_sort",
    "_TRAIN_sort_SKIP4"       : "/home/worklab/Documents/derek/detrac-lbt/results/TEST_skipping_sort",
    "_TRAIN_sort_SKIP5"       : "/home/worklab/Documents/derek/detrac-lbt/results/TEST_skipping_sort",
    }

all_results = dict([(item,{}) for item in directories.keys()])

for key in directories:
    path = directories[key]
    
    for file in os.listdir(path):
        file = os.path.join(path,file)
        
        if "PR" in key:
            
            det_step = int(file.split("_")[-3])
            track_id = int(file.split("_")[-4])
            conf = float(file.split("_")[-1].split(".cpkl")[0])
            
            if track_id in [40712,40774,40773,40772,40771,40711,40792,40775,39361,40901]:
                continue
            
            # add conf information
            track_id = "{}_{}".format(track_id,conf)
            #det_step = "{}-{}".format(det_step,conf)
           
        elif "DOWNSAMPLE" in key:
            track_id = int(file.split("_")[-3])
            det_step = float(file.split("_")[-1].split(".cpkl")[0]) # put ds ratio in det_step field so we can use the same code below
        
        elif "SKIP" in key:
            det_step = int(file.split("_")[-3])
            track_id = int(file.split("_")[-4])
            skip_step = int(file.split("_")[-2].split(".cpkl")[0])
            if skip_step-1 != int(key.split("SKIP")[1]):
                continue
        
        else:
            # get det_step
            det_step = int(file.split("_")[-2].split(".cpkl")[0])
            track_id = int(file.split("_")[-3])
        
        try:
            with open(file,"rb") as f:
                (tracklets,metrics,time_metrics) = pickle.load(f)
        except:
            try:
                with open(file,"rb") as f:
                    (tracklets,metrics,time_metrics,_) = pickle.load(f)
            except:
                continue
          
        print(file)

        
          
        # add tracks
        try:
            all_results[key][det_step][track_id] = (tracklets,metrics,time_metrics)
        except:
            all_results[key][det_step] = {track_id:(tracklets,metrics,time_metrics)}
            
#%%            
agg = {}
for key in all_results:
    agg[key] = {}
    for det_step in all_results[key]:
        agg[key][det_step] = {}
        
        n = 0
        aggregator = {}
        for track_id in all_results[key][det_step]:
            n += 1
            for metric in all_results[key][det_step][track_id][1]:
                try:
                    aggregator[metric] += all_results[key][det_step][track_id][1][metric][0]
                except:
                    aggregator[metric] = all_results[key][det_step][track_id][1][metric][0]
        for item in aggregator:
            aggregator[item] = aggregator[item]/n
        agg[key][det_step] = aggregator

#%%
with open('tracking_results.csv', mode='w') as f:
    writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    
    
    for key in agg:
        #write test titles
        writer.writerow([key])
        results = agg[key]
        
        # write metric titles
        all_metrics = ["det step"]
        first_key = list(results.keys())[0]
        for metric in results[first_key].keys():
            all_metrics.append(metric)
        writer.writerow(all_metrics)
        
        # write actual results
        det_steps = list(results.keys())
        det_steps.sort()
        for det_step in det_steps:
            result = results[det_step]
            
            line = [result[metric] for metric in result.keys()]
            line = [det_step] + line
            writer.writerow(line)
           
        # blank line between tests for clarity
        writer.writerow([])
        
        