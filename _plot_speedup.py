import csv
import matplotlib.pyplot as plt

file = "tracking_results.csv"
#file = "results/SKIP_iou_test.csv"
#file = "results/Skip_original.csv"
skip_results = []
combo_results = []

with open(file,"r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            try:
                if len(row[0]) == 0: # space row
                    continue
                
                elif row[0][0] == "_": # name row
                        name = str(row[0][1:])
                
                elif row[0][0] == "d": # label row
                    continue
                
                else: # data row
                    det_step,_,_,_,mota,motp,precision,recall,idsw,mt,pt,ml,frag,fp,fn,fps = row
                    det_step = int(det_step)
                    mota = float(mota)
                    fps = float(fps)
                    
                    datum = [name,det_step,mota,fps]
                    if "SOTA" in name:
                        skip_results.append(datum)
                    else:
                        combo_results.append(datum)
            except:
                print(row)
                    
            
skip_mota = [datum[2] for datum in skip_results]
skip_fps = [datum[3] for datum in skip_results]
combo_mota = [datum[2] for datum in combo_results]
combo_fps = [datum[3] for datum in combo_results]

plt.scatter(combo_fps,combo_mota)
plt.plot(skip_fps,skip_mota)
plt.show()