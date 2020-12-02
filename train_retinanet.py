"""
Derek Gloudemans - August 4, 2020
This file contains a simple script to train a retinanet object detector on the UA Detrac
detection dataset.
- Pytorch framework
- Resnet-50 Backbone
- Manual file separation of training and validation data
- Automatic periodic checkpointing
"""

### Imports

import os ,sys
import numpy as np
import random 
import cv2
random.seed = 0
import torch
from torch.utils import data
from torch import optim
import collections

# add relevant packages and directories to path
detector_path = os.path.join(os.getcwd(),"models","pytorch_retinanet_detector")
sys.path.insert(0,detector_path)
detrac_util_path = os.path.join(os.getcwd(),"util_detrac")
sys.path.insert(0,detrac_util_path)

#from _detectors.pytorch_retinanet.retinanet import model, csv_eval 
from models.pytorch_retinanet_detector.retinanet import model
from util_detrac.detrac_detection_dataset import Detection_Dataset,collate


# surpress XML warnings (for UA detrac data)
import warnings
warnings.filterwarnings(action='once')

def plot_detections(dataset,retinanet):
    """
    Plots detections output
    """
    retinanet.training = False
    retinanet.eval()

    idx = np.random.randint(0,len(dataset))

    im,label,meta = dataset[idx]

    im = im.to(device).unsqueeze(0).float()
    #im = im[:,:,:224,:224]


    with torch.no_grad():

        scores,labels, boxes = retinanet(im)

    if len(boxes) > 0:
        keep = []    
        for i in range(len(scores)):
            if scores[i] > 0.5:
                keep.append(i)
        boxes = boxes[keep,:]

    im = dataset.denorm(im[0])
    cv_im = np.array(im.cpu()) 
    cv_im = np.clip(cv_im, 0, 1)

    # Convert RGB to BGR 
    cv_im = cv_im[::-1, :, :]  

    im = cv_im.transpose((1,2,0))

    for box in boxes:
        box = box.int()
        im = cv2.rectangle(im,(box[0],box[1]),(box[2],box[3]),(0.7,0.3,0.2),1)
    cv2.imshow("Frame",im)
    cv2.waitKey(2000)

    retinanet.train()
    retinanet.training = True
    retinanet.module.freeze_bn()


if __name__ == "__main__":


    depth = 50
    num_classes = 13
    patience = 0
    max_epochs = 50
    start_epoch = 0
    checkpoint_file = None

    # Paths to data here
    label_dir       = "/home/worklab/Desktop/detrac/DETRAC-Train-Annotations-XML-v3"
    train_partition = "/home/worklab/Desktop/detrac/DETRAC-train-data"
    val_partition   = "/home/worklab/Desktop/detrac/DETRAC-val-data"



    ###########################################################################


    # Create the model
    if depth == 18:
        retinanet = model.resnet18(num_classes=num_classes, pretrained=True)
    elif depth == 34:
        retinanet = model.resnet34(num_classes=num_classes, pretrained=True)
    elif depth == 50:
        retinanet = model.resnet50(num_classes=num_classes, pretrained=True)
    elif depth == 101:
        retinanet = model.resnet101(num_classes=num_classes, pretrained=True)
    elif depth == 152:
        retinanet = model.resnet152(num_classes=num_classes, pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')


    try:
        train_data
    except:
        # get dataloaders
        train_data = Detection_Dataset(train_partition,label_dir)
        val_data = Detection_Dataset(val_partition,label_dir)
        #train_data = LocMulti_Dataset(train_partition,label_dir)
        #val_data = LocMulti_Dataset(val_partition,label_dir)
        params = {'batch_size' : 8,
              'shuffle'    : True,
              'num_workers': 0,
              'drop_last'  : True,
              'collate_fn' : collate
              }
        trainloader = data.DataLoader(train_data,**params)
        testloader = data.DataLoader(val_data,**params)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # CUDA
    if use_cuda:
        if torch.cuda.device_count() > 1:
            retinanet = torch.nn.DataParallel(retinanet,device_ids = [0,1])
            retinanet = retinanet.to(device)
        else:
            retinanet = retinanet.to(device)


    try:
        if checkpoint_file is not None:
            retinanet.load_state_dict(torch.load(checkpoint_file).state_dict())
    except:
        retinanet.load_state_dict(torch.load(checkpoint_file)["model_state_dict"])


    retinanet.training = True
    # define optimizer and lr scheduler
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, verbose=True, mode = "min")


    # training mode
    retinanet.train()
    retinanet.module.freeze_bn()

    loss_hist = collections.deque(maxlen=500)
    most_recent_mAP = 0

    print('Num training images: {}'.format(len(train_data)))

    for epoch_num in range(start_epoch,max_epochs):


        print("Starting epoch {}".format(epoch_num))

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, (im,label,ignore) in enumerate(trainloader):
            
            retinanet.train()
            retinanet.training = True
            retinanet.module.freeze_bn()    
            
            try:
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([im.to(device).float(), label.to(device).float()])
                else:
                    classification_loss, regression_loss = retinanet([im.float(),label.float()])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                if iter_num % 10 == 0:
                    print(
                        'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))
                if iter_num % 100 == 0:
                    plot_detections(val_data, retinanet)

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        print("Epoch {} training complete".format(epoch_num))
        #print("Evaluating on validation dataset")
        #most_recent_mAP = csv_eval.evaluate(val_data,retinanet,iou_threshold = 0.7)

        scheduler.step(np.mean(epoch_loss))
        torch.cuda.empty_cache()
        #save
        PATH = "detrac_retinanet_34_{}.pt".format(epoch_num)
        torch.save(retinanet.state_dict(),PATH)


    retinanet.eval()

    #torch.save(retinanet, 'model_final.pt')
    most_recent_mAP = csv_eval.evaluate(val_data,retinanet,iou_threshold = 0.7)