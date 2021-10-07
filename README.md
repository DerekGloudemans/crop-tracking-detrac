# Crop-based Tracking 


This repository implements a fast integrated approach to object detection and tracking that performs detection on small crops from an overall frame rather than on the full frame. Results are extremely promising in terms of pushing real-time tracking without extensive implementation engineering provided that a decent explicit motion model can be formualted for object motion within the frame. Work resulting from this project will be presented at ICMLA 2021.

## Included Files:
- config/ - contains configuration file for Kalman filter. If you want to use the pretrained [crop detector](https://github.com/DerekGloudemans/localization-based-tracking/releases/download/v1.0-alpha/localizer_state_dict.pt) and [full-frame detector](https://github.com/DerekGloudemans/localization-based-tracking/releases/download/v1.0-alpha/detector_state_dict.pt) weights from training on UA Detrac, save them in this folder.
- demo/ - currently empty, you can save track sequences (as series of numerically labeled frames, or standard video file formats) here and the demo will then track them. You can download two sequences from the UA Detrac Dataset [here](https://github.com/DerekGloudemans/localization-based-tracking/releases/download/v1.0-alpha/demo.zip) as an example.
- util_detrac/ - contains pytorch datasets for loading UA Detrac data for a variety of CV tasks, such as localization, tracking , object detection.
- util_eval/ - contains MOT evaluation code. It is not used in the demo. Examples of its use can be found in the older version of this repo (see below).
- util_track - contains utilities used for tracking.
- ***tracker.py*** - definition for Crop-based Tracker. If you want to reproduce this tracker, this is the file you'll want to use. Pretty much all of the other files are specific either to the UA Detrac Dataset or to the Retinanet (Resnet50 FPN) object detector, but the tracking framework is generally extensible.
- demo.py - uses Crop-based Tracker to track all sequences saved within the demo/ directory.
- train_detector.py - trains a retinanet full-frame detector on the UA Detrac dataset.
- train_localizer.py - trains a retinanet crop detector on the UA Detrac dataset.
- tune_kf_params.py - tunes the model and measurement error covariances for the kalman filter to improve tracking performance.

## Acknowledgements:
I broke the submodule functionality of this repo so I don't technically include these repos as submodules, but I used them in the code.
- pymot-metrics - for evaluation of multiple object tracking outputs according to various well-established metrics: https://github.com/cheind/py-motmetrics 
- pytorch-retinanet - for detection and localization network architectures: https://github.com/yhenon/pytorch-retinanet
- Demo sequences are from UA Detrac.

## Older Code versions:
Previous versions of this work used the name [**Localization-based tracking**](https://arxiv.org/pdf/2104.05823.pdf)  (*localization* refers to finding the location of a single object within a frame, which is roughly the task performed by the crop detector). The term "localizer" still persists throughout this repository and can be thought to mean the same thing as "crop detector" for purposes of understanding the code and comments. (As an aside, you can still find LBT on the MOT20 tracking leaderboard! LBT was 10th overall when I was still working on the MOT dataset).

This code was ported for simplicity from https://github.com/DerekGloudemans-oldcode/tracking-by-localization. This repository includes a ton of files used during code development, with various localizer formulations, different state formulations, and a bunch of other doodads. If you're interested in a specific functionality that this repo doesn't contain, check that code as it may well contain the function. Fair warning, it is not extensively commented. Feel free to message for more details.

## Limitations
I appologize for the somewhat unpolished nature of this repository. I'm a lone grad student working on this and several other projects in a dev-focused manner, so I don't always have time to clean und package everything as I'd really like to. If you want to reproduce or extend this work, feel free to email me and I'll do my best to respond quickly and thoroughly. And if you're a recruiter, please check out some of my slighly more polished pinned repositories!
