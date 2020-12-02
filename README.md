# Localization-based Tracking 

I'm going to keep this repo light on identifying details for now to maintain due to double-blind submissions in progress. Check back later for more details!

This repository implements a fast integrated approach to object detection and tracking based primarily on localization rather than detection. Results, when published, are extremely promising in terms of pushing real-time trackign without extensive implementation engineering.

## Included Files:
- config/ - contains configuration file for Kalman filter. If you want to use the pretrained [localizer](https://github.com/DerekGloudemans/localization-based-tracking/releases/download/v1.0-alpha/localizer_state_dict.pt) and [detector](https://github.com/DerekGloudemans/localization-based-tracking/releases/download/v1.0-alpha/detector_state_dict.pt) weights from training on UA Detrac, save them in this folder.
- demo/ - currently empty, you can save track sequences (as series of numerically labeled frames, or standard video file formats) here and the demo will then track them. You can download two sequences from the UA Detrac Dataset [here](https://github.com/DerekGloudemans/localization-based-tracking/releases/download/v1.0-alpha/demo.zip) as an example.
- util_detrac/ - contains pytorch datasets for loading UA Detrac data for a variety of CV tasks, such as localization, tracking , object detection.
- util_eval/ - contains MOT evaluation code. It is not used in the demo. Examples of its use can be found in the older version of this repo (see below).
- util_track - contains utilities used for tracking.
- tracker.py - definition for Localization-based Tracker.
- demo.py - uses Localization-based Tracker to track all sequences saved within the demo/ directory.
- train_detector.py - trains a retinanet detector on the UA Detrac dataset.
- train_localizer.py - trains a retinanet localizer on the UA Detrac datset.
- tune_kf_params.py - tunes the model and measurement error covariances for the kalman filter to improve tracking performance.

## Acknowledgements:
I broke the submodule functionality of this repo so I don't technically include these repos as submodules, but I used them in the code.
- pymot-metrics - for evaluation of multiple object tracking outputs according to various well-established metrics: https://github.com/cheind/py-motmetrics 
- pytorch-retinanet - for detection and localization network architectures: https://github.com/yhenon/pytorch-retinanet
- Demo sequences are from UA Detrac.

## Older Code versions:
This code was ported for simplicity from https://github.com/DerekGloudemans-oldcode/tracking-by-localization. This repository includes a ton of files used during code development, with various localizer formulations, different state formulations, and a bunch of other doodads. If you're interested in a specific functionality that this repo doesn't contain, check that code as it may well contain the function. Fair warning, it is not extensively commented. Feel free to message for more details.
