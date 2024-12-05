# Action-Recognition-Model
The project aims to develop an action recognition model using the MMAction framework, combining pose estimation (skeleton tracking) and object detection. The model should recognize specific actions performed by operators based on user poses and manipulated objects. Annotated videos will be provided directly and do not require additional development by the external team.

Objectives

Model Development:
- Create an action recognition architecture that integrates inputs from pose estimation (skeleton) and object detection.
- Design a custom pipeline within the MMAction framework to combine pose and object detection data for accurate action interpretation.

Training Script:
- Develop a Python training script that utilizes the custom MMAction pipeline.
- Implement the logic for model training and evaluation, focusing on achieving reliable and consistent results with defined performance metrics.
- Preparation for Inference Phase:
- Integrate external models for pose estimation (e.g., HRNet) and object detection (e.g., Faster R-CNN, YOLO).
- Configure the inference pipeline in MMAction to combine the outputs of these models during action recognition (fusion modules).
- Implement and validate the complete inference pipeline, ensuring it can process the provided videos in real-time or near real-time for action recognition.

Technical Requirements

Framework and Languages:
Framework: MMAction (based on PyTorch)

Programming Languages: Python
Libraries: PyTorch, OpenCV, MMDetection, MMSkeleton


Dataset Usage:

- Annotated videos with pose and object detections will be provided directly. There is no need for further annotation tool development.
- The model should be capable of using these annotations directly for training.

Custom Pipeline:
- Create an MMAction pipeline that combines pose estimation and object detection data.
- The pipeline should handle multi-stream inputs (pose and objects) and use an appropriate fusion method for accurate action interpretation.
Integrate and test external pose estimation and object detection models within the MMAction inference pipeline.

Training and Evaluation:
- The training script must support GPU training and include overfitting control techniques such as dropout and data augmentation.
- Implement evaluation methods for the model, such as accuracy, precision, recall, and F1-score, tailored to the specific context of the recognized actions.

Performance and Optimization:
- The model should achieve a minimum accuracy of 90% on a defined test set.
- Optimize the model for real-time inference, with a maximum processing time per frame of 50 ms.


Deliverables

      0. Proof of Competencies about composing MMAction pipeline with a Video Test
Compose Pipeline
Start a Training
Show model result

Model Code:
Python scripts for training and inference of the model.
Custom MMAction pipeline configured for the combination of pose and object detection.

Documentation:
Guide for setting up and configuring the development environment.
Manual for running the training script and evaluating the model.
Report on the model's performance, including the required evaluation metrics.

Trained Model:
Trained and optimized model ready for deployment.
Inference Pipeline:

Scripts and configurations for integrating pose estimation and object detection models during inference.
Validation of the inference pipeline with specific test cases to ensure real-time performance.
=================
To build an action recognition model using the MMAction framework with pose estimation and object detection, the following steps outline how to set up and implement the system. We'll focus on integrating external models for pose estimation (like HRNet) and object detection (e.g., Faster R-CNN or YOLO), followed by developing a custom MMAction pipeline for action recognition. This will also include training scripts, evaluation, and deployment strategies for real-time action recognition.
Step-by-Step Breakdown:

    Framework Setup
        Install necessary libraries:

    pip install torch torchvision mmcv mmaction mmdetection mmskeleton opencv-python

Setup MMAction Environment

    Clone the MMAction repository from GitHub:

    git clone https://github.com/open-mmlab/mmaction2.git
    cd mmaction2
    pip install -e .

Pose Estimation Integration (HRNet):

    HRNet is used for pose estimation. You can leverage the MMDetection or MMSkeleton framework for pose estimation integration. If the pre-trained HRNet model isn't directly available in MMAction, you may need to implement it separately.
    Use the pre-trained HRNet model or train it on your own dataset to get human pose estimations (skeletons) for each frame.
    Example of loading HRNet for pose estimation (assuming you have mmskeleton installed):

    from mmskeleton.apis import init_pose_model

    model = init_pose_model('configs/hrnet/hrnet_w32_coco_256x192.py', checkpoint='checkpoint.pth')

Object Detection Integration (Faster R-CNN/YOLO):

    Use MMDetection to integrate Faster R-CNN or YOLO for object detection.
    Example of loading Faster R-CNN model:

    from mmdet.apis import init_detector, inference_detector

    model = init_detector('configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py', checkpoint='checkpoint.pth')

Custom MMAction Pipeline Development: The pipeline will combine the outputs from pose estimation (HRNet) and object detection (Faster R-CNN or YOLO) for action recognition.

Key Steps:

    Preprocess the data: For each frame in the video, extract pose skeletons and objects.
    Combine the pose and object information: Use fusion methods (e.g., concatenation or attention-based fusion) to integrate these two data sources.
    Use the fused data as input to the action recognition model.

from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.apis import train_model
from mmcv.runner import get_dist_info, init_dist

# Define the custom action recognition model using pose + object features
model_cfg = {
    'type': 'Recognizer3D',  # Can also use 2D model depending on your dataset
    'backbone': {
        'type': 'ResNet',
        'depth': 50,
    },
    'cls_head': {
        'type': 'I3DHead',
        'num_classes': 10,  # Adjust based on the number of actions
    },
    'train_cfg': None,
    'test_cfg': {
        'average_clips': 'score',
    },
}

# Build the model
model = build_model(model_cfg)

# Specify dataset and pipeline for pose + object fusion
dataset_cfg = {
    'type': 'RawframeDataset',
    'ann_file': 'annotations/pose_and_object.json',  # Custom dataset with pose + object labels
    'data_prefix': 'data/video_frames',
    'pipeline': [
        # Custom preprocessing for pose + object data
    ],
}
dataset = build_dataset(dataset_cfg)

# Train the model
train_model(model, dataset)

Training Script: The training script trains the model using your custom dataset and the MMAction framework.

    Ensure that your dataset is annotated with both pose (skeleton) and object (bounding boxes) data for each frame.
    Define your custom pipeline to handle pose data (like joints) and object data (bounding boxes, class labels).

Example of training:

from mmaction.apis import train_model
from mmcv.runner import init_dist

# Custom data loader for video frames and annotations
train_dataset = build_dataset(dataset_cfg)

# Create model and config
model = build_model(model_cfg)

# Train the model
train_model(model, train_dataset)

Evaluation: Implement an evaluation script to assess model performance using common metrics for action recognition, such as accuracy, precision, recall, and F1-score.

from mmaction.apis import inference_recognizer
from mmcv import Config
from mmaction.datasets import build_dataloader

# Load trained model
model = build_model(model_cfg)

# Prepare test dataset
dataset_cfg['type'] = 'RawframeDataset'  # Or other dataset type
dataset_cfg['ann_file'] = 'annotations/test_annotations.json'
dataset_cfg['data_prefix'] = 'data/test_video_frames'

# Run inference
results = inference_recognizer(model, dataset_cfg, 'test_video.mp4')

# Print evaluation metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')

Inference Pipeline: In the inference phase, we need to combine the outputs from pose and object detection models and pass them to the action recognition model for prediction.

    def inference_pipeline(video_path):
        # Step 1: Pose Estimation
        pose_output = run_pose_estimation(video_path)  # Use HRNet or similar model
         
        # Step 2: Object Detection
        object_output = run_object_detection(video_path)  # Use Faster R-CNN or YOLO

        # Step 3: Combine Pose + Object Detection Features
        fused_features = fuse_pose_object_data(pose_output, object_output)
        
        # Step 4: Predict Action
        action = predict_action(fused_features)  # Pass fused features to the action recognition model
        
        return action

    Real-Time Optimization:
        Ensure that the model can process frames in real-time or near-real-time with a frame processing time below 50 ms.
        Use batch processing or model optimization techniques like quantization and pruning for faster inference.

    Deployment: For deployment, wrap the inference pipeline into a REST API or integrate it into your application. This way, you can handle real-time video streams for action recognition.

Deliverables:

    Code: Python scripts for training, inference, and evaluation of the model.
    Custom MMAction Pipeline: An integrated pipeline that combines pose and object detection data.
    Trained Model: The trained and optimized model, ready for deployment.
    Documentation: Instructions for setup, environment configuration, and usage.
    Performance Metrics: Report on model performance, including accuracy, precision, recall, and F1-score.

Conclusion:

This solution combines the MMAction framework with pose estimation and object detection to create a powerful action recognition model. The model leverages both human pose data and object interactions, allowing for accurate recognition of complex actions. By following the above steps, you can implement, train, and deploy a robust action recognition model using MMAction.
