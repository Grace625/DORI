# DORI

## Introduction

Depth perception is crucial for numerous robotic applications. However, conventional self-supervised monocular depth estimation, using image reprojection loss as supervision, assumes a static scene, thus suffering from performance deterioration in dynamic scenes.% due to mismatch in those moving object areas.

In this repository, we propose DORI, a dynamic object restoration algorithm with instance segmentation, to resolve this issue. Taking inspiration by an intuitive but powerful observation that object movement between neighboring frames could be approximately regarded as uniform linear motion, we design and implement DORI which can restore the moved objects to avoid the mismatch in loss computation and enhance depth estimation performance. Experimental results on KITTI show that our CNN-based method outperforms previous CNN-based state-of-the-art(SOTA) by up to 2.5%, and sets a new SOTA $RMSE_{log}$ even in comparison with transformer-based SOTA baseline. As for other metrics, we can achieve comparable performance with the transformer-based SOTA, while consuming only 1.4% inference time of it, striking a better balance between efficiency and effectiveness. 



## Pretrained  Models

[ManyDepth + DORI](https://drive.google.com/drive/folders/1GwlUcRukLcddNnU3VuDb6ma4rbAoOBTr?usp=sharing)

[MonoDepth2 + DORI](https://drive.google.com/drive/folders/1l4iPz8IsmWxTlHB4eVT2yG2KdX5oKnk5?usp=sharing)



## Reproducing Paper Results

### Preparation for Mask2Former:

Install detectron2 from https://detectron2.readthedocs.io/en/latest/tutorials/install.html, then: 

```bash
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```



### Data preparation:

Download and process KITTI raw data following the instructions in : https://github.com/nianticlabs/monodepth2#-kitti-training-data



### Training:

```bash
# Train ManyDepth + DORI on KITTI Eigen split
CUDA_VISIBLE_DEVICES=<your_desired_GPU> \
python -m manydepth.train \
    --data_path <your_KITTI_path> \
    --log_dir <your_save_path>  \
    --model_name <your_model_name> \
    --temporal
# Train ManyDepth + DORI on KITTI Odom split
CUDA_VISIBLE_DEVICES=<your_desired_GPU> \
python -m manydepth.train \
    --data_path <your_KITTI_path> \
    --log_dir <your_save_path>  \
    --model_name <your_model_name> \
    --split odom \
    --dataset kitti_odom \
    --temporal   
    
# Train MonoDepth2 + DORI
CUDA_VISIBLE_DEVICES=<your_desired_GPU> \
python -m monodepth2.train \
    --data_path <your_KITTI_path> \
    --log_dir <your_save_path>  \
    --model_name <your_model_name> \
    --temporal
```



### Evaluation:

First run `export_gt_depth.py` to extract ground truth files, then run:

```bash
# Evaluate ManyDepth + DORI in KITTI Eigen split
CUDA_VISIBLE_DEVICES=<your_desired_GPU> \
python -m manydepth.evaluate_depth \
    --data_path <your_KITTI_path> \
    --load_weights_folder <your_model_path>
    --eval_mono
# Evaluate ManyDepth + DORI in KITTI Odom split
CUDA_VISIBLE_DEVICES=<your_desired_GPU> \
python -m manydepth.evaluate_pose \ 
    --eval_split odom_9 \ 
    --data_path <your_KITTI_path> \
    --load_weights_folder <your_model_path>
    --eval_mono
CUDA_VISIBLE_DEVICES=<your_desired_GPU> \
python -m manydepth.evaluate_pose \ 
    --eval_split odom_10 \ 
    --data_path <your_KITTI_path> \
    --load_weights_folder <your_model_path>
    --eval_mono

# Evaluate MonoDepth2 + DORI in KITTI Eigen split
CUDA_VISIBLE_DEVICES=<your_desired_GPU> \
python -m monodepth2.evaluate_depth \
    --data_path <your_KITTI_path> \
    --load_weights_folder <your_model_path>
    --eval_mono
```



## 