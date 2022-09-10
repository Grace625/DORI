# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402

import numpy as np
import time
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from .utils import readlines, sec_to_hm_str
from .layers import SSIM, BackprojectDepth, Project3D, transformation_from_parameters, \
    disp_to_depth, get_smooth_loss, compute_depth_errors, Sobel

from manydepth import datasets, networks
import matplotlib.pyplot as plt

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.config import CfgNode as CN
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from mask2former import add_maskformer2_config

from .matcher import HungarianMatcher

_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def setup_cfg(cfg_path, weight_path):
    """
    Load config from file and command-line arguments for Mask2Former.
    Args:
        cfg_path: path of config file(.yaml)
        weight_path: path of weight file(.pkl)
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = weight_path
    cfg.freeze()
    return cfg

@torch.jit.script
def fill_dynamic_obj(mask, delta_x, delta_y, source, img):
    """
    Restore the moved dynamic object to calculated position.
    Args:
        mask: Instance segmentation mask of the dynamic object.
        delta_x: Vertical displacement from current position to target position.
        delta_y: Horizontal displacement from current position to target position.
        source: The synthesized image I_{t+n->t}.
        img: The generated background.
    Return:
        img: Motion-resumed image.
    """

    N, H, W = mask.shape
    
    start_hl = torch.max(torch.zeros_like(delta_x), delta_x)
    end_hl = torch.min(torch.ones_like(delta_x) * H, H+delta_x)
    start_hr = torch.max(torch.zeros_like(delta_x), -delta_x)
    end_hr = torch.min(torch.ones_like(delta_x) * H, H-delta_x)

    start_wl = torch.max(torch.zeros_like(delta_y), delta_y)
    end_wl = torch.min(torch.ones_like(delta_y) * W, W+delta_y)
    start_wr = torch.max(torch.zeros_like(delta_y), -delta_y)
    end_wr = torch.min(W-delta_y, torch.ones_like(delta_y) * W)

    source_mv = torch.zeros((N, 3, H, W), device=source.device)
    mask_mv = torch.zeros(mask.shape, dtype=torch.bool, device=mask.device)
    for i in range(len(mask)):
        source_mv[i, :, start_hl[i]:end_hl[i], start_wl[i]:end_wl[i]] = source[:, start_hr[i]:end_hr[i], start_wr[i]:end_wr[i]]
        mask_mv[i, start_hl[i]:end_hl[i], start_wl[i]:end_wl[i]] = mask[i, start_hr[i]:end_hr[i], start_wr[i]:end_wr[i]]

    img_mv = mask_mv.unsqueeze(1).repeat(1,3,1,1) * source_mv
    img_sum = img_mv.sum(dim=0) # 3, H, W

    mask_or = torch.zeros((H, W), dtype=torch.bool, device=mask.device)
    for mask_item in mask_mv:
        mask_or = mask_or | mask_item
    img = torch.where(mask_or, img_sum, img)
    
    return img

@torch.jit.script
def generate_dynamic_instance(grid_h, grid_w, mask_last, mask_next, img_last, img_next):
    """
    Generate motion-resumed images I_{t+n->t}^{rec}
    Args:
        grid_h: torch.arange(H), helps to compute instance boundaries
        grid_w: torch.arange(W), helps to compute instance boundaries
        mask_last: Instance masks in I_{t-1->t}
        mask_next: Instance masks in I_{t+1->t}
        img_last: I_{t-1->t}
        img_next: I_{t+1->t}
    Return:
        Motion-resumed images.
        rec_last: I_{t-1->t}^{rec}
        rec_next: I_{t+1->t}^{rec}
    """
    
    mask_or = mask_last | mask_next
    mask_or_ = torch.zeros_like(mask_or[0])
    for mask_item in mask_or:
        mask_or_ = mask_or_ | mask_item
    num, H, W = mask_last.shape

    x = torch.arange(H, device=mask_last.device)
    y = torch.arange(W, device=mask_last.device)
    
    # compute instance boundaries
    grid_h = grid_h.repeat(num, 1, 1)
    grid_w = grid_w.repeat(num, 1, 1)

    # calculate four boundaries of bounding box
    inf = (H + 1) * (W + 1)

    h_sum_last = (mask_last * grid_h).sum(dim=2) # B, H
    h_nonzero = torch.where(h_sum_last==0, 0, x)
    low_last = h_nonzero.argmax(dim=1)
    h_nonzero = torch.where(h_nonzero == 0, inf, h_nonzero)
    top_last = h_nonzero.argmin(dim=1)

    w_sum_last = (mask_last * grid_w).sum(dim=1) # B, H
    w_nonzero = torch.where(w_sum_last==0, 0, y)
    right_last = w_nonzero.argmax(dim=1)
    w_nonzero = torch.where(w_nonzero == 0, inf, w_nonzero)
    left_last = w_nonzero.argmin(dim=1)
    
    h_sum_next = (mask_next * grid_h).sum(dim=2) # B, H
    h_nonzero = torch.where(h_sum_next==0, 0, x)
    low_next = h_nonzero.argmax(dim=1)
    h_nonzero = torch.where(h_nonzero == 0, inf, h_nonzero)
    top_next = h_nonzero.argmin(dim=1)

    w_sum_next = (mask_next * grid_w).sum(dim=1) # B, H
    w_nonzero = torch.where(w_sum_next==0, 0, y)
    right_next = w_nonzero.argmax(dim=1)
    w_nonzero = torch.where(w_nonzero == 0, inf, w_nonzero)
    left_next = w_nonzero.argmin(dim=1)

    # compute displacement of instance, according to margin displacement
    batch_slice = torch.arange(num, device=mask_last.device)
    delta_x = torch.stack([low_next-low_last, top_next-top_last], dim=1)
    delta_x_selected = delta_x[batch_slice, delta_x.abs().argmax(dim=1)]# .squeeze(dim=-1)

    delta_y = torch.stack([right_next-right_last, left_next-left_last], dim=1)
    delta_y_selected = delta_y[batch_slice, delta_y.abs().argmax(dim=1)]# .squeeze(dim=-1)
    disp_x = torch.round(delta_x_selected / 2).long()
    disp_y = torch.round(delta_y_selected / 2).long()
    
    delta_x_last = disp_x
    delta_y_last = disp_y
    delta_x_next = -disp_x
    delta_y_next = -disp_y
    
    # place dynamic objects on the generated background
    mask = mask_last & (~ mask_next)
    mask_bg = torch.zeros_like(mask[0])
    for ms in mask:
        mask_bg = mask_bg | ms
    img_bg = torch.where(mask_bg, img_next, img_last)
    
    image_syn_last = fill_dynamic_obj(mask_last, delta_x_last, delta_y_last, img_last, img_bg)
    rec_last = torch.where(mask_or_, image_syn_last, img_last)

    mask2 = mask_next & (~ mask_last)
    mask_bg2 = torch.zeros_like(mask2[0])
    for ms in mask2:
        mask_bg2 = mask_bg2 | ms
    img_bg2 = torch.where(mask_bg2, img_last, img_next)
    
    image_syn_next = fill_dynamic_obj(mask_next, delta_x_next, delta_y_next, img_next, img_bg2)
    rec_next = torch.where(mask_or_, image_syn_next, img_next)
    return rec_last, rec_next

class Trainer:
    def __init__(self, options):
        self.opt = options

        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        assert len(self.opt.frame_ids) > 1, "frame_ids must have more than 1 frame specified"

        self.train_teacher_and_pose = not self.opt.freeze_teacher_and_pose
        if self.train_teacher_and_pose:
            print('using adaptive depth binning!')
            self.min_depth_tracker = 0.1
            self.max_depth_tracker = 10.0
        else:
            print('fixing pose network and monocular network!')

        # check the frames we need the dataloader to load
        frames_to_load = self.opt.frame_ids.copy()
        self.matching_ids = [0]
        if self.opt.use_future_frame:
            self.matching_ids.append(1)
        for idx in range(-1, -1 - self.opt.num_matching_frames, -1):
            self.matching_ids.append(idx)
            if idx not in frames_to_load:
                frames_to_load.append(idx)

        print('Loading frames: {}'.format(frames_to_load))

        # MODEL SETUP
        self.models["encoder"] = networks.ResnetEncoderMatching(
            self.opt.num_layers, self.opt.weights_init == "pretrained",
            input_height=self.opt.height, input_width=self.opt.width,
            adaptive_bins=True, min_depth_bin=0.1, max_depth_bin=20.0,
            depth_binning=self.opt.depth_binning, num_depth_bins=self.opt.num_depth_bins)
        self.models["encoder"].to(self.device)

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)

        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.parameters_to_train += list(self.models["depth"].parameters())

        self.models["mono_encoder"] = \
            networks.ResnetEncoder(18, self.opt.weights_init == "pretrained")
        self.models["mono_encoder"].to(self.device)

        self.models["mono_depth"] = \
            networks.DepthDecoder(self.models["mono_encoder"].num_ch_enc, self.opt.scales)
        self.models["mono_depth"].to(self.device)

        if self.train_teacher_and_pose:
            self.parameters_to_train += list(self.models["mono_encoder"].parameters())
            self.parameters_to_train += list(self.models["mono_depth"].parameters())

        self.models["pose_encoder"] = \
            networks.ResnetEncoder(18, self.opt.weights_init == "pretrained",
                                   num_input_images=self.num_pose_frames)
        self.models["pose_encoder"].to(self.device)

        self.models["pose"] = \
            networks.PoseDecoder(self.models["pose_encoder"].num_ch_enc,
                                 num_input_features=1,
                                 num_frames_to_predict_for=2)
        self.models["pose"].to(self.device)

        if self.train_teacher_and_pose:
            self.parameters_to_train += list(self.models["pose_encoder"].parameters())
            self.parameters_to_train += list(self.models["pose"].parameters())
        
        if self.opt.temporal:
            cfg_ins = setup_cfg(self.opt.ins_config_path, self.opt.ins_weight_path)
            self.models["ins"] = build_model(cfg_ins)
            DetectionCheckpointer(self.models["ins"]).load(cfg_ins.MODEL.WEIGHTS)
            self.models["ins"].eval()

            self.matcher = HungarianMatcher()
        
        print("temporal = {}".format(self.opt.temporal))

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        if self.opt.mono_weights_folder is not None:
            self.load_mono_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # DATA
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "cityscapes_preprocessed": datasets.CityscapesPreprocessedDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join("splits", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            frames_to_load, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True,
            worker_init_fn=seed_worker)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            frames_to_load, 4, is_train=False, img_ext=img_ext)
        # self.val_loader = DataLoader(
        #     val_dataset, self.opt.batch_size, True,
        #     num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        # self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}

        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

        self.train_pose = True

    def set_train(self):
        """Convert all models to training mode
        """

        for k, m in self.models.items():
            if not self.train_pose:
                train_keys = ['mono_encoder', 'mono_depth', 'depth', 'encoder'] # if self.opt.train_sem else ['depth', 'encoder']
                if k in train_keys:
                    m.train()
            else:
                if self.train_teacher_and_pose:
                    if k == 'ins':
                        m.eval()
                    else:
                        m.train()
                else:
                    # if teacher + pose is frozen, then only use training batch norm stats for
                    # multi components
                    train_keys = ['depth', 'encoder']
                    if k in train_keys:
                        m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            if self.epoch == self.opt.freeze_teacher_epoch:
                if self.opt.freeze_pose_only:
                    self.freeze_pose()
                else: 
                    self.freeze_teacher()

            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
    
    def freeze_pose(self):
        if self.train_pose:
            self.train_pose = False
            print('freezing pose networks!')

            # here we reinitialise our optimizer to ensure there are no updates to the
            # teacher and pose networks
            self.parameters_to_train = []
            self.parameters_to_train += list(self.models["encoder"].parameters())
            self.parameters_to_train += list(self.models["depth"].parameters())
            self.parameters_to_train += list(self.models["mono_encoder"].parameters())
            self.parameters_to_train += list(self.models["mono_depth"].parameters())
            self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
            self.model_lr_scheduler = optim.lr_scheduler.StepLR(
                self.model_optimizer, self.opt.scheduler_step_size, 0.1)

            # set eval so that teacher + pose batch norm is running average
            self.set_eval()
            # set train so that multi batch norm is in train mode
            self.set_train()


    def freeze_teacher(self):
        if self.train_teacher_and_pose:
            self.train_teacher_and_pose = False
            self.temporal = False
            print('freezing teacher and pose networks!')

            # here we reinitialise our optimizer to ensure there are no updates to the
            # teacher and pose networks
            self.parameters_to_train = []
            self.parameters_to_train += list(self.models["encoder"].parameters())
            self.parameters_to_train += list(self.models["depth"].parameters())
            self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
            self.model_lr_scheduler = optim.lr_scheduler.StepLR(
                self.model_optimizer, self.opt.scheduler_step_size, 0.1)

            # set eval so that teacher + pose batch norm is running average
            self.set_eval()
            # set train so that multi batch norm is in train mode
            self.set_train()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs, is_train=True)
            
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                # self.val()

            if self.opt.save_intermediate_models and late_phase:
                self.save_model(save_step=True)

            if self.step == self.opt.freeze_teacher_step:
                if self.opt.freeze_pose_only:
                    self.freeze_pose()
                else:
                    self.freeze_teacher()

            self.step += 1
        self.model_lr_scheduler.step()
        
    def process_batch(self, inputs, is_train=False):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        mono_outputs = {}
        outputs = {}

        # predict poses for all frames
        if self.train_teacher_and_pose and self.train_pose:
            pose_pred = self.predict_poses(inputs, None)
        else:
            with torch.no_grad():
                pose_pred = self.predict_poses(inputs, None)
        outputs.update(pose_pred)
        mono_outputs.update(pose_pred)

        # grab poses + frames and stack for input to the multi frame network
        relative_poses = [inputs[('relative_pose', idx)] for idx in self.matching_ids[1:]]
        relative_poses = torch.stack(relative_poses, 1)

        lookup_frames = [inputs[('color_aug', idx, 0)] for idx in self.matching_ids[1:]]
        lookup_frames = torch.stack(lookup_frames, 1)  # batch x frames x 3 x h x w

        # apply static frame and zero cost volume augmentation
        batch_size = len(lookup_frames)
        augmentation_mask = torch.zeros([batch_size, 1, 1, 1]).to(self.device).float()
        if is_train and not self.opt.no_matching_augmentation:
            for batch_idx in range(batch_size):
                rand_num = random.random()
                # static camera augmentation -> overwrite lookup frames with current frame
                if rand_num < 0.25:
                    replace_frames = \
                        [inputs[('color', 0, 0)][batch_idx] for _ in self.matching_ids[1:]]
                    replace_frames = torch.stack(replace_frames, 0)
                    lookup_frames[batch_idx] = replace_frames
                    augmentation_mask[batch_idx] += 1
                # missing cost volume augmentation -> set all poses to 0, the cost volume will
                # skip these frames
                elif rand_num < 0.5:
                    relative_poses[batch_idx] *= 0
                    augmentation_mask[batch_idx] += 1
        outputs['augmentation_mask'] = augmentation_mask

        min_depth_bin = self.min_depth_tracker
        max_depth_bin = self.max_depth_tracker

        # single frame path
        if self.train_teacher_and_pose:
            feats = self.models["mono_encoder"](inputs["color_aug", 0, 0])
            mono_outputs.update(self.models['mono_depth'](feats))
        else:
            with torch.no_grad():
                feats = self.models["mono_encoder"](inputs["color_aug", 0, 0])
                mono_outputs.update(self.models['mono_depth'](feats))

        self.generate_images_pred(inputs, mono_outputs, mono=True)
        mono_losses = self.compute_losses(inputs, mono_outputs, is_multi=False)

        # update multi frame outputs dictionary with single frame outputs
        for key in list(mono_outputs.keys()):
            _key = list(key)
            if _key[0] in ['depth', 'disp']:
                _key[0] = 'mono_' + key[0]
                _key = tuple(_key)
                outputs[_key] = mono_outputs[key]

        # multi frame path
        features, lowest_cost, confidence_mask = self.models["encoder"](inputs["color_aug", 0, 0],
                                                                        lookup_frames,
                                                                        relative_poses,
                                                                        inputs[('K', 2)],
                                                                        inputs[('inv_K', 2)],
                                                                        min_depth_bin=min_depth_bin,
                                                                        max_depth_bin=max_depth_bin)
        outputs.update(self.models["depth"](features))
        
        outputs["lowest_cost"] = F.interpolate(lowest_cost.unsqueeze(1),
                                            [self.opt.height, self.opt.width],
                                            mode="nearest")[:, 0]
        outputs["consistency_mask"] = F.interpolate(confidence_mask.unsqueeze(1),
                                                    [self.opt.height, self.opt.width],
                                                    mode="nearest")[:, 0]

        if not self.opt.disable_motion_masking:
            outputs["consistency_mask"] = (outputs["consistency_mask"] *
                                        self.compute_matching_mask(outputs))

        self.generate_images_pred(inputs, outputs, is_multi=True)
        losses = self.compute_losses(inputs, outputs, is_multi=True)

        # update losses with single frame losses
        if self.train_teacher_and_pose:
            for key, val in mono_losses.items():
                losses[key] += val

        # update adaptive depth bins
        if self.train_teacher_and_pose and self.train_pose:
            self.update_adaptive_depth_bins(outputs)

        return outputs, losses

    def update_adaptive_depth_bins(self, outputs):
        """Update the current estimates of min/max depth using exponental weighted average"""

        min_depth = outputs[('mono_depth', 0, 0)].detach().min(-1)[0].min(-1)[0]
        max_depth = outputs[('mono_depth', 0, 0)].detach().max(-1)[0].max(-1)[0]

        min_depth = min_depth.mean().cpu().item()
        max_depth = max_depth.mean().cpu().item()

        # increase range slightly
        min_depth = max(self.opt.min_depth, min_depth * 0.9)
        max_depth = max_depth * 1.1

        self.max_depth_tracker = self.max_depth_tracker * 0.99 + max_depth * 0.01
        self.min_depth_tracker = self.min_depth_tracker * 0.99 + min_depth * 0.01

    def predict_poses(self, inputs, features=None):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # predict poses for reprojection loss
            # select what features the pose network takes as input
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
            for f_i in self.opt.frame_ids[1:]:
                # f_i = -1, 1
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

            # now we need poses for matching - compute without gradients
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.matching_ids}
            with torch.no_grad():
                # compute pose from 0->-1, -1->-2, -2->-3 etc and multiply to find 0->-3
                for fi in self.matching_ids[1:]:
                    # fi = -1
                    if fi < 0:
                        pose_inputs = [pose_feats[fi], pose_feats[fi + 1]]
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                        axisangle, translation = self.models["pose"](pose_inputs)
                        pose = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=True)

                        # now find 0->fi pose
                        if fi != -1:
                            pose = torch.matmul(pose, inputs[('relative_pose', fi + 1)])

                    else:
                        pose_inputs = [pose_feats[fi - 1], pose_feats[fi]]
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                        axisangle, translation = self.models["pose"](pose_inputs)
                        pose = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=False)

                        # now find 0->fi pose
                        if fi != 1:
                            pose = torch.matmul(pose, inputs[('relative_pose', fi - 1)])

                    # set missing images to 0 pose
                    for batch_idx, feat in enumerate(pose_feats[fi]):
                        if feat.sum() == 0:
                            pose[batch_idx] *= 0

                    inputs[('relative_pose', fi)] = pose
        else:
            raise NotImplementedError

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs, mono=False, is_multi=False):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth
            
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                T = outputs[("cam_T_cam", 0, frame_id)]
                if is_multi:
                    # don't update posenet based on multi frame prediction
                    T = T.detach()

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)
                
                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]
            
            # generate I_rec from It+1->t and It-1->t
            if mono and self.opt.temporal:
                self.generate_synthesised_image(inputs, outputs, scale)
            if not mono and self.opt.main_temporal:
                self.generate_synthesised_image(inputs, outputs, scale, mono=False)
                

    def generate_synthesised_image(self, inputs, outputs, scale, mono=True):
        """
        Restore dynamic object in warped images I_{t-1->t} & I_{t+1->t}
        """

        # predict instances for outputs[("color", frame_id, scale)]
        bs = inputs[("color", 0, 0)].shape[0]
        instances = self.generate_instances(inputs[("color", 0, 0)])
        
        syn_last = outputs[("color", -1, scale)].clone()
        syn_next = outputs[("color", 1, scale)].clone()
        if mono:
            self.has_ins = False
        else:
            self.main_has_ins = False

        H= syn_last.shape[-2]
        W= syn_last.shape[-1]

        x = torch.arange(H, device=syn_last.device)
        y = torch.arange(W, device=syn_last.device)
        grid_h, grid_w = torch.meshgrid(x, y, indexing='ij')
        
        for b in range(bs):
            instances_cur = instances[b]["instances"][instances[b]["instances"].scores > self.opt.ins_threshold]
            
            if len(instances_cur) == 0:
                continue
            
            img_last = outputs[("color", -1, scale)][b].clone()
            img_next = outputs[("color", 1, scale)][b].clone()

            instances_all = self.generate_instances(torch.cat([img_last.unsqueeze(0), img_next.unsqueeze(0)], dim=0))
            ins_last = instances_all[0]["instances"]
            ins_next = instances_all[1]["instances"]

            # Instance matching across frames
            slice_last, slice_next = self.matcher(ins_last, ins_next, instances_cur)
            
            if len(slice_last) + len(slice_next) == 0:
                continue 
            
            if mono:
                self.has_ins = True
            else:
                self.main_has_ins = True

            # Restore dynamic object, generated rectified images
            mask_last = ins_last.pred_masks[slice_last].bool()
            mask_next = ins_next.pred_masks[slice_next].bool()

            tmp_last, tmp_next = generate_dynamic_instance(grid_h, grid_w, mask_last, mask_next, img_last, img_next)
            syn_last[b] = (tmp_last.unsqueeze(0))
            syn_next[b] = (tmp_next.unsqueeze(0))

        if mono: 
            if self.has_ins:
                outputs[("syn", -1, scale)] = syn_last
                outputs[("syn", 1, scale)] = syn_next
        else:
            if self.main_has_ins:
                outputs[("syn", -1, scale)] = syn_last
                outputs[("syn", 1, scale)] = syn_next

    def generate_instances(self, images):
        """
        Generate instance segmentation results with Mask2Former
        """
        (height, width) = images.shape[-2:]

        # convert rgb to bgr
        permute = [2, 1, 0]
        images = images[:, permute, :, :]

        images = images * 255

        input_ls_dict = [
            {"image": img, "height": height, "width": width}
            for img in images
        ]
        with torch.no_grad():
            pred_instances = self.models["ins"](input_ls_dict)
        
        return pred_instances
    
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    @staticmethod
    def compute_loss_masks(reprojection_loss, identity_reprojection_loss):
        """ Compute loss masks for each of standard reprojection and depth hint
        reprojection"""

        if identity_reprojection_loss is None:
            # we are not using automasking - standard reprojection loss applied to all pixels
            reprojection_loss_mask = torch.ones_like(reprojection_loss)

        else:
            
            # we are using automasking
            all_losses = torch.cat([reprojection_loss, identity_reprojection_loss], dim=1)
            # print(all_losses.shape) # B, 2, H, W
            idxs = torch.argmin(all_losses, dim=1, keepdim=True)
            # print(idxs.shape) # B, 1, H, W
            reprojection_loss_mask = (idxs == 0).float()

        return reprojection_loss_mask

    def compute_matching_mask(self, outputs):
        """Generate a mask of where we cannot trust the cost volume, based on the difference
        between the cost volume and the teacher, monocular network"""

        mono_output = outputs[('mono_depth', 0, 0)]
        matching_depth = 1 / outputs['lowest_cost'].unsqueeze(1).to(self.device)

        # mask where they differ by a large amount
        mask = ((matching_depth - mono_output) / mono_output) < 1.0
        mask *= ((mono_output - matching_depth) / matching_depth) < 1.0
        return mask[:, 0]

    def compute_losses(self, inputs, outputs, is_multi=False):
        """Compute the reprojection, smoothness and proxy supervised losses for a minibatch
        """
        losses = {}
        total_loss = 0
        total_grad_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            if is_multi == False and self.opt.temporal and self.has_ins:
                for frame_id in self.opt.frame_ids[1:]:
                    pred = outputs[("syn", frame_id, scale)]
                    reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            
            if is_multi and self.opt.main_temporal and self.main_has_ins:
                for frame_id in self.opt.frame_ids[1:]:
                    pred = outputs[("syn", frame_id, scale)]
                    reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            
            if len(reprojection_losses) == 0:
                for frame_id in self.opt.frame_ids[1:]:
                    pred = outputs[("color", frame_id, scale)]
                    reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1) 

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # differently to Monodepth2, compute mins as we go
                    identity_reprojection_loss, _ = torch.min(identity_reprojection_losses, dim=1,
                                                              keepdim=True)
            else:
                identity_reprojection_loss = None

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                # differently to Monodepth2, compute mins as we go
                reprojection_loss, frame_idxs = torch.min(reprojection_losses, dim=1, keepdim=True)

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).to(self.device) * 0.00001

            # find minimum losses from [reprojection, identity]
            reprojection_loss_mask = self.compute_loss_masks(reprojection_loss,
                                                             identity_reprojection_loss)

            # find which pixels to apply reprojection loss to, and which pixels to apply
            # consistency loss to
            if is_multi:
                if self.opt.new_mask:
                    tmp_mask = torch.ones_like(reprojection_loss_mask)
                    if not self.opt.disable_motion_masking:
                        tmp_mask = (tmp_mask * outputs['consistency_mask'].unsqueeze(1))
                    if not self.opt.no_matching_augmentation:
                        tmp_mask = (tmp_mask * (1 - outputs['augmentation_mask']))
                    consistency_mask = (1 - tmp_mask).float()
             
                else:
                    reprojection_loss_mask = torch.ones_like(reprojection_loss_mask)
                    if not self.opt.disable_motion_masking:
                        reprojection_loss_mask = (reprojection_loss_mask *
                                                outputs['consistency_mask'].unsqueeze(1))
                    if not self.opt.no_matching_augmentation:
                        reprojection_loss_mask = (reprojection_loss_mask *
                                                (1 - outputs['augmentation_mask']))
                    consistency_mask = (1 - reprojection_loss_mask).float()

            # standard reprojection loss
            reprojection_loss = reprojection_loss * reprojection_loss_mask
            reprojection_loss = reprojection_loss.sum() / (reprojection_loss_mask.sum() + 1e-7)

            # consistency loss:
            # encourage multi frame prediction to be like singe frame where masking is happening
            if is_multi:
                multi_depth = outputs[("depth", 0, scale)]
                # no gradients for mono prediction!
                mono_depth = outputs[("mono_depth", 0, scale)].detach()
                consistency_loss = torch.abs(multi_depth - mono_depth) * consistency_mask
                consistency_loss = consistency_loss.mean()

                # save for logging to tensorboard
                consistency_target = (mono_depth.detach() * consistency_mask +
                                      multi_depth.detach() * (1 - consistency_mask))
                consistency_target = 1 / consistency_target
                outputs["consistency_target/{}".format(scale)] = consistency_target
                losses['consistency_loss/{}'.format(scale)] = consistency_loss
            else:
                consistency_loss = 0

            losses['reproj_loss/{}'.format(scale)] = reprojection_loss

            loss += reprojection_loss + consistency_loss

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss

            losses["ori_loss/{}".format(scale)] = loss
        
        total_loss /= self.num_scales

        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        min_depth = 1e-3
        max_depth = 80

        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = (depth_gt > min_depth) * (depth_gt < max_depth)

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        # for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
        #     s = 0  # log only max scale
        #     for frame_id in self.opt.frame_ids:
        #         writer.add_image(
        #             "color_{}_{}/{}".format(frame_id, s, j),
        #             inputs[("color", frame_id, s)][j].data, self.step)
        #         if s == 0 and frame_id != 0:
        #             writer.add_image(
        #                 "color_pred_{}_{}/{}".format(frame_id, s, j),
        #                 outputs[("color", frame_id, s)][j].data, self.step)

        #     disp = colormap(outputs[("disp", s)][j, 0])
        #     writer.add_image(
        #         "disp_multi_{}/{}".format(s, j),
        #         disp, self.step)

        #     disp = colormap(outputs[('mono_disp', s)][j, 0])
        #     writer.add_image(
        #         "disp_mono/{}".format(j),
        #         disp, self.step)

        #     if outputs.get("lowest_cost") is not None:
        #         lowest_cost = outputs["lowest_cost"][j]

        #         consistency_mask = \
        #             outputs['consistency_mask'][j].cpu().detach().unsqueeze(0).numpy()

        #         min_val = np.percentile(lowest_cost.numpy(), 10)
        #         max_val = np.percentile(lowest_cost.numpy(), 90)
        #         lowest_cost = torch.clamp(lowest_cost, min_val, max_val)
        #         lowest_cost = colormap(lowest_cost)

        #         writer.add_image(
        #             "lowest_cost/{}".format(j),
        #             lowest_cost, self.step)
        #         writer.add_image(
        #             "lowest_cost_masked/{}".format(j),
        #             lowest_cost * consistency_mask, self.step)
        #         writer.add_image(
        #             "consistency_mask/{}".format(j),
        #             consistency_mask, self.step)

        #         consistency_target = colormap(outputs["consistency_target/0"][j])
        #         writer.add_image(
        #             "consistency_target/{}".format(j),
        #             consistency_target.squeeze(), self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, save_step=False):
        """Save model weights to disk
        """
        if save_step:
            save_folder = os.path.join(self.log_path, "models", "weights_{}_{}".format(self.epoch,
                                                                                       self.step))
        else:
            save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            if model_name == 'sem':
                continue
            if model_name == 'ins':
                continue
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                # save estimates of depth bins
                to_save['min_depth_bin'] = self.min_depth_tracker
                to_save['max_depth_bin'] = self.max_depth_tracker
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_mono_model(self):

        model_list = ['pose_encoder', 'pose', 'mono_encoder', 'mono_depth']
        for n in model_list:
            print('loading {}'.format(n))
            path = os.path.join(self.opt.mono_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)

            if n == 'encoder':
                min_depth_bin = pretrained_dict.get('min_depth_bin')
                max_depth_bin = pretrained_dict.get('max_depth_bin')
                print('min depth', min_depth_bin, 'max_depth', max_depth_bin)
                if min_depth_bin is not None:
                    # recompute bins
                    print('setting depth bins!')
                    self.models['encoder'].compute_depth_bins(min_depth_bin, max_depth_bin)

                    self.min_depth_tracker = min_depth_bin
                    self.max_depth_tracker = max_depth_bin

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            try:
                print("Loading Adam weights")
                optimizer_dict = torch.load(optimizer_load_path)
                self.model_optimizer.load_state_dict(optimizer_dict)
            except ValueError:
                print("Can't load Adam - using random")
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")


def colormap(inputs, normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis
