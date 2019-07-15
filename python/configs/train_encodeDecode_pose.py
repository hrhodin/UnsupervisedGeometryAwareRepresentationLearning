import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt

from datasets import collected_dataset
import sys, os, shutil

from utils import io as utils_io
import numpy as np
import torch
torch.cuda.current_device() # to prevent  "Cannot re-initialize CUDA in forked subprocess." error on some configurations
import torch.optim
#import pickle
import IPython

import train_encodeDecode
from losses import generic as losses_generic
from losses import poses as losses_poses

class IgniteTrainPose(train_encodeDecode.IgniteTrainNVS):
    def loadOptimizer(self, network, config_dict):
        params_all_id = list(map(id, network.parameters()))
        params_posenet_id = list(map(id, network.to_pose.parameters()))
        params_toOptimize = [p for p in network.parameters() if id(p) in params_posenet_id]

        params_static_id = [id_p for id_p in params_all_id if not id_p in params_posenet_id]

        # disable gradient computation for static params, saves memory and computation
        for p in network.parameters():
            if id(p) in params_static_id:
                p.requires_grad = False

        print("Normal learning rate: {} params".format(len(params_posenet_id)))
        print("Static learning rate: {} params".format(len(params_static_id)))
        print("Total: {} params".format(len(params_all_id)))

        opt_params = [{'params': params_toOptimize, 'lr': config_dict['learning_rate']}]
        optimizer = torch.optim.Adam(opt_params, lr=config_dict['learning_rate']) #weight_decay=0.0005
        return optimizer

    def load_loss(self, config_dict):
        weight = 1
        if config_dict['training_set'] in ['h36m','h36m_mpii','h36m_cross']:
            weight = 17 / 16  # becasue spine is set to root = 0
        print("MPJPE test weight = {}, to normalize different number of joints".format(weight))
    
        pose_key = '3D'
        loss_train = losses_generic.LossLabelMeanStdNormalized(pose_key, torch.nn.MSELoss())
        loss_test = losses_generic.LossLabelMeanStdUnNormalized(pose_key, losses_poses.Criterion3DPose_leastQuaresScaled(losses_poses.MPJPECriterion(weight)), scale_normalized=False)

        # annotation and pred is organized as a list, to facilitate multiple output types (e.g. heatmap and 3d loss)
        return loss_train, loss_test

    def get_parameter_description(self, config_dict):#, config_dict):
        folder = "../output/trainPose_{note}_{encoderType}_layers{num_encoding_layers}_implR{implicit_rotation}_w3Dp{loss_weight_pose3D}_w3D{loss_weight_3d}_wRGB{loss_weight_rgb}_wGrad{loss_weight_gradient}_wImgNet{loss_weight_imageNet}_skipBG{latent_bg}_fg{latent_fg}_3d{skip_background}_lh3Dp{n_hidden_to3Dpose}_ldrop{latent_dropout}_billin{upsampling_bilinear}_fscale{feature_scale}_shuffleFG{shuffle_fg}_shuffle3d{shuffle_3d}_{training_set}_nth{every_nth_frame}_c{active_cameras}_sub{actor_subset}_bs{useCamBatches}_lr{learning_rate}_".format(**config_dict)
        folder = folder.replace(' ','').replace('../','[DOT_SHLASH]').replace('.','o').replace('[DOT_SHLASH]','../').replace(',','_')
        return folder

if __name__ == "__main__":
    config_dict_module = utils_io.loadModule("configs/config_train_encodeDecode_pose.py")
    config_dict = config_dict_module.config_dict
    ignite = IgniteTrainPose()
    ignite.run(config_dict_module.__file__, config_dict)