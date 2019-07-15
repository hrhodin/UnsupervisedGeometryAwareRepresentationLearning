import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt

from datasets import collected_dataset
import sys, os, shutil

import numpy as np
#import pickle
import IPython

from utils import io as utils_io
from utils import datasets as utils_data
from utils import training as utils_train
from utils import plot_dict_batch as utils_plot_batch

from models import unet_encode3D
from losses import generic as losses_generic
from losses import images as losses_images

import math
import torch
torch.cuda.current_device() # to prevent  "Cannot re-initialize CUDA in forked subprocess." error on some configurations
import torch.optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models_tv

# for loading of training sets
#sys.path.insert(0,'../pytorch_human_reconstruction')
#import pytorch_datasets.dataset_factory as dataset_factory

import sys
sys.path.insert(0,'./ignite')
from ignite._utils import convert_tensor
from ignite.engine import Events

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

class IgniteTrainNVS:
    def run(self, config_dict_file, config_dict):
        #config_dict_test = {k:v for k,v in config_dict.items()}
        #config_dict_cams = {k:v for k,v in config_dict.items()}
        
        # some default values
        config_dict['implicit_rotation'] = config_dict.get('implicit_rotation', False)
        config_dict['skip_background'] = config_dict.get('skip_background', True)
        config_dict['loss_weight_pose3D'] = config_dict.get('loss_weight_pose3D', 0)
        config_dict['n_hidden_to3Dpose'] = config_dict.get('n_hidden_to3Dpose', 2)
        
        # create visualization windows
        try:
            import visdom
            vis = visdom.Visdom()
            if not vis.check_connection():
                vis = None
            print("WARNING: Visdom server not running. Please run python -m visdom.server to see visual output")
        except ImportError:
            vis = None
            print("WARNING: No visdom package is found. Please install it with command: \n pip install visdom to see visual output")
            #raise RuntimeError("WARNING: No visdom package is found. Please install it with command: \n pip install visdom to see visual output")
        vis_windows = {}
    
        # save path and config files
        save_path = self.get_parameter_description(config_dict)
        utils_io.savePythonFile(config_dict_file, save_path)
        utils_io.savePythonFile(__file__, save_path)
        
        # now do training stuff
        epochs = 40
        train_loader = self.load_data_train(config_dict)
        test_loader = self.load_data_test(config_dict)
        model = self.load_network(config_dict)
        model = model.to(device)
        optimizer = self.loadOptimizer(model,config_dict)
        loss_train,loss_test = self.load_loss(config_dict)
            
        trainer = utils_train.create_supervised_trainer(model, optimizer, loss_train, device=device)
        evaluator = utils_train.create_supervised_evaluator(model,
                                                metrics={#'accuracy': CategoricalAccuracy(),
                                                         'primary': utils_train.AccumulatedLoss(loss_test)},
                                                device=device)
    
        #@trainer.on(Events.STARTED)
        def load_previous_state(engine):
            utils_train.load_previous_state(save_path, model, optimizer, engine.state)
             
        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_progress(engine):
            # log the loss
            iteration = engine.state.iteration - 1
            if iteration % config_dict['print_every'] == 0:
                utils_train.save_training_error(save_path, engine, vis, vis_windows)
        
            # log batch example image
            if iteration in [0,100] or iteration % config_dict['plot_every'] == 0:
                utils_train.save_training_example(save_path, engine, vis, vis_windows, config_dict)
                
        #@trainer.on(Events.EPOCH_COMPLETED)
        @trainer.on(Events.ITERATION_COMPLETED)
        def validate_model(engine):
            iteration = engine.state.iteration - 1
            if (iteration+1) % config_dict['test_every'] != 0: # +1 to prevent evaluation at iteration 0
                return
            print("Running evaluation at iteration",iteration)
            evaluator.run(test_loader)
            avg_accuracy = utils_train.save_testing_error(save_path, engine, evaluator, vis, vis_windows)
    
            # save the best model
            utils_train.save_model_state(save_path, trainer, avg_accuracy, model, optimizer, engine.state)
    
        # print test result
        @evaluator.on(Events.ITERATION_COMPLETED)
        def log_test_loss(engine):
            iteration = engine.state.iteration - 1
            if iteration in [0,100]:
                utils_train.save_test_example(save_path, trainer, evaluator, vis, vis_windows, config_dict)
    
        # kick everything off
        trainer.run(train_loader, max_epochs=epochs)
        
    def load_network(self, config_dict):
        output_types= config_dict['output_types']
        
        use_billinear_upsampling = config_dict.get('upsampling_bilinear', False)
        lower_billinear = 'upsampling_bilinear' in config_dict.keys() and config_dict['upsampling_bilinear'] == 'half'
        upper_billinear = 'upsampling_bilinear' in config_dict.keys() and config_dict['upsampling_bilinear'] == 'upper'
        
        from_latent_hidden_layers = config_dict.get('from_latent_hidden_layers', 0)
        num_encoding_layers = config_dict.get('num_encoding_layers', 4)
        
        num_cameras = 4
        if config_dict['active_cameras']: # for H36M it is set to False
            num_cameras = len(config_dict['active_cameras'])
        
        if lower_billinear:
            use_billinear_upsampling = False
        network_single = unet_encode3D.unet(dimension_bg=config_dict['latent_bg'],
                                            dimension_fg=config_dict['latent_fg'],
                                            dimension_3d=config_dict['latent_3d'],
                                            feature_scale=config_dict['feature_scale'],
                                            shuffle_fg=config_dict['shuffle_fg'],
                                            shuffle_3d=config_dict['shuffle_3d'],
                                            latent_dropout=config_dict['latent_dropout'],
                                            in_resolution=config_dict['inputDimension'],
                                            encoderType=config_dict['encoderType'],
                                            is_deconv=not use_billinear_upsampling,
                                            upper_billinear=upper_billinear,
                                            lower_billinear=lower_billinear,
                                            from_latent_hidden_layers=from_latent_hidden_layers,
                                            n_hidden_to3Dpose=config_dict['n_hidden_to3Dpose'],
                                            num_encoding_layers=num_encoding_layers,
                                            output_types=output_types,
                                            subbatch_size=config_dict['useCamBatches'],
                                            implicit_rotation=config_dict['implicit_rotation'],
                                            skip_background=config_dict['skip_background'],
                                            num_cameras=num_cameras,
                                            )

        if 'pretrained_network_path' in config_dict.keys(): # automatic
            if config_dict['pretrained_network_path'] == 'MPII2Dpose':
                pretrained_network_path = '/cvlabdata1/home/rhodin/code/humanposeannotation/output_save/CVPR18_H36M/TransferLearning2DNetwork/h36m_23d_crop_relative_s1_s5_aug_from2D_2017-08-22_15-52_3d_resnet/models/network_000000.pth'
                print("Loading weights from MPII2Dpose")
                pretrained_states = torch.load(pretrained_network_path, map_location=device)
                utils_train.transfer_partial_weights(pretrained_states, network_single, submodule=0, add_prefix='encoder.') # last argument is to remove "network.single" prefix in saved network
            else:
                print("Loading weights from config_dict['pretrained_network_path']")
                pretrained_network_path = config_dict['pretrained_network_path']            
                pretrained_states = torch.load(pretrained_network_path, map_location=device)
                utils_train.transfer_partial_weights(pretrained_states, network_single, submodule=0) # last argument is to remove "network.single" prefix in saved network
                print("Done loading weights from config_dict['pretrained_network_path']")
        
        if 'pretrained_posenet_network_path' in config_dict.keys(): # automatic
            print("Loading weights from config_dict['pretrained_posenet_network_path']")
            pretrained_network_path = config_dict['pretrained_posenet_network_path']            
            pretrained_states = torch.load(pretrained_network_path, map_location=device)
            utils_train.transfer_partial_weights(pretrained_states, network_single.to_pose, submodule=0) # last argument is to remove "network.single" prefix in saved network
            print("Done loading weights from config_dict['pretrained_posenet_network_path']")
        return network_single
    
    def loadOptimizer(self,network, config_dict):
        if network.encoderType == "ResNet":
            params_all_id = list(map(id, network.parameters()))
            params_resnet_id = list(map(id, network.encoder.parameters()))
            params_except_resnet = [i for i in params_all_id if i not in params_resnet_id]
            
            # for the more complex setup
            params_toOptimize_id = (params_except_resnet
                             + list(map(id, network.encoder.layer4_reg.parameters()))
                             + list(map(id, network.encoder.layer3.parameters()))
                             + list(map(id, network.encoder.l4_reg_toVec.parameters()))
                             + list(map(id, network.encoder.fc.parameters())))
            params_toOptimize    = [p for p in network.parameters() if id(p) in params_toOptimize_id]
    
            params_static_id = [id_p for id_p in params_all_id if not id_p in params_toOptimize_id]
    
            # disable gradient computation for static params, saves memory and computation
            for p in network.parameters():
                if id(p) in params_static_id:
                    p.requires_grad = False
    
            print("Normal learning rate: {} params".format(len(params_toOptimize_id)))
            print("Static learning rate: {} params".format(len(params_static_id)))
            print("Total: {} params".format(len(params_all_id)))
    
            opt_params = [{'params': params_toOptimize, 'lr': config_dict['learning_rate']}]
            optimizer = torch.optim.Adam(opt_params, lr=config_dict['learning_rate']) #weight_decay=0.0005
        else:
            optimizer = torch.optim.Adam(network.parameters(), lr=config_dict['learning_rate'])
        return optimizer
    
    def load_data_train(self,config_dict):
        dataset = collected_dataset.CollectedDataset(data_folder=config_dict['dataset_folder_train'],
            input_types=config_dict['input_types'], label_types=config_dict['label_types_train'])

        batch_sampler = collected_dataset.CollectedDatasetSampler(data_folder=config_dict['dataset_folder_train'],
              actor_subset=config_dict['actor_subset'],
              useSubjectBatches=config_dict['useSubjectBatches'], useCamBatches=config_dict['useCamBatches'],
              batch_size=config_dict['batch_size_train'],
              randomize=True,
              every_nth_frame=config_dict['every_nth_frame'])

        loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0, pin_memory=False,
                                             collate_fn=utils_data.default_collate_with_string)
        return loader
    
    def load_data_test(self,config_dict):
        dataset = collected_dataset.CollectedDataset(data_folder=config_dict['dataset_folder_test'],
            input_types=config_dict['input_types'], label_types=config_dict['label_types_test'])

        batch_sampler = collected_dataset.CollectedDatasetSampler(data_folder=config_dict['dataset_folder_test'],
            useSubjectBatches=0, useCamBatches=config_dict['useCamBatches'],
            batch_size=config_dict['batch_size_test'],
            randomize=True,
            every_nth_frame=config_dict['every_nth_frame'])

        loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0, pin_memory=False,
                                             collate_fn=utils_data.default_collate_with_string)
        return loader
    
    def load_loss(self, config_dict):
        # normal
        if config_dict.get('MAE', False):
            pairwise_loss = torch.nn.modules.loss.L1Loss()
        else:
            pairwise_loss = torch.nn.modules.loss.MSELoss()
        image_pixel_loss = losses_generic.LossOnDict(key='img_crop', loss=pairwise_loss)
        
        image_imgNet_bare = losses_images.ImageNetCriterium(criterion=pairwise_loss, weight=config_dict['loss_weight_imageNet'], do_maxpooling=config_dict.get('do_maxpooling',True))
        image_imgNet_loss = losses_generic.LossOnDict(key='img_crop', loss=image_imgNet_bare)
    
        
        losses_train = []
        losses_test = []
        
        if 'img_crop' in config_dict['output_types']:
            if config_dict['loss_weight_rgb']>0:
                losses_train.append(image_pixel_loss)
                losses_test.append(image_pixel_loss)
            if config_dict['loss_weight_imageNet']>0:
                losses_train.append(image_imgNet_loss)
                losses_test.append(image_imgNet_loss)
                
        loss_train = losses_generic.PreApplyCriterionListDict(losses_train, sum_losses=True)
        loss_test  = losses_generic.PreApplyCriterionListDict(losses_test,  sum_losses=True)
                
        # annotation and pred is organized as a list, to facilitate multiple output types (e.g. heatmap and 3d loss)
        return loss_train, loss_test
    
    def get_parameter_description(self, config_dict):#, config_dict):
        folder = "../output/trainNVS_{note}_{encoderType}_layers{num_encoding_layers}_implR{implicit_rotation}_w3Dp{loss_weight_pose3D}_w3D{loss_weight_3d}_wRGB{loss_weight_rgb}_wGrad{loss_weight_gradient}_wImgNet{loss_weight_imageNet}_skipBG{latent_bg}_fg{latent_fg}_3d{skip_background}_lh3Dp{n_hidden_to3Dpose}_ldrop{latent_dropout}_billin{upsampling_bilinear}_fscale{feature_scale}_shuffleFG{shuffle_fg}_shuffle3d{shuffle_3d}_{training_set}_nth{every_nth_frame}_c{active_cameras}_sub{actor_subset}_bs{useCamBatches}_lr{learning_rate}_".format(**config_dict)
        folder = folder.replace(' ','').replace('../','[DOT_SHLASH]').replace('.','o').replace('[DOT_SHLASH]','../').replace(',','_')
        #config_dict['storage_folder'] = folder
        return folder
        
    
if __name__ == "__main__":
    config_dict_module = utils_io.loadModule("configs/config_train_encodeDecode.py")
    config_dict = config_dict_module.config_dict
    ignite = IgniteTrainNVS()
    ignite.run(config_dict_module.__file__, config_dict)