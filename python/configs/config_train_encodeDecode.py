from utils import skeleton

# problem class parameters
numJoints = 17
inputDimension = 128

config_dict = {
    # general params
    'dpi' : 190,
    'config_class_file': 'dict_configs/config_class_encodeDecode.py',
    'input_types'       : ['img_crop','extrinsic_rot','extrinsic_rot_inv','bg_crop'],
    'output_types'      : ['3D','img_crop'],
    'label_types_train' : ['img_crop','3D','bounding_box_cam','intrinsic_crop','extrinsic_rot','extrinsic_rot_inv'],
    'label_types_test'  : ['img_crop','3D','bounding_box_cam','intrinsic_crop','extrinsic_rot','extrinsic_rot_inv'],
    'num_workers'       : 8,
    
    # problem class parameters
    'bones' : skeleton.bones_h36m,

    # opt parameters    
    'num_training_iterations' : 600000,
    'save_every' : 100000,
    'learning_rate' : 1e-3,# baseline: 0.001=1e-3
    'test_every' : 5000,
    'plot_every' : 5000,
    'print_every' : 100,

    # network parameters
    'batch_size_train' : 16,
    'batch_size_test' : 16, #10 #self.batch_size # Note, needs to be = self.batch_size for multi-view validation
    'outputDimension_3d' : numJoints * 3,
    'outputDimension_2d' : inputDimension // 8,

    # loss 
    'train_scale_normalized' : True,
    'train_crop_relative' : False,

    # dataset
    'dataset_folder_train' : '/cvlabdata1/home/rhodin/code/humanposeannotation/python/pytorch_human_reconstruction/TMP/H36M-MultiView-train',
    'dataset_folder_test' : '/cvlabdata1/home/rhodin/code/humanposeannotation/python/pytorch_human_reconstruction/TMP/H36M-MultiView-test',
    #'dataset_folder' :'/Users/rhodin/H36M-MultiView-test',
    'training_set' : 'h36m',
    'img_mean' : (0.485, 0.456, 0.406),
    'img_std' : (0.229, 0.224, 0.225),
    'actor_subset' : [1,5,6,7,8], # all training subjects
    'active_cameras' : False,
    'inputDimension' : inputDimension,
    'mirror_augmentation' : False,
    'perspectiveCorrection' : True,
    'rotation_augmentation' : True,
    'shear_augmentation' : 0,
    'scale_augmentation' : False,
    'seam_scaling' : 1.0,
    'useCamBatches' : 4,
    'useSubjectBatches' : True,
    'every_nth_frame' : 100,
    
    'note' : 'resL3',

    # encode decode
    'latent_bg' : 0,
    'latent_fg' : 24,
    'latent_3d' : 200*3,
    'latent_dropout' : 0.3,
    'from_latent_hidden_layers' : 0,
    'upsampling_bilinear' : 'upper',
    'shuffle_fg' : True,
    'shuffle_3d' : True,
    'feature_scale' : 4,
    'num_encoding_layers' : 4,
    'loss_weight_rgb' : 1,
    'loss_weight_gradient' : 0.01,
    'loss_weight_imageNet' : 2,
    'loss_weight_3d' : 0,
    'do_maxpooling' : False,
    'encoderType' : 'UNet',
    'implicit_rotation' : False,
    'predict_rotation' : False,
    'skip_background' : True,
}

# learning rate influence
config_dict['learning_rate'] = 1e-3
config_dict['actor_subset']  = [1,5,6,7,8]

config_dict['batch_size_train'] = 16
config_dict['batch_size_test'] = 8

config_dict['latent_fg'] = 128; config_dict['feature_scale'] = 2

config_dict['loss_weight_rgb']      = 1
config_dict['loss_weight_gradient'] = 0
config_dict['loss_weight_imageNet'] = 2

config_dict['useCamBatches'] = 2

# classic auto encoder, with some billinear layers
if 0:
    config_dict['shuffle_fg'] = False
if 0:
    config_dict['shuffle_3d'] = False

# no appearance
if 0:
    config_dict['latent_fg'] = 0

# RESNET
if 1:
    config_dict['batch_size_train'] = 32
    config_dict['encoderType'] = 'ResNet'

# dropout tests
if 0:
    config_dict['latent_dropout'] = 0

# enable or disable maxpooling in image net loss
if 1:
    config_dict['do_maxpooling'] = True

# Mean absolute or squared error?
if 1:
    config_dict['MAE'] = True

# smaller unsupervised subsets
if 1:
    config_dict['actor_subset'] = [1,5,6,7,8]
    #config_dict['actor_subset'] = [1,5,6]
    config_dict['actor_subset'] = [1]
