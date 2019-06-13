import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt

import math
import numpy as np

import torchvision

from utils import plotting as utils_plt
from utils import skeleton as util_skel
import torch

def normalize_mean_std_tensor(pose_tensor, label_dict):
    pose_mean = label_dict["pose_mean"]
    pose_std  = label_dict["pose_std"]
    return (pose_tensor-pose_mean)/pose_std

def denormalize_mean_std_tensor(pose_tensor, label_dict):
    pose_mean = label_dict["pose_mean"]
    pose_std  = label_dict["pose_std"]
    return pose_tensor*pose_std + pose_mean

def accumulate_heat_channels(heat_map_batch):
    plot_heat = heat_map_batch[:,-3:,:,:]
    num_tripels = heat_map_batch.size()[1]//3
    for i in range(0, num_tripels):
        plot_heat = torch.max(plot_heat, heat_map_batch[:,i*3:(i+1)*3,:,:])
    return plot_heat

def tensor_imshow(ax, img):
    npimg = img.numpy()
    npimg = np.swapaxes(npimg, 0, 2)
    npimg = np.swapaxes(npimg, 0, 1)

    npimg = np.clip(npimg, 0., 1.)
    ax.imshow(npimg)
    
def tensor_imshow_normalized(ax, img, mean=None, stdDev=None, im_plot_handle=None, x_label=None, clip=True):
    npimg = img.numpy()
    npimg = np.swapaxes(npimg, 0, 2)
    npimg = np.swapaxes(npimg, 0, 1)

    if mean is None:
        mean = (0.0, 0.0, 0.0)
    mean = np.array(mean)
    if stdDev is None:
        stdDev = np.array([1.0, 1.0, 1.0])
    stdDev = np.array(stdDev)

    npimg = npimg * stdDev + mean  # unnormalize
    
    if clip:
        npimg = np.clip(npimg, 0, 1)

    if im_plot_handle is not None:
        im_plot_handle.set_array(npimg)
    else:
        im_plot_handle = ax.imshow(npimg)
        
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off
    # when plotting 2D keypoints on top, this ensures that it only plots on the image region
    ax.set_ylim([img.size()[1],0])

    if x_label is not None:
        plt.xlabel(x_label)   

    return im_plot_handle

def plot_2Dpose_batch(ax, batch, offset_factor=0.8, bones=util_skel.bones_h36m, colormap='hsv'):
    num_batches = batch.shape[0]
    pose_2d_batchlinear = batch.reshape((num_batches,-1))
    num_joints  = pose_2d_batchlinear.shape[1]//2
    num_bones  = len(bones)
    pose_2d_cat = batch.reshape((-1,2))

    bones_cat = []
    color_order_cat = []
    for batchi in range(0,num_batches):
        # offset bones
        bones_new = []
        offset_i = batchi*num_joints
        for bone in bones:
            bone_new = [bone[0]+offset_i, bone[1]+offset_i]
            if pose_2d_cat[bone_new[0],0] <=0 or pose_2d_cat[bone_new[0],1]<=0 or pose_2d_cat[bone_new[1],0] <=0 or pose_2d_cat[bone_new[1],1] <=0:
                bone_new = [offset_i,offset_i] # disable line drawing, but don't remove to not disturb color ordering
            bones_new.append(bone_new)

        bones_cat.extend(bones_new)
        # offset colors
        color_order_cat.extend(range(0,num_bones))
        # offset skeletons horizontally
        offset_x = offset_factor*(batchi %8)
        offset_y = offset_factor*(batchi//8)
        pose_2d_cat[num_joints*batchi:num_joints*(batchi+1),:] += np.array([[offset_x,offset_y]])
    #plot_2Dpose(ax, pose_2d, bones, bones_dashed=[], bones_dashdot=[], color='red', linewidth=1, limits=None):
    utils_plt.plot_2Dpose(ax, pose_2d_cat.T, bones=bones_cat, colormap=colormap, color_order=color_order_cat)

def plot_3Dpose_batch(ax, batch_raw, offset_factor_x=None, offset_factor_y=None, bones=util_skel.bones_h36m, radius=0.01, colormap='hsv', row_length=8):
    num_batch_indices = batch_raw.shape[0]
    batch = batch_raw.reshape(num_batch_indices, -1)
    num_joints  = batch.shape[1]//3
    num_bones  = len(bones)
    pose_3d_cat = batch.reshape((-1,3))

    bones_cat = []
    color_order_cat = []
    for batchi in range(0,num_batch_indices):
        # offset bones
        bones_new = []
        offset = batchi*num_joints
        for bone in bones:
            bones_new.append([bone[0]+offset, bone[1]+offset])
        bones_cat.extend(bones_new)
        # offset colors
        color_order_cat.extend(range(0,num_bones))
        # offset skeletons horizontally
        if offset_factor_x is None:
            max_val_x = np.max(pose_3d_cat[num_joints*batchi:num_joints*(batchi+1),0])
            min_val_x = np.min(pose_3d_cat[num_joints*batchi:num_joints*(batchi+1),0])
            offset_factor_x = 1.5 * (max_val_x-min_val_x)
            radius = offset_factor_x/50
        if offset_factor_y is None:
            max_val_y = np.max(pose_3d_cat[num_joints*batchi:num_joints*(batchi+1),1])
            min_val_y = np.min(pose_3d_cat[num_joints*batchi:num_joints*(batchi+1),1])
            offset_factor_y = 1.3 * (max_val_y-min_val_y)
        offset_x = offset_factor_x*(batchi % row_length)
        offset_y = offset_factor_y*(batchi // row_length)
        pose_3d_cat[num_joints*batchi:num_joints*(batchi+1),:] += np.array([[offset_x,offset_y,0]])
    utils_plt.plot_3Dpose(ax, pose_3d_cat.T, bones_cat, radius=radius, colormap=colormap, color_order=color_order_cat, transparentBG=True)
    

def plot_iol(inputs_raw, labels_raw, outputs_dict, config_dict, keyword, image_name):
    print("labels_raw.keys() = {}, inputs_raw.keys() = {}, outputs_dict.keys() = {}".format(labels_raw.keys(), inputs_raw.keys(), outputs_dict.keys()))
        
    # init figure grid dimensions in an recursive call
    created_sub_plots = 0
    if not hasattr(plot_iol, 'created_sub_plots_last'):
        plot_iol.created_sub_plots_last = {}
    if keyword not in plot_iol.created_sub_plots_last:
        plot_iol.created_sub_plots_last[keyword] = 100 # some large defaul value to fit all..
        # call recursively once, to determine number of subplots
        plot_iol(inputs_raw, labels_raw, outputs_dict, config_dict, keyword, image_name)
        
    num_subplots_columns = 2
    title_font_size = 8
    num_subplots_rows = math.ceil(plot_iol.created_sub_plots_last[keyword]/2)
    
    # create figure
    plt.close("all")
    verbose = False
    if verbose:
        plt.switch_backend('Qt5Agg')
    else:
        plt.switch_backend('Agg') # works without xwindow
    fig    = plt.figure(0)
    plt.clf()

    ############### inputs ################
    # display input images
    if 'img_crop' in inputs_raw.keys():
        images_fg  = inputs_raw['img_crop'].cpu().data
        created_sub_plots += 1
        ax_img = fig.add_subplot(num_subplots_rows,num_subplots_columns,created_sub_plots)
        ax_img.set_title("Input images", size=title_font_size)
        grid_t = torchvision.utils.make_grid(images_fg, padding=0)
        if 'frame_info' in labels_raw.keys() and len(images_fg)<8:
            frame_info = labels_raw['frame_info'].data
            cam_idx_str = ', '.join([str(int(tensor)) for tensor in frame_info[:,0]])
            global_idx_str = ', '.join([str(int(tensor)) for tensor in frame_info[:,1]])
            x_label = "cams: {}".format(cam_idx_str)
        else:
            x_label = ""
        tensor_imshow_normalized(ax_img, grid_t, mean=config_dict['img_mean'], stdDev=config_dict['img_std'], x_label=x_label)
    # display input images
    if 'bg_crop' in inputs_raw.keys():
        images_bg  = inputs_raw['bg_crop'].cpu().data
        created_sub_plots += 1
        ax_img = fig.add_subplot(num_subplots_rows,num_subplots_columns,created_sub_plots)
        ax_img.set_title("Background images", size=title_font_size)
        grid_t = torchvision.utils.make_grid(images_bg, padding=0)
        x_label = ""
        tensor_imshow_normalized(ax_img, grid_t, mean=config_dict['img_mean'], stdDev=config_dict['img_std'], x_label=x_label)
        
        # difference
        images_diff = torch.abs(images_fg-images_bg)
        images_diff_max, i = torch.max(images_diff,dim=1,keepdim=True)

        images_diff = images_diff_max.expand_as(images_diff)
        images_diff = images_diff/torch.max(images_diff)
        created_sub_plots += 1
        ax_img = fig.add_subplot(num_subplots_rows,num_subplots_columns,created_sub_plots)
        ax_img.set_title("Background - foreground", size=title_font_size)
        grid_t = torchvision.utils.make_grid(images_diff, padding=0)
        tensor_imshow_normalized(ax_img, grid_t, x_label=x_label)
        
    # display input heat map
    if '2D_heat' in inputs_raw.keys():
        created_sub_plots += 1
        ax_img = fig.add_subplot(num_subplots_rows,num_subplots_columns,created_sub_plots)
        ax_img.set_title("2D heat input", size=title_font_size)
        input_heat = accumulate_heat_channels(inputs_raw['2D_heat']).data.cpu()
        tensor_imshow(ax_img, torchvision.utils.make_grid( input_heat, padding=0))

    # display input images
    depth_maps_norm = None
    if 'depth_map' in inputs_raw.keys():
        depth_maps  = inputs_raw['depth_map'].cpu().data
        created_sub_plots += 1
        ax_img = fig.add_subplot(num_subplots_rows,num_subplots_columns,created_sub_plots)

        msk_valid = depth_maps != 0
        msk_zero = depth_maps == 0
        if msk_valid.sum()>0:
            min_v = depth_maps[msk_valid].min()
            max_v = depth_maps[msk_valid].max()
        else:
            min_v = 0
            max_v = 1
        depth_maps_norm = (depth_maps - min_v) / (max_v - min_v) * 1.
        # display background as black for better contrast
        if msk_zero.sum()>0:
            depth_maps_norm[msk_zero] = 1.

        grid_t = torchvision.utils.make_grid(depth_maps_norm, padding=0)
        tensor_imshow_normalized(ax_img, grid_t, mean=config_dict['img_mean'], stdDev=config_dict['img_std'])
        ax_img.set_title("Input depth map\n(min={:0.4f},\n max={:0.4f})".format(min_v,max_v), size=title_font_size)
        
    ############### labels_raw ################
    # heatmap label
    if '2D_heat' in labels_raw.keys():
        created_sub_plots += 1
        ax_label = fig.add_subplot(num_subplots_rows,num_subplots_columns,created_sub_plots)
        ax_label.set_title("2D heat label", size=title_font_size)
        label_heat = labels_raw['2D_heat'].data.cpu()
        numJoints = label_heat.size()[1]
        plot_heat = accumulate_heat_channels(label_heat)
        tensor_imshow(ax_label, torchvision.utils.make_grid( plot_heat[:,:,:,:], padding=0))
        
    # 2D labelss
    if '2D' in labels_raw.keys():
        created_sub_plots += 1
        ax_img = fig.add_subplot(num_subplots_rows,num_subplots_columns,created_sub_plots)
        ax_img.set_title("2D labels_raw (crop relative)", size=title_font_size)
        if 'img_crop' in inputs_raw.keys():
            grid_t = torchvision.utils.make_grid(images_fg, padding=0)
            tensor_imshow_normalized(ax_img, grid_t, mean=config_dict['img_mean'], stdDev=config_dict['img_std'])
        elif depth_maps_norm is not None:
            grid_t = torchvision.utils.make_grid(depth_maps_norm, padding=0)
            tensor_imshow_normalized(ax_img, grid_t, mean=config_dict['img_mean'], stdDev=config_dict['img_std'])
        
        label_pose = labels_raw['2D'].data.cpu()
        #outputs_pose_3d = outputs_pose.numpy().reshape(-1,3)
        utils_plt.plot_2Dpose_batch(ax_img, label_pose.numpy()*256, offset_factor=256, bones=config_dict['bones'], colormap='hsv')

    if '2D_noAug' in labels_raw.keys() and 'img_crop_noAug' in inputs_raw.keys():
        created_sub_plots += 1
        ax_img = fig.add_subplot(num_subplots_rows,num_subplots_columns,created_sub_plots)
        ax_img.set_title("2D labels_raw (noAug)", size=title_font_size)
        images_noAug  = inputs_raw['img_crop_noAug'].cpu().data
        grid_t = torchvision.utils.make_grid(images_noAug, padding=0)
        tensor_imshow_normalized(ax_img, grid_t, mean=config_dict['img_mean'], stdDev=config_dict['img_std'])
        label_pose = labels_raw['2D_noAug'].data.cpu()
        img_shape = images_noAug[0].size()[1]
        utils_plt.plot_2Dpose_batch(ax_img, label_pose.numpy()*img_shape, offset_factor=img_shape, bones=config_dict['bones'], colormap='hsv')
           
    # plot 3D pose labels_raw
    if any(x in labels_raw.keys() for x in ['3D','3D_crop_coord']):
        try:    lable_pose = labels_raw['3D']
        except:
            try:    lable_pose = labels_raw['3D_crop_coord']
            except: lable_pose = None
            
        if lable_pose is not None:
            created_sub_plots += 1
            ax_3d_l   = fig.add_subplot(num_subplots_rows,num_subplots_columns,created_sub_plots, projection='3d')
            ax_3d_l.set_title("3D pose labels_raw", size=title_font_size)
            plot_3Dpose_batch(ax_3d_l, lable_pose.data.cpu().numpy(), bones=config_dict['bones'], radius=0.01, colormap='hsv')
            ax_3d_l.invert_zaxis()
            ax_3d_l.grid(False)
            if 1: # display a rotated version
                created_sub_plots += 1
                ax_3d_l   = fig.add_subplot(num_subplots_rows,num_subplots_columns,created_sub_plots, projection='3d')
                ax_3d_l.set_title("3D pose labels_raw (rotated)", size=title_font_size)
                a = -np.pi/2
                R = np.array([[np.cos(a),0,-np.sin(a)],
                              [0,1,0],
                              [np.sin(a),0, np.cos(a)]])
                pose_orig = lable_pose.data.cpu().numpy()
                pose_rotated = pose_orig.reshape(-1,3) @ R.T
                plot_3Dpose_batch(ax_3d_l, pose_rotated.reshape(pose_orig.shape), bones=config_dict['bones'], radius=0.01, colormap='hsv')
                ax_3d_l.invert_zaxis()
                ax_3d_l.grid(False)

            
    # draw projection of 3D pose
    if '3D_global' in labels_raw.keys():
        created_sub_plots += 1
        ax_img = fig.add_subplot(num_subplots_rows,num_subplots_columns,created_sub_plots)
        ax_img.set_title("Projected 3D labels_raw", size=title_font_size)
        if 'img_crop' in inputs_raw.keys():
            grid_t = torchvision.utils.make_grid(images_fg, padding=0)
            tensor_imshow_normalized(ax_img, grid_t, mean=config_dict['img_mean'], stdDev=config_dict['img_std'])

    ############### network output ################
    # 3D pose label
    #train_crop_relative = hasattr(self, 'train_crop_relative') and self.train_crop_relative
    if '3D' in outputs_dict.keys():
            outputs_pose = outputs_dict['3D']
            outputs_pose = outputs_pose.cpu().data
            if config_dict['train_scale_normalized'] == 'mean_std':
                outputs_pose = denormalize_mean_std_tensor(outputs_pose, labels_raw)
            
            created_sub_plots += 1
            ax_3dp_p   = fig.add_subplot(num_subplots_rows,num_subplots_columns,created_sub_plots, projection='3d')
            ax_3dp_p.set_title("3D prediction", size=title_font_size)
            plot_3Dpose_batch(ax_3dp_p, outputs_pose.numpy(), bones=config_dict['bones'], radius=0.01, colormap='hsv')
            ax_3dp_p.invert_zaxis()
            ax_3dp_p.grid(False)
            if 1: # display a rotated version
                created_sub_plots += 1
                ax_3d_l   = fig.add_subplot(num_subplots_rows,num_subplots_columns,created_sub_plots, projection='3d')
                ax_3d_l.set_title("3D pose prediction (rotated)", size=title_font_size)
                a = -np.pi/2
                R = np.array([[np.cos(a),0,-np.sin(a)],
                              [0,1,0],
                              [np.sin(a),0, np.cos(a)]])
                pose_rotated = outputs_pose.numpy().reshape(-1,3) @ R.T
                plot_3Dpose_batch(ax_3d_l, pose_rotated.reshape(outputs_pose.numpy().shape), bones=config_dict['bones'], radius=0.01, colormap='hsv')
                ax_3d_l.invert_zaxis()
                ax_3d_l.grid(False)
        
    if '2D_heat' in outputs_dict.keys():
#           output_index = config_dict['output_types'].index("2D_heat")
        output_heat = outputs_dict['2D_heat'].cpu().data
        #utils_plt.plot_2Dpose(ax_img, pose_2d_cat.T, bones=bones_cat, colormap=colormap, color_order=color_order_cat)
        created_sub_plots += 1
        ax_img = fig.add_subplot(num_subplots_rows,num_subplots_columns,created_sub_plots)
        ax_img.set_title("Predicted 2D labels_raw", size=title_font_size)
        if 'img_crop' in inputs_raw.keys():
            grid_t = torchvision.utils.make_grid(images_fg, padding=0)
            tensor_imshow_normalized(ax_img, grid_t, mean=config_dict['img_mean'], stdDev=config_dict['img_std'])
        for bi in range(0,output_heat.size()[0]):
            jointPositions_2D, confidences, joints_confident = utils_generic.jointPositionsFromHeatmap(output_heat[bi])
            map_width = output_heat[bi].size()[2]
            jointPositions_2D_crop = jointPositions_2D / map_width # normalize to 0..1
#            X = [xy[0] / 32*256 for xy in joints_confident.values()]
#            Y = [xy[1] / 32*256 for xy in joints_confident.values()]
            by = bi //8
            bx = bi % 8
            jointPositions_2D_pix =  np.concatenate((256*(bx+jointPositions_2D_crop[:,0,np.newaxis]), 256*(by+jointPositions_2D_crop[:,1,np.newaxis])),1)
            utils_plt.plot_2Dpose(ax_img, jointPositions_2D_pix.T, bones=utils_plt.bones_h36m, colormap='hsv')

        created_sub_plots += 1
        ax_label = fig.add_subplot(num_subplots_rows,num_subplots_columns,created_sub_plots)
        ax_label.set_title("Predicted 2D heatmaps", size=title_font_size)
        plot_heat = accumulate_heat_channels(output_heat)
        tensor_imshow(ax_label, torchvision.utils.make_grid(plot_heat, padding=0))

        # also display backtransformed heatmaps (undoing augmentation and perspective correction)
        if 0: #'trans_2d_inv' in labels_raw.keys():
            numJoints = output_heat.size()[1]//3
            heat_batch = outputs_dict['2D_heat'].cpu().data
            batch_size = heat_batch.size()[0]
            heatmap_width = heat_batch.size()[2]
            output_heats_global = []
            for bi in range(0,batch_size):
                trans_2D_inv = labels_raw['trans_2d_inv'][bi].numpy()
                heatmap_bi = heat_batch[bi].numpy().transpose((1, 2, 0)) + 0.2
                heatmap_bi_trans = self.augmentation_test.apply2DImage(trans_2D_inv, heatmap_bi, [256,256]).transpose((2, 0, 1))
                output_heats_global.append(torch.from_numpy(heatmap_bi_trans))
            output_heats_global = torch.stack(output_heats_global)
            output_heats_global_mean = torch.stack([sum(output_heats_global)/batch_size])
            #output_heat = output_heats_global
            ax_label2 = fig.add_subplot(num_subplots_rows,num_subplots_columns,created_sub_plots)
            ax_label2.set_title("2D prediction transformed", size=title_font_size)
            tensor_imshow(ax_label2, torchvision.utils.make_grid( output_heats_global[:,numJoints-3:numJoints,:,:],padding=0))
            ax_label3   = fig.add_subplot(4,3,9)
            ax_label3.set_title("2D prediction averaged", size=title_font_size)
            tensor_imshow(ax_label3, torchvision.utils.make_grid( output_heats_global_mean[:,numJoints-3:numJoints,:,:],padding=0))
   
    # generated image
    if 'img_crop' in outputs_dict.keys():
        images_out  = outputs_dict['img_crop'].cpu().data
        created_sub_plots += 1
        ax_img = fig.add_subplot(num_subplots_rows,num_subplots_columns,created_sub_plots)
        ax_img.set_title("Output images", size=title_font_size)
        grid_t = torchvision.utils.make_grid(images_out, padding=0)
        tensor_imshow_normalized(ax_img, grid_t, mean=config_dict['img_mean'], stdDev=config_dict['img_std'])
        # difference
        images_diff = torch.abs(images_fg-images_out)
        images_diff_max, i = torch.max(images_diff,dim=1,keepdim=True)
        images_diff = images_diff_max.expand_as(images_diff)
        images_diff = images_diff/torch.max(images_diff)
        created_sub_plots += 1
        ax_img = fig.add_subplot(num_subplots_rows,num_subplots_columns,created_sub_plots)
        ax_img.set_title("Output - foreground", size=title_font_size)
        grid_t = torchvision.utils.make_grid(images_diff, padding=0)
        tensor_imshow_normalized(ax_img, grid_t, x_label=x_label)

    if 'shuffled_pose' in outputs_dict.keys():
        shuffle_out = outputs_dict['shuffled_pose'].cpu().data
        shuffle_fg  = outputs_dict['shuffled_appearance'].cpu().data
        created_sub_plots += 1
        ax_img = fig.add_subplot(num_subplots_rows,num_subplots_columns,created_sub_plots)
        ax_img.set_title("shuffling\n3d: {} \nfg: {}".format(shuffle_out.numpy(),shuffle_fg.numpy()), size='4')

    if plot_iol.created_sub_plots_last[keyword] == created_sub_plots: # Don't save the dummy run that determines the number of plots
        plt.savefig(image_name,  dpi=config_dict['dpi'], transparent=True)
        print("Written image to {} at dpi={}".format(image_name, config_dict['dpi']))

    if verbose:
        plt.show()
    plt.close("all")
    plot_iol.created_sub_plots_last[keyword] = created_sub_plots
