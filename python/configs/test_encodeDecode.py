import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt

import torch
torch.cuda.current_device() # to prevent  "Cannot re-initialize CUDA in forked subprocess." error on some configurations
import numpy as np
import numpy.linalg as la
import IPython

from utils import io as utils_io
from utils import datasets as utils_data
from utils import plotting as utils_plt
from utils import skeleton as utils_skel

import train_encodeDecode
from ignite._utils import convert_tensor
from ignite.engine import Events

from matplotlib.widgets import Slider, Button

# load data
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

class IgniteTestNVS(train_encodeDecode.IgniteTrainNVS):
    def run(self, config_dict_file, config_dict):
        config_dict['n_hidden_to3Dpose'] = config_dict.get('n_hidden_to3Dpose', 2)


        if 1: # load small example data
            import pickle
            data_loader = pickle.load(open('../examples/test_set.pickl',"rb"))
        else:
            data_loader = self.load_data_test(config_dict)
            # save example data
            if 0:
                import pickle
                IPython.embed()
                data_iterator = iter(data_loader)
                data_cach = [next(data_iterator) for i in range(10)]
                data_cach = tuple(data_cach)
                pickle.dump(data_cach, open('../examples/test_set.pickl', "wb"))

        # load model
        model = self.load_network(config_dict)
        model = model.to(device)

        def tensor_to_npimg(torch_array):
            return np.swapaxes(np.swapaxes(torch_array.numpy(), 0, 2), 0, 1)

        def denormalize(np_array):
            return np_array * np.array(config_dict['img_std']) + np.array(config_dict['img_mean'])

        # extract image
        def tensor_to_img(output_tensor):
            output_img = tensor_to_npimg(output_tensor)
            output_img = denormalize(output_img)
            output_img = np.clip(output_img, 0, 1)
            return output_img

        def rotationMatrixXZY(theta, phi, psi):
            Ax = np.matrix([[1, 0, 0],
                            [0, np.cos(theta), -np.sin(theta)],
                            [0, np.sin(theta), np.cos(theta)]])
            Ay = np.matrix([[np.cos(phi), 0, -np.sin(phi)],
                            [0, 1, 0],
                            [np.sin(phi), 0, np.cos(phi)]])
            Az = np.matrix([[np.cos(psi), -np.sin(psi), 0],
                            [np.sin(psi), np.cos(psi), 0],
                            [0, 0, 1], ])
            return Az * Ay * Ax

        # get next image
        input_dict, label_dict = None, None
        data_iterator = iter(data_loader)
        def nextImage():
            nonlocal input_dict, label_dict
            input_dict, label_dict = next(data_iterator)
            input_dict['external_rotation_global'] = torch.from_numpy(np.eye(3)).float().to(device)
        nextImage()


        # apply model on images
        output_dict = None
        def predict():
            nonlocal output_dict
            model.eval()
            with torch.no_grad():
                input_dict_cuda, label_dict_cuda = utils_data.nestedDictToDevice((input_dict, label_dict), device=device)
                output_dict_cuda = model(input_dict_cuda)
                output_dict = utils_data.nestedDictToDevice(output_dict_cuda, device='cpu')
        predict()

        # init figure
        my_dpi = 400
        fig, ax_blank = plt.subplots(figsize=(5 * 800 / my_dpi, 5 * 300 / my_dpi))
        plt.axis('off')
        # gt skeleton
        ax_gt_skel = fig.add_subplot(111, projection='3d')
        ax_gt_skel.set_position([0.8, 0.0, 0.2, 0.98])
        handle_gt_skel = utils_plt.plot_3Dpose_simple(ax_gt_skel, label_dict['3D'][0].numpy().reshape([-1, 3]).T,
                                                       bones=utils_skel.bones_h36m, linewidth=5,
                                                       plot_handles=None)  # , colormap='Greys')
        ax_gt_skel.invert_zaxis()
        ax_gt_skel.grid(False)
        ax_gt_skel.set_axis_off()
        ax_gt_skel.set_title("GT pose")
        # output skeleton
        ax_pred_skel = fig.add_subplot(111, projection='3d')
        ax_pred_skel.set_position([0.65, 0.0, 0.2, 0.98])
        handle_pred_skel = utils_plt.plot_3Dpose_simple(ax_pred_skel, label_dict['3D'][0].numpy().reshape([-1, 3]).T,
                                                       bones=utils_skel.bones_h36m, linewidth=5,
                                                       plot_handles=None)  # , colormap='Greys')
        ax_pred_skel.invert_zaxis()
        ax_pred_skel.grid(False)
        ax_pred_skel.set_axis_off()
        ax_pred_skel.set_title("Pred. pose")
        # input image
        ax_in_img = plt.axes([-0.16, 0.2, 0.7, 0.7])
        ax_in_img.axis('off')
        im_input = plt.imshow(tensor_to_img(input_dict['img_crop'][0]), animated=True)
        ax_in_img.set_title("Input img")
        # output image
        ax_out_img = plt.axes([0.15, 0.2, 0.7, 0.7])
        ax_out_img.axis('off')
        im_pred = plt.imshow(tensor_to_img(output_dict['img_crop'][0]), animated=True)
        ax_out_img.set_title("Output img")

        # update figure with new data
        def update_figure():
            # images
            im_input.set_array(tensor_to_img(input_dict['img_crop'][0]))
            im_pred.set_array(tensor_to_img(output_dict['img_crop'][0]))
            # gt 3D poses
            gt_pose = label_dict['3D'][0]
            R_cam_2_world = label_dict['extrinsic_rot_inv'][0].numpy()
            R_world_in_cam = la.inv(R_cam_2_world) @ input_dict['external_rotation_global'].cpu().numpy() @ R_cam_2_world
            pose_rotated = R_world_in_cam @ gt_pose.numpy().reshape([-1, 3]).T
            utils_plt.plot_3Dpose_simple(ax_gt_skel, pose_rotated, bones=utils_skel.bones_h36m,
                                         plot_handles=handle_gt_skel)
            # prediction 3D poses
            pose_mean = label_dict['pose_mean'][0].numpy()
            pose_std = label_dict['pose_std'][0].numpy()
            pred_pose = (output_dict['3D'][0].numpy().reshape(pose_mean.shape) * pose_std) + pose_mean
            pose_rotated = R_world_in_cam @ pred_pose.reshape([-1, 3]).T
            utils_plt.plot_3Dpose_simple(ax_pred_skel, pose_rotated, bones=utils_skel.bones_h36m,
                                         plot_handles=handle_pred_skel)

            # flush drawings
            fig.canvas.draw_idle()

        def update_rotation(event):
            rot = slider_yaw_glob.val
            print("Rotationg ",rot)
            batch_size = input_dict['img_crop'].size()[0]
            input_dict['external_rotation_global'] = torch.from_numpy(rotationMatrixXZY(theta=0, phi=0, psi=rot)).float().to(device)
            input_dict['external_rotation_cam'] = torch.from_numpy(np.eye(3)).float().to(device) # torch.from_numpy(rotationMatrixXZY(theta=0, phi=rot, psi=0)).float().cuda()
            predict()
            update_figure()

        ax_next = plt.axes([0.05, 0.1, 0.15, 0.04])
        button_next = Button(ax_next, 'Next image', color='lightgray', hovercolor='0.975')
        def nextButtonPressed(event):
            nextImage()
            predict()
            update_figure()
        button_next.on_clicked(nextButtonPressed)
        ax_yaw_glob = plt.axes([0.25, 0.1, 0.65, 0.015], facecolor='lightgray')
        slider_range = 2 * np.pi
        slider_yaw_glob = Slider(ax_yaw_glob, 'Yaw', -slider_range, slider_range, valinit=0)
        slider_yaw_glob.on_changed(update_rotation)
        plt.show()

if __name__ == "__main__":
    config_dict_module = utils_io.loadModule("configs/config_test_encodeDecode.py")
    config_dict = config_dict_module.config_dict
    ignite = IgniteTestNVS()
    ignite.run(config_dict_module.__file__, config_dict)
