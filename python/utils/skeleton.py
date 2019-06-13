import numpy as np
import numpy.linalg as la
import torch

bones_h36m = [[0, 1], [1, 2], [2, 3],
             [0, 4], [4, 5], [5, 6],
             [0, 7], [7, 8], [8, 9], [9, 10],
             [8, 14], [14, 15], [15, 16],
             [8, 11], [11, 12], [12, 13],
            ]

bones_mpii = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5],
         [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15],
         [6, 8], [8, 9]]
root_index_h36m = 0

bones_hand = [[0, 1], [1, 2], [2, 3], # little finger, starting from tip
              [4, 5], [5, 6], [6, 7], 
              [8, 9], [9, 10], [10, 11], 
              [12, 13], [13, 14], [14, 15], 
              [16, 17], [17, 18], [18, 19]] # thumb
root_index_hand = 19


joint_indices_ski=[    0,             1,          2,           3,            4,          5,           6,       7,     8,      9,        10,        11,            12,         13,         14,             15,         16,         17,         18,      19,        20,   21,   22,   23]
joint_names_ski_meyer_2d = ['head',  'lshoulder',   'lelbow',      'lhand',     'lpole','rshoulder',    'relbow', 'rhand','rpole','lhip',   'lknee',  'lankle',    'lfoottip',    'lheel',   'lskitip',        'rhip',    'rknee',   'rankle', 'rfoottip', 'rheel', 'rskitip', 'b3', 'b4', 'b5']
joint_names_ski_meyer = ['head',      'torso','upper arm right','upper arm left','lower arm right','lower arm left','hand right','hand left','upper leg right','upper leg left','lower leg right','lower leg left','foot right','foot left','pole right','pole left','morpho point',]

joint_names_ski_spoerri = ['HeadGPSAntenna', 'Head', 'Neck', 'RightHip', 'LeftHip', 'RightKnee', 'LeftKnee', 'RightAnkle', 'LeftAnkle', 'RightShoulder', 'LeftShoulder', 'RightElbow', 'LeftElbow', 'RightHand', 'LeftHand', 'RightSkiTip', 'RightSkiTail', 'LeftSkiTip', 'LeftSkiTail', 'RightStickTail', 'LeftStickTail', 'COM_body', 'COM_system']  

joint_indices_h36m=[    0,             1,          2,           3,            4,         5,           6,       7,     8,      9,        10,        11,            12,         13,         14,             15,         16 ]
joint_weights_h36m=[  6.3,          10.0,       10.3,         5.8,         10.0,      10.3,         5.8,       0,   5.0,    7.3,       1.0,      10.0,           2.3,        1.6,       10.0,            2.3,        1.6 ]
joint_names_h36m = ['hip','right_up_leg','right_leg','right_foot','left_up_leg','left_leg', 'left_foot','spine1','neck', 'head','head-top','left-arm','left_forearm','left_hand','right_arm','right_forearm','right_hand']

joint_symmetry_h36m = [[0,0],[1,4],[2,5],[3,6],[7,7],[8,8],[9,9],[10,10],[11,14],[12,15],[13,16]]
joint_torso       = [0,1,4,7,8,11,14]
joint_limbs       = [joint_names_h36m.index('right_leg'),joint_names_h36m.index('right_foot'),joint_names_h36m.index('right_forearm'),joint_names_h36m.index('right_hand'),
                     joint_names_h36m.index('left_leg'),joint_names_h36m.index('left_foot'),joint_names_h36m.index('left_forearm'),joint_names_h36m.index('left_hand')]

bones_h36m_limbSymmetries = [
    [bones_h36m.index([joint_names_h36m.index('left_up_leg'), joint_names_h36m.index('left_leg')]),bones_h36m.index([joint_names_h36m.index('right_up_leg'), joint_names_h36m.index('right_leg')])],
    [bones_h36m.index([joint_names_h36m.index('left_leg'), joint_names_h36m.index('left_foot')]),bones_h36m.index([joint_names_h36m.index('right_leg'), joint_names_h36m.index('right_foot')])],
    [bones_h36m.index([joint_names_h36m.index('left-arm'), joint_names_h36m.index('left_forearm')]),bones_h36m.index([joint_names_h36m.index('right_arm'), joint_names_h36m.index('right_forearm')])],
    [bones_h36m.index([joint_names_h36m.index('left_forearm'), joint_names_h36m.index('left_hand')]),bones_h36m.index([joint_names_h36m.index('right_forearm'), joint_names_h36m.index('right_hand')])],
    ]               

joint_indices_cpm =[    0,             1,          2,           3,            4,         5,           6,       7,     8,      9,        10,        11,            12,         13,         14]
joint_names_cpm =  ['head',        'neck',    'Rsho',      'Relb',       'Rwri',    'Lsho',      'Lelb',  'Lwri','Rhip', 'Rkne',    'Rank',    'Lhip',        'Lkne',     'Lank',     'root']

joint_indices_mpii=[    0,             1,          2,           3,            4,         5,           6,       7,     8,      9,        10,        11,            12,         13,         14,             15,         16 ]
joint_names_mpii=  ['Rank',        'Rkne',    'Rhip',      'Lhip',       'Lkne',    'Lank',      'root',  '????','neck', 'head',    'Rwri',    'Relb',        'Rsho',     'Lsho',     'Lelb',           'Lwri']
joint_symmetry_mpii = [[0,5],[1,4],[2,3], [6,6],[7,7],[8,8],[9,9],[10,15],[11,14],[12,13],[16,16]]

cpm2h36m =         [   14,             8,          9,          10,           11,        12,          13,      14,     1,      0,        0,         5,              6,          7,          2,              3,         4  ]
mpii_to_cpm =      [    9,             8,         12,          11,           10,        13,          14,      15,     2,      1,        0,         3,              4,          5,          6,              7]
mpii_to_h36m = list(np.array(mpii_to_cpm)[cpm2h36m])
h36m_to_mpii = [3, 2, 1, 4, 5, 6, 0, 8, 9, 10, 16, 15, 14, 11, 12, 13]

joint_names_ski_spoerri = ['HeadGPSAntenna', 'Head', 'Neck', 'RightHip', 'LeftHip', 'RightKnee', 'LeftKnee', 'RightAnkle', 'LeftAnkle', 'RightShoulder', 'LeftShoulder', 'RightElbow', 'LeftElbow', 'RightHand', 'LeftHand', 'RightSkiTip', 'RightSkiTail', 'LeftSkiTip', 'LeftSkiTail', 'RightStickTail', 'LeftStickTail', 'COM_body', 'COM_system']  
ski_spoerri_to_h36m     = np.zeros((17,21))
ski_spoerri_to_h36m[joint_names_h36m.index('hip'), joint_names_ski_spoerri.index('LeftHip')] = 0.5
ski_spoerri_to_h36m[joint_names_h36m.index('hip'), joint_names_ski_spoerri.index('RightHip')] = 0.5
ski_spoerri_to_h36m[joint_names_h36m.index('right_up_leg'), joint_names_ski_spoerri.index('LeftHip')] = 0.1 # shift a little, since the marker is at the leg outside
ski_spoerri_to_h36m[joint_names_h36m.index('right_up_leg'), joint_names_ski_spoerri.index('RightHip')] = 0.9
ski_spoerri_to_h36m[joint_names_h36m.index('right_leg'), joint_names_ski_spoerri.index('RightKnee')] = 1
ski_spoerri_to_h36m[joint_names_h36m.index('right_foot'), joint_names_ski_spoerri.index('RightAnkle')] = 1 #should be rheel, but is wirdly placed
ski_spoerri_to_h36m[joint_names_h36m.index('left_up_leg'), joint_names_ski_spoerri.index('LeftHip')] = 0.9
ski_spoerri_to_h36m[joint_names_h36m.index('left_up_leg'), joint_names_ski_spoerri.index('RightHip')] = 0.1
ski_spoerri_to_h36m[joint_names_h36m.index('left_leg'), joint_names_ski_spoerri.index('LeftKnee')] = 1
ski_spoerri_to_h36m[joint_names_h36m.index('left_foot'), joint_names_ski_spoerri.index('LeftAnkle')] = 1 #should be lheel, but is missing
head_hip_factor = 0.4
ski_spoerri_to_h36m[joint_names_h36m.index('spine1'), joint_names_ski_spoerri.index('LeftHip')] = 0.5*(1-head_hip_factor)
ski_spoerri_to_h36m[joint_names_h36m.index('spine1'), joint_names_ski_spoerri.index('RightHip')] = 0.5*(1-head_hip_factor)
ski_spoerri_to_h36m[joint_names_h36m.index('spine1'), joint_names_ski_spoerri.index('Head')] = head_hip_factor
head_hip_factor = 0.8
ski_spoerri_to_h36m[joint_names_h36m.index('neck'), joint_names_ski_spoerri.index('LeftHip')] = 0.5*(1-head_hip_factor)
ski_spoerri_to_h36m[joint_names_h36m.index('neck'), joint_names_ski_spoerri.index('RightHip')] = 0.5*(1-head_hip_factor)
ski_spoerri_to_h36m[joint_names_h36m.index('neck'), joint_names_ski_spoerri.index('Head')] = head_hip_factor
ski_spoerri_to_h36m[joint_names_h36m.index('head'), joint_names_ski_spoerri.index('Head')] = 1
ski_spoerri_to_h36m[joint_names_h36m.index('head-top'), joint_names_ski_spoerri.index('Head')] = 1 # TODO XXX HACK
ski_spoerri_to_h36m[joint_names_h36m.index('left-arm'), joint_names_ski_spoerri.index('LeftShoulder')] = 1
ski_spoerri_to_h36m[joint_names_h36m.index('left_forearm'), joint_names_ski_spoerri.index('LeftElbow')] = 1
ski_spoerri_to_h36m[joint_names_h36m.index('left_hand'), joint_names_ski_spoerri.index('LeftHand')] = 1
ski_spoerri_to_h36m[joint_names_h36m.index('right_arm'), joint_names_ski_spoerri.index('RightShoulder')] = 1
ski_spoerri_to_h36m[joint_names_h36m.index('right_forearm'), joint_names_ski_spoerri.index('RightElbow')] = 1
ski_spoerri_to_h36m[joint_names_h36m.index('right_hand'), joint_names_ski_spoerri.index('RightHand')] = 1

joint_names_roman = ['head_top', 'neck',
                             'right_shoulder', 'right_ellbow', 'right_hand', 'right_pole_basket',
                             'left_shoulder', 'left_ellbow', 'left_hand', 'left_pole_basket',
                             'right_hip', 'right_knee', 'right_ankle',
                             'left_hip', 'left_knee', 'left_ankle',
                             'right_ski_tip', 'right_toes', 'right_heel', 'right_ski_rear',
                             'left_ski_tip', 'left_toes', 'left_heel', 'left_ski_rear']
bones_roman = [[0,1], [1,2], [2,3], [3,4], [4,5], [1,6], [6,7], [7,8], [8,9],
                        [2,10], [10,11], [11,12], [6,13], [13,14], [14,15],
                        [16,17], [17,18], [18,19], [12,17], [12,18],
                        [20,21], [21,22], [22,23], [15,21], [15,22]]
ski_spoerri_to_roman     = np.zeros((24,21))
ski_spoerri_to_roman[joint_names_roman.index('head_top'), joint_names_ski_spoerri.index('Head')] = 1
ski_spoerri_to_roman[joint_names_roman.index('neck'), joint_names_ski_spoerri.index('LeftHip')] = 0.5*(1-head_hip_factor)
ski_spoerri_to_roman[joint_names_roman.index('neck'), joint_names_ski_spoerri.index('RightHip')] = 0.5*(1-head_hip_factor)
ski_spoerri_to_roman[joint_names_roman.index('neck'), joint_names_ski_spoerri.index('Head')] = head_hip_factor
ski_spoerri_to_roman[joint_names_roman.index('right_shoulder'), joint_names_ski_spoerri.index('RightShoulder')] = 1
ski_spoerri_to_roman[joint_names_roman.index('right_ellbow'), joint_names_ski_spoerri.index('RightElbow')] = 1
ski_spoerri_to_roman[joint_names_roman.index('right_hand'), joint_names_ski_spoerri.index('RightHand')] = 1
ski_spoerri_to_roman[joint_names_roman.index('right_pole_basket'), joint_names_ski_spoerri.index('RightStickTail')] = 1
ski_spoerri_to_roman[joint_names_roman.index('left_shoulder'), joint_names_ski_spoerri.index('LeftShoulder')] = 1
ski_spoerri_to_roman[joint_names_roman.index('left_ellbow'), joint_names_ski_spoerri.index('LeftElbow')] = 1
ski_spoerri_to_roman[joint_names_roman.index('left_hand'), joint_names_ski_spoerri.index('LeftHand')] = 1
ski_spoerri_to_roman[joint_names_roman.index('left_pole_basket'), joint_names_ski_spoerri.index('LeftStickTail')] = 1
ski_spoerri_to_roman[joint_names_roman.index('right_hip'), joint_names_ski_spoerri.index('LeftHip')] = 0.1 # shift a little, since the marker is at the leg outside
ski_spoerri_to_roman[joint_names_roman.index('right_hip'), joint_names_ski_spoerri.index('RightHip')] = 0.9
ski_spoerri_to_roman[joint_names_roman.index('right_knee'), joint_names_ski_spoerri.index('RightKnee')] = 1
ski_spoerri_to_roman[joint_names_roman.index('right_ankle'), joint_names_ski_spoerri.index('RightAnkle')] = 1 #should be rheel, but is wirdly placed
ski_spoerri_to_roman[joint_names_roman.index('left_hip'), joint_names_ski_spoerri.index('RightHip')] = 0.1 # shift a little, since the marker is at the leg outside
ski_spoerri_to_roman[joint_names_roman.index('left_hip'), joint_names_ski_spoerri.index('LeftHip')] = 0.9
ski_spoerri_to_roman[joint_names_roman.index('left_knee'), joint_names_ski_spoerri.index('LeftKnee')] = 1
ski_spoerri_to_roman[joint_names_roman.index('left_ankle'), joint_names_ski_spoerri.index('LeftAnkle')] = 1 #should be rheel, but is wirdly placed

ski_spoerri_to_roman[joint_names_roman.index('right_ski_tip'), joint_names_ski_spoerri.index('RightSkiTip')] = 1
ski_spoerri_to_roman[joint_names_roman.index('right_toes'), joint_names_ski_spoerri.index('RightSkiTip')] = 0.55
ski_spoerri_to_roman[joint_names_roman.index('right_toes'), joint_names_ski_spoerri.index('RightSkiTail')] = 0.45
ski_spoerri_to_roman[joint_names_roman.index('right_heel'), joint_names_ski_spoerri.index('RightSkiTip')] = 0.35
ski_spoerri_to_roman[joint_names_roman.index('right_heel'), joint_names_ski_spoerri.index('RightSkiTail')] = 0.65
ski_spoerri_to_roman[joint_names_roman.index('right_ski_rear'), joint_names_ski_spoerri.index('RightSkiTail')] = 1 #should be lheel, but is missing
ski_spoerri_to_roman[joint_names_roman.index('left_ski_tip'), joint_names_ski_spoerri.index('LeftSkiTip')] = 1
ski_spoerri_to_roman[joint_names_roman.index('left_toes'), joint_names_ski_spoerri.index('LeftSkiTip')] = 0.55
ski_spoerri_to_roman[joint_names_roman.index('left_toes'), joint_names_ski_spoerri.index('LeftSkiTail')] = 0.45
ski_spoerri_to_roman[joint_names_roman.index('left_heel'), joint_names_ski_spoerri.index('LeftSkiTip')] = 0.35
ski_spoerri_to_roman[joint_names_roman.index('left_heel'), joint_names_ski_spoerri.index('LeftSkiTail')] = 0.65
ski_spoerri_to_roman[joint_names_roman.index('left_ski_rear'), joint_names_ski_spoerri.index('LeftSkiTail')] = 1 #should be lheel, but is missing

# matrix to map from ski to h36m (inverse to map in the other way)
ski_to_h36m     = np.zeros((17,17))
ski_to_h36m[joint_names_h36m.index('hip'), joint_names_ski_meyer.index('upper leg left')] = 0.5
ski_to_h36m[joint_names_h36m.index('hip'), joint_names_ski_meyer.index('upper leg right')] = 0.5
ski_to_h36m[joint_names_h36m.index('right_up_leg'), joint_names_ski_meyer.index('upper leg left')] = 0.1 # shift a little, since the marker is at the leg outside
ski_to_h36m[joint_names_h36m.index('right_up_leg'), joint_names_ski_meyer.index('upper leg right')] = 0.9
ski_to_h36m[joint_names_h36m.index('right_leg'), joint_names_ski_meyer.index('lower leg right')] = 1
ski_to_h36m[joint_names_h36m.index('right_foot'), joint_names_ski_meyer.index('foot right')] = 1 #should be rheel, but is wirdly placed
ski_to_h36m[joint_names_h36m.index('left_up_leg'), joint_names_ski_meyer.index('upper leg left')] = 0.9
ski_to_h36m[joint_names_h36m.index('left_up_leg'), joint_names_ski_meyer.index('upper leg right')] = 0.1
ski_to_h36m[joint_names_h36m.index('left_leg'), joint_names_ski_meyer.index('lower leg left')] = 1
ski_to_h36m[joint_names_h36m.index('left_foot'), joint_names_ski_meyer.index('foot left')] = 1 #should be lheel, but is missing
head_hip_factor = 0.4
ski_to_h36m[joint_names_h36m.index('spine1'), joint_names_ski_meyer.index('upper leg left')] = 0.5*(1-head_hip_factor)
ski_to_h36m[joint_names_h36m.index('spine1'), joint_names_ski_meyer.index('upper leg right')] = 0.5*(1-head_hip_factor)
ski_to_h36m[joint_names_h36m.index('spine1'), joint_names_ski_meyer.index('head')] = head_hip_factor
head_hip_factor = 0.8
ski_to_h36m[joint_names_h36m.index('neck'), joint_names_ski_meyer.index('upper leg left')] = 0.5*(1-head_hip_factor)
ski_to_h36m[joint_names_h36m.index('neck'), joint_names_ski_meyer.index('upper leg right')] = 0.5*(1-head_hip_factor)
ski_to_h36m[joint_names_h36m.index('neck'), joint_names_ski_meyer.index('head')] = head_hip_factor
#ski_to_h36m[joint_names_h36m.index('neck'), joint_names_ski_meyer.index('head')] = 1
ski_to_h36m[joint_names_h36m.index('head'), joint_names_ski_meyer.index('head')] = 1
ski_to_h36m[joint_names_h36m.index('head-top'), joint_names_ski_meyer.index('head')] = 1 # TODO XXX HACK
ski_to_h36m[joint_names_h36m.index('left-arm'), joint_names_ski_meyer.index('upper arm left')] = 1
ski_to_h36m[joint_names_h36m.index('left_forearm'), joint_names_ski_meyer.index('lower arm left')] = 1
ski_to_h36m[joint_names_h36m.index('left_hand'), joint_names_ski_meyer.index('hand left')] = 1
ski_to_h36m[joint_names_h36m.index('right_arm'), joint_names_ski_meyer.index('upper arm right')] = 1
ski_to_h36m[joint_names_h36m.index('right_forearm'), joint_names_ski_meyer.index('lower arm right')] = 1
ski_to_h36m[joint_names_h36m.index('right_hand'), joint_names_ski_meyer.index('hand right')] = 1

ski_to_h36m_meyer_2d     = np.zeros((17,20))
ski_to_h36m_meyer_2d[joint_names_h36m.index('hip'), joint_names_ski_meyer_2d.index('lhip')] = 0.5
ski_to_h36m_meyer_2d[joint_names_h36m.index('hip'), joint_names_ski_meyer_2d.index('rhip')] = 0.5
ski_to_h36m_meyer_2d[joint_names_h36m.index('right_up_leg'), joint_names_ski_meyer_2d.index('lhip')] = 0.1 # shift a little, since the marker is at the leg outside
ski_to_h36m_meyer_2d[joint_names_h36m.index('right_up_leg'), joint_names_ski_meyer_2d.index('rhip')] = 0.9
ski_to_h36m_meyer_2d[joint_names_h36m.index('right_leg'), joint_names_ski_meyer_2d.index('rknee')] = 1
ski_to_h36m_meyer_2d[joint_names_h36m.index('right_foot'), joint_names_ski_meyer_2d.index('rankle')] = 1 #should be rheel, but is wirdly placed
ski_to_h36m_meyer_2d[joint_names_h36m.index('left_up_leg'), joint_names_ski_meyer_2d.index('lhip')] = 0.9
ski_to_h36m_meyer_2d[joint_names_h36m.index('left_up_leg'), joint_names_ski_meyer_2d.index('rhip')] = 0.1
ski_to_h36m_meyer_2d[joint_names_h36m.index('left_leg'), joint_names_ski_meyer_2d.index('lknee')] = 1
ski_to_h36m_meyer_2d[joint_names_h36m.index('left_foot'), joint_names_ski_meyer_2d.index('lankle')] = 1 #should be lheel, but is missing
# ski_to_h36m_meyer_2d[joint_names_h36m.index('right_up_leg'), joint_names_ski_meyer_2d.index('rknee')] = 1
# ski_to_h36m_meyer_2d[joint_names_h36m.index('right_leg'), joint_names_ski_meyer_2d.index('rankle')] = 1
# ski_to_h36m_meyer_2d[joint_names_h36m.index('right_foot'), joint_names_ski_meyer_2d.index('rfoottip')] = 1 #should be rheel, but is wirdly placed
# ski_to_h36m_meyer_2d[joint_names_h36m.index('left_up_leg'), joint_names_ski_meyer_2d.index('lknee')] = 1
# ski_to_h36m_meyer_2d[joint_names_h36m.index('left_leg'), joint_names_ski_meyer_2d.index('lankle')] = 1
# ski_to_h36m_meyer_2d[joint_names_h36m.index('left_foot'), joint_names_ski_meyer_2d.index('lfoottip')] = 1 #should be lheel, but is missing
head_hip_factor = 0.4
ski_to_h36m_meyer_2d[joint_names_h36m.index('spine1'), joint_names_ski_meyer_2d.index('lhip')] = 0.5*(1-head_hip_factor)
ski_to_h36m_meyer_2d[joint_names_h36m.index('spine1'), joint_names_ski_meyer_2d.index('rhip')] = 0.5*(1-head_hip_factor)
ski_to_h36m_meyer_2d[joint_names_h36m.index('spine1'), joint_names_ski_meyer_2d.index('head')] = head_hip_factor
head_hip_factor = 0.8
ski_to_h36m_meyer_2d[joint_names_h36m.index('neck'), joint_names_ski_meyer_2d.index('lhip')] = 0.5*(1-head_hip_factor)
ski_to_h36m_meyer_2d[joint_names_h36m.index('neck'), joint_names_ski_meyer_2d.index('rhip')] = 0.5*(1-head_hip_factor)
ski_to_h36m_meyer_2d[joint_names_h36m.index('neck'), joint_names_ski_meyer_2d.index('head')] = head_hip_factor
#ski_to_h36m_meyer_2d[joint_names_h36m.index('neck'), joint_names_ski_meyer_2d.index('head')] = 1
ski_to_h36m_meyer_2d[joint_names_h36m.index('head'), joint_names_ski_meyer_2d.index('head')] = 1
ski_to_h36m_meyer_2d[joint_names_h36m.index('head-top'), joint_names_ski_meyer_2d.index('head')] = 1 # TODO XXX HACK
ski_to_h36m_meyer_2d[joint_names_h36m.index('left-arm'), joint_names_ski_meyer_2d.index('lshoulder')] = 1
ski_to_h36m_meyer_2d[joint_names_h36m.index('left_forearm'), joint_names_ski_meyer_2d.index('lelbow')] = 1
ski_to_h36m_meyer_2d[joint_names_h36m.index('left_hand'), joint_names_ski_meyer_2d.index('lhand')] = 1
ski_to_h36m_meyer_2d[joint_names_h36m.index('right_arm'), joint_names_ski_meyer_2d.index('rshoulder')] = 1
ski_to_h36m_meyer_2d[joint_names_h36m.index('right_forearm'), joint_names_ski_meyer_2d.index('relbow')] = 1
ski_to_h36m_meyer_2d[joint_names_h36m.index('right_hand'), joint_names_ski_meyer_2d.index('rhand')] = 1

def computeBoneLengths(pose_tensor, bones):
    pose_tensor_3d = pose_tensor.view(-1,3)
    length_list = [torch.norm(pose_tensor_3d[bone[0]] - pose_tensor_3d[bone[1]])
                     for bone in bones]
    return length_list

def computeBoneLengths_np(pose_tensor, bones):
    pose_tensor_3d = pose_tensor.reshape(-1,3)
    length_list = [la.norm(pose_tensor_3d[bone[0]] - pose_tensor_3d[bone[1]])
                     for bone in bones]
    return length_list
