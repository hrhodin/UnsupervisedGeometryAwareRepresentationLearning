import torch

class Criterion3DPose_leastQuaresScaled(torch.nn.Module):
    """
    Normalize the scale in the least squares sense, then apply the specified criterion
    """
    def __init__(self, criterion):
        super(Criterion3DPose_leastQuaresScaled, self).__init__()
        self.criterion = criterion

    def forward(self, pred, label):
        #Optimal scale transform
        batch_size = pred.size()[0]
        pred_vec = pred.view(batch_size,-1)
        gt_vec = label.view(batch_size,-1)
        dot_pose_pose = torch.sum(torch.mul(pred_vec,pred_vec),1,keepdim=True)
        dot_pose_gt   = torch.sum(torch.mul(pred_vec,gt_vec),1,keepdim=True)

        s_opt = dot_pose_gt / dot_pose_pose

        return self.criterion.forward(s_opt.expand_as(pred)*pred, label)

class MPJPECriterion(torch.nn.Module):
    """
    Mean per-joint error, assuming joint in interleaved format (x1,y1,z1,x2,y2...)
    """
    def __init__(self, weight=1):
        super(MPJPECriterion, self).__init__()
        self.weight = weight

    def forward(self, pred, label):
        size_orig = pred.size()
        batchSize = size_orig[0]
        diff = pred - label
        diff_sq = torch.mul(diff,diff)

        diff_sq = diff_sq.view((batchSize, -1, 3))  # dimension 2 now spans x,y,z
        diff_3d_len_sq = torch.sum(diff_sq, 2)

        diff_3d_len = torch.sqrt(diff_3d_len_sq)

        #print('diff_3d_len_sq', diff_3d_len_sq.size())

        return self.weight*torch.mean(diff_3d_len);    # mean across batch and joints
