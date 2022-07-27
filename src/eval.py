from common.skeleton import Skeleton
from utils import net_utils
from rhythmnet.net_initiator import net_instance
import itertools
import torch
import numpy as np
from rhythmdata import data_loader
from utils.config import Config 
import ipdb
from code1.dualquats import currentdq2localquats
import json
import torch.nn.functional as F

def compute_accel(features):
    """
    Borrowed from: https://github.com/mkocabas/VIBE/blob/master/lib/utils/eval_utils.py
    Computes acceleration of 3D joints.
    Args:
        joints (BxNxJx3).
    Returns:
        Accelerations loss () int.
    """

    batch = features.shape[0]
    # markers = markers.reshape(-1, markers.shape[0], 67, 3)
    total_accel = 0
    for m_s in features:
        velocities = m_s[1:] - m_s[:-1]
        acceleration = velocities[1:] - velocities[:-1]
        acceleration_normed = np.linalg.norm(acceleration, axis=2)
        
        sample_accel = np.mean(acceleration_normed, axis=1)
        total_accel += np.mean(sample_accel)
    return total_accel/batch


def rot6d_to_rotmat(x):
    """
    Borrowed from: https://github.com/nkolot/SPIN/blob/master/utils/geometry.py
    Adapted by Nefeli Andreou
    Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
#             x = x.reshape(-1,3,2) #changed compared to https://github.com/nkolot/SPIN/blob/master/utils/geometry.py
    a1 = x[:, 0:3] #changed compared to https://github.com/nkolot/SPIN/blob/master/utils/geometry.py
    a2 = x[:, 3:6] #changed compared to https://github.com/nkolot/SPIN/blob/master/utils/geometry.py
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)
def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    '''
    This function is borrowed from https://github.com/kornia/kornia
    Convert 3x4 rotation matrix to 4d quaternion vector
    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201
    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.
    Return:
        Tensor: the rotation in quaternion
    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`
    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    '''
    if rotation_matrix.shape[1:] == (3, 3):
        rotation_matrix = rotation_matrix.reshape(-1, 3, 3)
        hom = torch.tensor([0, 0, 1], dtype=torch.float32,
                            device=rotation_matrix.device).reshape(1, 3, 1).expand(rotation_matrix.shape[0], -1, -1)
        rotation_matrix = torch.cat([rotation_matrix, hom], dim=-1)

    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q
def _copysign(a, b):
    """
    Borrowed from: https://github.com/Mathux/ACTOR/blob/master/src/utils/rotation_conversions.py
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.
    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.
    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def _sqrt_positive_part(x):
    """
    Borrowed from: https://github.com/Mathux/ACTOR/blob/master/src/utils/rotation_conversions.py
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret
    
def matrix_to_quaternion(matrix):
    """
    Borrowed from: https://github.com/Mathux/ACTOR/blob/master/src/utils/rotation_conversions.py
    Convert rotations given as rotation matrices to quaternions.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    return torch.stack((o0, o1, o2, o3), -1)

def fast_npss(gt_seq, pred_seq):
    """
    Borrowed from: https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/benchmarks.py
    Computes Normalized Power Spectrum Similarity (NPSS).
    This is the metric proposed by Gropalakrishnan et al (2019).
    This implementation uses numpy parallelism for improved performance.
    :param gt_seq: ground-truth array of shape : (Batchsize, Timesteps, Dimension)
    :param pred_seq: shape : (Batchsize, Timesteps, Dimension)
    :return: The average npss metric for the batch
    """
    # Fourier coefficients along the time dimension
    gt_fourier_coeffs = np.real(np.fft.fft(gt_seq, axis=1))
    pred_fourier_coeffs = np.real(np.fft.fft(pred_seq, axis=1))
    
    # Square of the Fourier coefficients
    gt_power = np.square(gt_fourier_coeffs)
    pred_power = np.square(pred_fourier_coeffs)

    # Sum of powers over time dimension
    gt_total_power = np.sum(gt_power, axis=1)
    pred_total_power = np.sum(pred_power, axis=1)
    
    # Normalize powers with totals
    if (gt_total_power==0).any():
        # print("WARNING: adding eps in fast_npss to avoid division by zeros")
        gt_norm_power = gt_power / (gt_total_power[:, np.newaxis, :] + 1e-12)
        pred_norm_power = pred_power / (pred_total_power[:, np.newaxis, :] +1e-12)
    else: 
        gt_norm_power = gt_power / gt_total_power[:, np.newaxis, :]
        pred_norm_power = pred_power / pred_total_power[:, np.newaxis, :]
    
    # Cumulative sum over time
    cdf_gt_power = np.cumsum(gt_norm_power, axis=1)
    cdf_pred_power = np.cumsum(pred_norm_power, axis=1)

    # Earth mover distance
    emd = np.linalg.norm((cdf_pred_power - cdf_gt_power), ord=1, axis=1)

    # Weighted EMD
    tmp = (cdf_pred_power - cdf_gt_power)
    power_weighted_emd = np.average(emd, weights=gt_total_power)

    return power_weighted_emd


#--------------------------------------------------------
net_confs =  [
# '/DATA/Projects/2020/lstm_pro/results/001_quatspos31/run.json',
'/DATA/Projects/2020/lstm_pro/results/002_dq/run.json',
'/DATA/Projects/2020/lstm_pro/results/002__/run.json',
'/DATA/Projects/2020/lstm_pro/results/004_ortho6Dpos/run.json',
'/DATA/Projects/2020/lstm_pro/results/004/run.json',
'/DATA/Projects/2020/lstm_pro/results/007/run.json',
# '/DATA/Projects/2020/lstm_pro/results/007_FK2/run.json',
'/DATA/Projects/2020/lstm_pro/results/007_FK3/run.json'
# '/DATA/Projects/2020/lstm_pro/results/007_final/run.json',
# '/DATA/Projects/2020/lstm_pro/results/007_FK/run.json'
]

params = []
weights = [ 
# '0010000', 
# '0020000', 
# '0030000', 
# '0040000', 
# '0050000',
# '0070000',
# '0100000',
'0150000'
]
allw= {}
for w in weights:
    allw[w] = []
for w in weights:
    for net_conf in net_confs:

        model = net_instance(net_conf).model
        model.conf.test_conf.read_weights_path = "/".join(model.conf.test_conf.read_weights_path.split("/")[:-1]) + f'/{w}.weight'
        model.iteration = net_utils.load_model_state(model.model, model.conf.test_conf.read_weights_path)

        dataset = model.dataset
        params.append(model.model.parameters())

        data = data_loader.DanceDataLoader(dataset, "test")
        gt = []
        pred = []
        
        p=0
        S = 100
        BS = 32 
        B = len(dataset.starting_indices['test'])//BS * BS
        J = 31

        sl = 1
        model.model.eval()
        with torch.no_grad():
            for i,batch in zip(range(len(dataset.starting_indices['test'])//BS),data):                    
                ## for predictions same as train time 
                # gt.append(model.dataset.de_norm(batch['gt_motion'].reshape(-1,batch['gt_motion'].shape[-1])).reshape(batch['gt_motion'].shape))
#                     aclstm_out = model.dataset.de_norm(model.model.forward(batch).reshape(-1,batch['gt_motion'].shape[-1])).reshape(batch['gt_motion'].shape).cpu().detach().numpy()
#                     pred.append(aclstm_out)


                ## for predictions same as test time
                for j in range(BS):
                    ps = batch['motion_name'][j]
                    frame_start = batch['start'][j]
                    frame_end = frame_start + 10
                    ff = 30
                    name = None
                    push_seq = model.dataset.get_push_sequence(clip_name=ps,
                                                          part='test',
                                                          frame_start=frame_start,
                                                          frame_end=frame_end)
                    outt,gtt = model.test_specific_input_dq(motion_clip_name=ps,
                                            motion_db_part='test',
                                            start_motion_frame=frame_start,
                        end_motion_frame=frame_end+ff,
                               push_sequence = push_seq, validation_run=False,names=name)
                    S = outt.shape[0]
                    if gtt.shape[0]!=outt.shape[0]:
                        continue
                    else:
                        gt.append(gtt[None])
                        pred.append(outt[None])
 
       
        pred = np.vstack(pred)
        gt = np.vstack(gt)
 
        B = len(pred)
        parh = [None, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 15, 13, 17, 18, 19, 20, 21, 20, 13, 24, 25, 26, 27, 28, 27]

        sk = Skeleton( 
                    offsets=model.conf.data_conf.offsets
                    , parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 15,
                               13, 17, 18, 19, 20, 21, 20, 13, 24, 25, 26, 27, 28, 27],
                    )


        if '002' in net_conf:
  
            q_gt = []
            q_pred = []
      
            # convert to local quats - remove root
            for i in gt[:,:,3:]:
                q_gt.append(currentdq2localquats(i,parh,31)[None])
            for i in pred[:,:,3:]:
                q_pred.append(currentdq2localquats(i,parh,31)[None])
    
   
            ipdb.set_trace()
            q_gt = np.vstack(q_gt)
            q_pred = np.vstack(q_pred)

            # get positions
            pos_gt = sk.forward_kinematics(torch.FloatTensor(q_gt.reshape(B,S,31,4)).cuda(), torch.zeros(B,S,3).cuda()*1e-10)
            pos_pred = sk.forward_kinematics(torch.FloatTensor(q_pred.reshape(B,S,31,4)).cuda(), torch.zeros(B,S,3).cuda()*1e-10)

        elif '004_ortho6Dpos'  in net_conf or '004' in net_conf: 
            # remove root and positions
            
            dims = 31*6+3
     
            gt_2 = gt[:,:,3:dims]
            pred_2 =  pred[:,:,3:dims]
        
            # ortho6D to quats
            mat_gt = rot6d_to_rotmat(torch.FloatTensor(gt_2).reshape(-1,6))
            mat_pred = rot6d_to_rotmat(torch.FloatTensor(pred_2).reshape(-1,6))
            q_gt = matrix_to_quaternion(mat_gt).reshape(B,S,-1)
            q_pred = matrix_to_quaternion(mat_pred).reshape(B,S,-1)

            # get positions
            pos_gt = sk.forward_kinematics(q_gt.reshape(B,S,31,4).cuda(), torch.zeros(B,S,3).cuda()*1e-10)
            pos_pred = sk.forward_kinematics(q_pred.reshape(B,S,31,4).cuda(), torch.zeros(B,S,3).cuda()*1e-10)
    
    
    

        elif '001' in net_conf or '007' in net_conf:
            # remove root and positions
            dims = 31*4+3

            gt_2 = gt[:,:,3:dims]
            pred_2 =  pred[:,:,3:dims]
            
            # normalize if not nornalized
       
            pred_2 = torch.nn.functional.normalize(torch.tensor(pred_2.reshape(-1,4))).reshape(pred_2.shape).cpu().detach().numpy()
    
            # get positions
            pos_gt = sk.forward_kinematics(torch.FloatTensor(gt_2.reshape(B,S,31,4)).cuda(), torch.zeros((B,S,3),dtype=torch.float).cuda()*1e-10)
            pos_pred = sk.forward_kinematics(torch.FloatTensor(pred_2.reshape(B,S,31,4)).cuda(), torch.zeros((B,S,3),dtype=torch.float).cuda()*1e-10)
        
        # calculate euclidean distance
        loss = torch.mean((pos_gt.reshape(B,S,31,3) - pos_pred.reshape(B,S,31,3)).norm(dim=3))

        # calculate acceleration 
        print(w, net_conf.split("/")[-2], "accel_pred", compute_accel(pos_pred.cpu().detach().numpy()))
        print(w, net_conf.split("/")[-2], "accel_pred", compute_accel(pos_gt.cpu().detach().numpy()))
        print(model.iteration, net_conf.split("/")[-2], "loss", loss.item())
        print(model.iteration, net_conf.split("/")[-2], "npss", fast_npss(pos_gt.cpu().detach().numpy(), pos_pred.cpu().detach().numpy()))
#             allw[w].append((net_conf, loss.item()))


    


with open('euclidean_loss.json', 'w') as f:
    json.dump(allw, f)
exit()
