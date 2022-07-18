import numpy as np
import quaternion
from DualQuaternion2 import DualQuaternion
import torch
from common.quaternion import qmul, qeuler, qeuler_np
import torch.nn as nn
import torch.nn.functional as F
Quaternion = quaternion.quaternion

def localquats2currentdq(lq, offsets, parh,joints_num = 32):
    """takes in local quaternion, offsets, and parents to produce hierarchy-aware dual quaternions

    inputs
    ------
    lq: array of local quaternions, size: (T,J*4) (#frames x (number of joints used*4))
    offsets: array, size: #joints used x 3
    parh: parents list , for us size = 31 (I think for MotioNet - the joints that the network predicts)


    outputs
    -------
    allcq: current dual quaternions for each joint, size: #frames x (#joints used *8)
    """
    allcq = []
    # print(parh)
    for ff in range(lq.shape[0]):
        cq = {}
        for i in range(joints_num):
            if i == 0:
                cq[i] = DualQuaternion.from_quat_pose_array(
                    list(lq[ff, i * 4:i * 4 + 4]) + list(offsets[0])).normalized().dq_array()
            else:
                cq[i] = (DualQuaternion.from_dq_array((cq[parh[i]])) * DualQuaternion.from_quat_pose_array(
                    list(lq[ff, i * 4:i * 4 + 4]) + list(offsets[i]))).normalized().dq_array()
        temp = []
        for i in cq.items():
            temp.append(i[1][0])
            temp.append(i[1][1])
            temp.append(i[1][2])
            temp.append(i[1][3])
            temp.append(i[1][4])
            temp.append(i[1][5])
            temp.append(i[1][6])
            temp.append(i[1][7])
        allcq.append(temp)
    allcq = np.array(allcq)
    return allcq


def currentdq2localquats(cq, parh,joints_num = 32):
    """takes in current dual quaternions, hierarchy and parents to produce local quaternions

    inputs
    ------
    cq: current dual quaternions for each joint, size: (TxJ*4) #frames x (#joints used *8)
    parhh: parents list, for us size = 31 (I think for MotioNet - the joints that the network predicts)
    parh: parents list (same size as h)


    outputs
    -------
    local quaternions, size #frames x (#predicted joints * 4)- used to create BVH
    """
    alllq = []

    for ff in range(cq.shape[0]):
        llq = {}

        for i in range(joints_num):
            if i == 0:
                llq[i] = list(
                    quaternion.as_float_array(DualQuaternion.from_dq_array(cq[ff, i * 8:i * 8 + 8]).q_r.normalized()))

            else:
                llq[i] =quaternion.as_float_array(
                    (DualQuaternion.from_dq_array(cq[ff, parh[i] * 8:parh[i] * 8 + 8]).q_r.normalized().inverse()) * \
                    (DualQuaternion.from_dq_array(cq[ff, i * 8:i * 8 + 8]).q_r.normalized()))
        temp = []
        for i in llq.items():
            temp.append(i[1][0])
            temp.append(i[1][1])
            temp.append(i[1][2])
            temp.append(i[1][3])

        alllq.append(temp)
    alllq = np.array(alllq)
    return alllq


# 2. Rotational loss
# a. directly applied on the current quaternion
def qrl_curr(out_seq, groundtruth_seq, batch_sz=32, joints_num=31, order='zyx'):  # groundtruth_seq 32,25100(251*100)
    """calculates quaternion rotational loss

    inputs
    -------
    out_seq: generated frames (if not in the format (T, J, 8)#frames x (#joints x 8) need to convert to that)
    groundtruth_seq: groundtruth frames (if not in the format #frames x (#joints x 8) need to convert to that)
    batch_sz: int, batch size
    joints_num: int, #predicted joints
    order: BVH order (zyx for CMU)

    outputs
    -------
     scalar, loss"""

    rotloss = 0

    ##This implementation assumes that the network outputs batch_size x (#joints*4)

    predicted_quats = out_seq.view(batch_sz, joints_num * 2, 4)[:, ::2]
    expeected_euler = groundtruth_seq

    predicted_euler = qeuler(predicted_quats, order)


    angle_distance = torch.remainder(predicted_euler - expeected_euler + np.pi, 2 * np.pi) - np.pi

    return torch.mean(torch.abs(angle_distance)).item()

# b. applied on local quaternions
def qrl(out_seq, groundtruth_seq, batch_sz=32, joints_num=31, parh=None,
        order='zyx'):  # groundtruth_seq 32,25100(251*100)
    """calculates quaternion rotational loss

    inputs
    -------
    out_seq: generated frames (if not in the format (T, J, 8) #frames x (#joints x 8) need to convert to that)
    groundtruth_seq: groundtruth frames (if not in the format #frames x (#joints x 8) need to convert to that)
    batch_sz: int, batch size
    joints_num: int, #predicted joints
    parh:list, parent hierarchy in order that joints appear (size: joints_num)
    order: BVH order (zyx for CMU)

    outputs
    -------
    scalar, loss"""
    predicted_quats= out_seq.view(batch_sz, out_seq.shape[1],joints_num , 8)[:, :,:,:4]
    expected_euler = groundtruth_seq
    parent_quats = predicted_quats[:, :, parh, :]
    parent_quats[:, :, 0] = torch.tensor([1, 0, 0, 0])
    predicted_euler = qeuler(qmul(qconj(
        parent_quats.reshape(parent_quats.shape[0] * parent_quats.shape[1] * parent_quats.shape[2], 4)).reshape(
        parent_quats.shape), predicted_quats), order = "zyx", epsilon=1e-6)
    angle_distance = torch.remainder(predicted_euler - expected_euler + np.pi, 2 * np.pi) - np.pi
    return torch.mean(torch.abs(angle_distance))


# Positional and Bone
def dl(out_seq, groundtruth_seq, batch_sz=32, joints_num=31, parh=None, bl=True):
    """
    calculates positional loss

    inputs
    --------
    out_seq: generated from network, has size (B,J,8) batch x (#joints *8)
    groundtruth_seq: groundtruth values, has size batch x (#joints *8)
    batch_sz: int, batch size
    joints_num: int, # joints to predict
    parh:list, parent hierarchy in order that joints appear (size: joints_num)
    bl: boolean, include bone constrain or not (recommended:True)

    outputs
    --------
    scalar, positional loss
    """
    batchloss_bl = 0
    batchloss = 0

    groundtruth_seq = groundtruth_seq.reshape(batch_sz, joints_num, 8)
    out_seq = out_seq.reshape(batch_sz, joints_num, 8)

    for f_gt, f_out in zip(groundtruth_seq, out_seq):
        posloss = 0
        boneloss = 0
        fout = f_out
        fgt = f_gt
        t_out = [DualQuaternion.from_dq_array(i).normalized().translation() for i in fout]
        t_gt = [DualQuaternion.from_dq_array(i).normalized().translation() for i in fgt]

        loss_function = nn.MSELoss()
        posloss = posloss + loss_function(torch.Tensor(t_out), torch.Tensor(t_gt))

        if bl:
            t_par_out = [t_out[0]] + [t_out[idx] for idx in parh[1:]]
            t_par_gt = [t_out[0]] + [t_gt[idx] for idx in parh[1:]]


            bones_out = [np.linalg.norm(i) for i in np.array(t_par_out) - np.array(t_out)]
            bones_gt = [np.linalg.norm(i) for i in np.array(t_par_gt) - np.array(t_gt)]
            boneloss = boneloss + loss_function(torch.Tensor(bones_out), torch.Tensor(bones_gt))
            #             assert np.allclose(bones_out,[np.linalg.norm(i) for i in offsets]) #for testing purposes
            #             assert np.allclose(bones_gt,[np.linalg.norm(i) for i in offsets])  #for testing purposes
            batchloss_bl += boneloss.item()

        batchloss += posloss.item()

    loss = batchloss / batch_sz
    loss_bl = batchloss_bl / batch_sz
    return loss, loss_bl
    
def dl2(out_seq, groundtruth_seq, batch_sz=32, joints_num=31, parh=None, bl=True):
    """
    calculates positional loss

    inputs
    --------
    out_seq: generated from network, has size (B,J,8) batch x (#joints *8)
    groundtruth_seq: groundtruth values, has size batch x (#joints *8)
    batch_sz: int, batch size
    joints_num: int, # joints to predict
    parh:list, parent hierarchy in order that joints appear (size: joints_num)
    bl: boolean, include bone constrain or not (recommended:True)

    outputs
    --------
    scalar, positional loss
    """
    batchloss_bl = 0
    batchloss = 0

    groundtruth_seq = groundtruth_seq.reshape(batch_sz, joints_num, 8)
    out_seq = out_seq.reshape(batch_sz, joints_num, 8)

    for f_gt, f_out in zip(groundtruth_seq, out_seq):
        posloss = 0
        boneloss = 0
        fout = f_out
        fgt = f_gt
        t_out = [DualQuaternion.from_dq_array(i).normalized().translation() for i in fout]
        t_gt = [DualQuaternion.from_dq_array(i).normalized().translation() for i in fgt]

        loss_function = nn.MSELoss()
        posloss = posloss + loss_function(torch.Tensor(t_out), torch.Tensor(t_gt))

        if bl:
            t_par_out = [t_out[0]] + [t_out[idx] for idx in parh[1:]]
            t_par_gt = [t_out[0]] + [t_gt[idx] for idx in parh[1:]]


            bones_out = [np.linalg.norm(i) for i in np.array(t_par_out) - np.array(t_out)]
            bones_gt = [np.linalg.norm(i) for i in np.array(t_par_gt) - np.array(t_gt)]
            boneloss = boneloss + loss_function(torch.Tensor(bones_out), torch.Tensor(bones_gt))
            #             assert np.allclose(bones_out,[np.linalg.norm(i) for i in offsets]) #for testing purposes
            #             assert np.allclose(bones_gt,[np.linalg.norm(i) for i in offsets])  #for testing purposes
            batchloss_bl += boneloss.item()

        batchloss += posloss.item()

    loss = batchloss / batch_sz
    loss_bl = batchloss_bl / batch_sz
    return loss, loss_bl

def qconj(q):
    """
    Returns conjugate of quaternion
    inputs
    -------
    q: torch.tensor, shape: (*,4)
    
    outputs
    -------
    torch.tensor (*,4), conjugate quaternion """
    return torch.cat((q[:,0][:,None],-q[:,1][:,None],-q[:,2][:,None],-q[:,3][:,None]),dim=1)

def qconj_np(q):
    """
    Returns conjugate of quaternion
    inputs
    -------
    q: np.array, shape: (*,4)
    
    outputs
    -------
    np.array (*,4), conjugate quaternion """
    return np.concatenate((q[:,0][:,None],-q[:,1][:,None],-q[:,2][:,None],-q[:,3][:,None]),1)

def dqconj(dq):
    """
    Returns conjugate of dual quaternion
    inputs
    -------
    q: torch.tensor, shape: (*,8)
    
    outputs
    -------
    torch.tensor (*,8), conjugate dual quaternion """
    return torch.cat((dq[:,0][:,None],-dq[:,1][:,None],-dq[:,2][:,None],-dq[:,3][:,None],-dq[:,4][:,None],dq[:,5][:,None],dq[:,6][:,None],dq[:,7][:,None]),dim=1)

def dqconj_np(dq):
    """
    Returns conjugate of dual quaternion
    inputs
    -------
    q: np.array, shape: (*,8)
    
    outputs
    -------
    np.array (*,8), conjugate dual quaternion """
    dq = torch.from_numpy(dq).contiguous()
    return dqconj(dq).numpy()
    
def dqinv(dq):
    """
    Returns inverse of dual quaternion
    inputs
    -------
    q: torch.tensor, shape: (*,8)
    
    outputs
    -------
    torch.tensor (*,8), inverse dual quaternion """
    return torch.cat((dq[:,0][:,None],-dq[:,1][:,None],-dq[:,2][:,None],-dq[:,3][:,None],dq[:,4][:,None],-dq[:,5][:,None],-dq[:,6][:,None],-dq[:,7][:,None]),dim=1)

def dqinv_np(dq):
    """
    Returns inverse of dual quaternion
    inputs
    -------
    q: np.array, shape: (*,8)
    
    outputs
    -------
    np.array (*,8), inverse dual quaternion """
    dq = torch.from_numpy(dq).contiguous()
    return dqinv(dq).numpy()

def dqrot(dq_trans,dq_point):
    """
    Rotates a point by a dual quaternion
    inputs
    -------
    dq_trans: transformation, torch.tensor, shape: (*,8)
    dq_point: point, torch.tensor, shape: (*,8)
    
    outputs
    -------
    torch.tensor (*,3), rotated point """
    return dqmul(dqmul(dq_trans,dq_point),dqconj(dq_trans))[:,5:]
    
def dqrot_np(dq_trans,dq_point):
    """
    Rotates a point by a dual quaternion
    inputs
    -------
    dq_trans: transformation, np.array.tensor, shape: (*,8)
    dq_point: point, np.array, shape: (*,8)
    
    outputs
    -------
    np.array (*,3), rotated point """
    dq_trans = torch.from_numpy(dq_trans).contiguous()
    dq_point= torch.from_numpy(dq_point).contiguous()
    return dqrot(dq_trans,dq_point).numpy()

def is_unit(dq,stop=True):
    """
    Checks if dual quaternion is unit.
    inputs
    -------
    dq: dual quaternions, torch.tensor, shape: (*,8)
    stop: bool, stop if not unit
    
    outputs
    -------
    """
    if stop==True:
        assert (torch.norm(dq[:,:4], dim=1)-1<1e-4).all() and (torch.sum(dq[:,:4]*dq[:,4:],dim=1)<1e-3).all()
    return (torch.norm(dq[:,:4], dim=1)-1<1e-4).all() and (torch.sum(dq[:,:4]*dq[:,4:],dim=1)<1e-3).all()

def is_unit_np(dq,stop=True):
   """
    Checks if dual quaternion is unit.
    inputs
    -------
    dq: dual quaternions, np.array, shape: (*,8)
    stop: bool, stop if not unit
    
    outputs
    -------
    """
    dq = torch.from_numpy(dq).contiguous()
    return is_unit(dq,stop)
    
def dqnorm(dq,force=False):
    """
    Normalizes dual quaternion.
    inputs
    -------
    dq: dual quaternions, torch.tensor, shape: (*,8)
    force: bool, force normalize (used for network outputs)
    
    outputs
    -------
    torch.tensor, shape: (*,8), normalized dual quaternions
    """
    quats = dq[:,:4]
    dualquats = dq[:,4:]
    quats_normalized = F.normalize(quats,dim=1)
    norm = torch.norm(quats,dim=1)
    norm =torch.stack((norm,norm,norm,norm),dim=1)
    dualquats_normalized = torch.div(dualquats,norm)
    if force:
        if is_unit(dq,stop=False)==False:

            q = dq[:, :4]
            d = dq[:, 4:]

            qnorm = torch.norm(q, dim=1)
            quats_normalized = F.normalize(q,dim=1) #torch.div(q, qnorm[:, None])

            qd = torch.sum(q * d, dim=1)
            dualquats_normalized = torch.div(d, qnorm[:, None]) - quats_normalized * torch.div(qd, qnorm ** 2)[:, None]
            if is_unit(torch.cat((quats_normalized,dualquats_normalized),1))==False:
                print(q,d)
                print(dq)
    return torch.cat((quats_normalized,dualquats_normalized),1)

def dqnorm_np(dq):
    """
    Normalizes dual quaternion.
    inputs
    -------
    dq: dual quaternions, np.array, shape: (*,8)
    force: bool, force normalize (used for network outputs)
    
    outputs
    -------
    np.array, shape: (*,8), normalized dual quaternions
    """
    dq = torch.from_numpy(dq).contiguous()
    return dqnorm(dq).numpy()
    

def translation(dq):
    """
    Extract translation component.
    inputs
    -------
    dq: dual quaternions, torch.tensor, shape: (*,8)

    outputs
    -------
    torch.tensor, shape: (*,3), translation
    """
    dq = dqnorm(dq)
    dualquats_normalized = dq[:,4:]
    qt = qconj(dq[:,:4])
    return qmul(torch.mul(2,dualquats_normalized),qt)[:,1:]

def translation_np(dq):
   """
    Extract translation component.
    inputs
    -------
    dq: dual quaternions, np.array, shape: (*,8)
    
    outputs
    -------
    np.array, shape: (*,3), translation
    """
    dq = torch.from_numpy(dq).contiguous()
    return translation(dq).numpy()
    
def dqmul(dq, dq1):  # accepts normalized quaternions (*,8)
    """
    Multiplies two normalized dual quaternions. 
    inputs
    -------
    dq1: dual quaternions, torch.tensor, shape: (*,8)
    dq2: dual quaternions, torch.tensor, shape: (*,8)
    
    outputs
    -------
    torch.tensor, shape: (*,8), multiplication result
    """
    q = dq[:, :4]
    r = dq1[:, :4]

    d_q = dq[:, 4:]
    d_r = dq1[:, 4:]

    q_ = qmul(q, r)
    d_ = qmul(q, d_r) + qmul(d_q, r)

    assert (q_ == qmul(q, r)).all()
    return torch.cat((q_, d_), 1)
    
def dqmul_np(dq,dq1): #accepts normalized quaternions (*,8)
    """
    Multiplies two normalized dual quaternions. 
    inputs
    -------
    dq1: dual quaternions, np.array, shape: (*,8)
    dq2: dual quaternions, np.array, shape: (*,8)
    
    outputs
    -------
    np.array, shape: (*,8), multiplication result
    """
    dq = torch.from_numpy(dq).contiguous()
    dq1 = torch.from_numpy(dq1).contiguous()
    return dqmul(dq, dq1).numpy()

def qfix_dq(dq):
    """
    Enforce dual quaternion continuity across the time dimension by selecting
    the representation (dq or -dq) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    Expects a tensor of shape (T, J, 8), where T is the sequence length and J is the number of joints.
    Returns a tensor of the same shape.
    
    inputs
    ------------
    dq: dual quaternions, shape: (T,J,8)
    outputs
    ------------
    torch.tensor, shape: (T,J,8) fixed dual quaternions
    """
    q = dq[:, :, :4]
    d = dq[:, :, 4:]
    assert len(q.shape) == 3
    assert q.shape[-1] == 4
    result = q.copy()
    dot_products = np.sum(q[1:] * q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    resultd = d.copy()
    resultd[1:][mask] *= -1
    return np.concatenate((result, resultd), 2)

# def get_offsets(dq,parh,nj=26):
#     parh[0] = 0
#     dq_par = dq[:,:,parh,:]
#     dq_par[:,:,0,:] = torch.Tensor([1,0,0,0,0,0,0,0])
#
#     dq = dq.reshape(-1,8)
#
#     dq_par = dq_par.reshape(-1,8)
#     return translation(dqmul(dqinv(dqnorm(dq_par)) , dq)).reshape(-1,nj,3)
#
# def get_offsets_np(dq,par):
#     par[0] = 0
#     joints_num = 31
#     batch = dq.shape[0]
#     seq_len = dq.shape[1]
#     dq_par = dq[:,:,par,:]
#     dq_par[:,:,0,:] = [1,0,0,0,0,0,0,0]
#
#     dq = dq.reshape(-1,8)
#
#     dq_par = dq_par.reshape(-1,8)
#     return translation_np(dqmul_np(dqinv_np(dqnorm_np(dq_par)) , dq)).reshape(batch,seq_len,joints_num,3)

def get_offsets(dq,parh,nj=26):
   """
    Extract offsets.
    inputs
    -------
    dq: dual quaternions, torch.tensor, shape: (*,8)
    parh: list, hierarchy of joints
    nj: int, number of joints
    outputs
    -------
    torch.tensor, shape: (*,3), offsets
    """
    parh[0] = 0
    dq_par = dq[:,:,parh,:]
    dq_par[:,:,0,:] = torch.Tensor([1,0,0,0,0,0,0,0])

    dq = dq.reshape(-1,8)

    dq_par = dq_par.reshape(-1,8)
    return translation(dqmul(dqinv(dqnorm(dq_par)) , dq)).reshape(-1,nj,3)

def get_offsets_np(dq,parh,nj=26):
   """
    Extract offsets.
    inputs
    -------
    dq: dual quaternions, np.array, shape: (*,8)
    parh: list, hierarchy of joints
    nj: int, number of joints
    outputs
    -------
    np.array, shape: (*,3), offsets
    """
    parh[0] = 0
    dq_par = dq[:,:,parh,:]
    dq_par[:,:,0,:] = [1,0,0,0,0,0,0,0]

    dq = dq.reshape(-1,8)

    dq_par = dq_par.reshape(-1,8)
    return translation_np(dqmul_np(dqinv_np(dqnorm_np(dq_par)) , dq)).reshape(-1,nj,3)
    

def currentdq2localdq(currentq,parh):
   """Current dual quaternions to local dual quaternions

    inputs
    ------
    currentq: torch.tensor, current dual quaternions for each joint, size: (TxJ,8) #frames x (#joints used *8)
    parh: parents list (same size as h)


    outputs
    -------
    local dual quaternions, size (T,J,8)
    """
    assert currentq.shape[-1] == 8
    assert len(currentq.shape)==3
    qparents  = currentq[ :, parh, :]
    qparents[ :, 0] = torch.tensor([1, 0, 0, 0,0, 0, 0, 0])

    localq = dqnorm(dqmul(dqinv(
    qparents.reshape(-1,8)), currentq.reshape(-1,8)))
    return localq.reshape(currentq.shape)

def currentq2localq(currentq,parh):
   """Current dual quaternions to local quaternions 

    inputs
    ------
    currentq: torch.tensor, current dual quaternions for each joint, size: (TxJ,8) #frames x (#joints used *8)
    parh: parents list (same size as h)


    outputs
    -------
    local quaternions, size (T,J,4)
    """
    assert currentq.shape[-1] == 4
    assert len(currentq.shape)==3
    qparents  = currentq[ :, parh, :]
    qparents[ :, 0] = torch.tensor([1, 0, 0, 0])
    localq = qmul(qconj(
    qparents.reshape(-1,4)), currentq.reshape(-1,4))
    return localq.reshape(currentq.shape)