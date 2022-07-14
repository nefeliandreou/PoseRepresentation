import numpy as np
import torch
from DualQuaternion2 import DualQuaternion
from loadbvh2 import loadbvh2
from loadbvh import loadbvh
import quaternion
import torch.nn.functional as F
from common.quaternion import qeuler, qeuler_np, qmul
from code1.dualquats import dqnorm, get_offsets, qconj, translation
import time
from torch import nn as nn

Quaternion = quaternion.quaternion


def qrl_local(out_seq, groundtruth_seq, batch=32, seq_len=100, joints_num=31, order='zxy', translation_dims=3,
              quats=False, fc=True, parh=None):
    predicted = out_seq
    expected = groundtruth_seq



    if quats:
       if fc:
           batch_gt = expected.reshape(batch, seq_len, joints_num * 4 + 5)[:, :, 3:-2].reshape(batch, seq_len,
                                                                                               joints_num,
                                                                                               4)[
                      :, :, :, :4]
           batch_out = predicted.reshape(batch, seq_len, joints_num * 4 + 5)[:, :, 3:-2].reshape(batch, seq_len,
                                                                                                 joints_num,
                                                                                                 4)[:, :, :, :4]
       else:
           batch_gt = expected.reshape(batch, seq_len, joints_num * 4 + 3)[:, :, 3:3+4*joints_num].reshape(batch, seq_len, joints_num,
                                                                                             4)[
                      :, :, :, :4]
           batch_out = predicted.reshape(batch, seq_len, joints_num * 4 + 3)[:, :, 3:3+4*joints_num].reshape(batch, seq_len,
                                                                                               joints_num,
                                                                                               4)[:, :, :, :4]

       return torch.mean(torch.abs(1 - torch.sum(batch_gt.contiguous() * batch_out.contiguous(), dim=3)))

    else:

        if fc:
            batch_gt = expected.reshape(batch, seq_len, joints_num * 8 + 5)[:, :, 3:-2].reshape(batch, seq_len, joints_num,
                                                                                                8)[
                       :, :, :, :4]
            batch_out = predicted.reshape(batch, seq_len, joints_num * 8 + 5)[:, :, 3:-2].reshape(batch, seq_len,
                                                                                                  joints_num,
                                                                                                  8)[:, :, :, :4]
        else:
            batch_gt = expected.reshape(batch, seq_len, joints_num * 8 + 3)[:, :, 3:].reshape(batch, seq_len, joints_num,
                                                                                              8)[
                       :, :, :, :4]
            batch_out = predicted.reshape(batch, seq_len, joints_num * 8 + 3)[:, :, 3:].reshape(batch, seq_len, joints_num,
                                                                                                8)[:, :, :, :4]


        expected_quats = batch_gt
        predicted_quats = batch_out
        parh[0] = 0
        predicted_quats = predicted_quats.view(batch, seq_len, joints_num, 4)
        expected_quats = expected_quats.view(batch, seq_len, joints_num, 4)

        predicted_parents = predicted_quats[:, :, parh]
        predicted_parents[:, :, 0] = torch.Tensor([1, 0, 0, 0])

        expected_parents = expected_quats[:, :, parh]
        expected_parents[:, :, 0] = torch.Tensor([1, 0, 0, 0])

        expected_quats=expected_quats.contiguous()
        predicted_quats = predicted_quats.contiguous()
        local_expected = F.normalize((qmul(qconj(expected_parents.view(-1, 4)).reshape(batch, seq_len, joints_num, 4),
                          expected_quats.reshape(batch, seq_len, joints_num, 4))).reshape(-1, 4)).reshape(
            expected_parents.shape)
        local_predicted = F.normalize((qmul(qconj(predicted_parents.reshape(-1, 4)).reshape(batch, seq_len, joints_num, 4),
                              predicted_quats.reshape(batch, seq_len, joints_num, 4))).reshape(-1, 4)).reshape(batch,
                                                                                                               seq_len,-1,4)

        assert torch.allclose(torch.sum(local_predicted**2,dim=-1),torch.ones((32,100,joints_num)).cuda())
        assert torch.allclose(torch.sum(local_expected ** 2, dim=-1), torch.ones((32,100,joints_num)).cuda())
        return torch.mean(torch.abs(1 - torch.sum(local_predicted * local_expected, dim=3)))

def qrl_curr(out_seq, groundtruth_seq, batch=32, seq_len=100, joints_num=31, order='zxy', translation_dims=3,
             fc=True):  # groundtruth_seq 32,25100(251*100)
    start = time.time()
    predicted_quats = out_seq
    expected_quats = groundtruth_seq

    if fc:
        expected_quats = expected_quats.view(batch, seq_len, joints_num * 8 + translation_dims + 2)[:, :, 3:-2].view(
            batch, seq_len, joints_num, 8)[:, :, :, :4]
        predicted_quats = predicted_quats.view(batch, seq_len, joints_num * 8 + translation_dims + 2)[:, :, 3:-2].view(
            batch, seq_len, joints_num, 8)[:, :, :, :4]
    else:
        expected_quats = expected_quats.view(batch, seq_len, joints_num * 8 + translation_dims)[:, :, 3:].view(batch,
                                                                                                               seq_len,
                                                                                                               joints_num,
                                                                                                               8)[:, :,
                         :, :4]
        predicted_quats = predicted_quats.view(batch, seq_len, joints_num * 8 + translation_dims)[:, :, 3:].view(batch,
                                                                                                                 seq_len,
                                                                                                                 joints_num,
                                                                                                                 8)[:,
                          :, :, :4]


    return torch.mean(torch.abs(1- torch.sum(predicted_quats * expected_quats,dim=3)))


def dl(out_seq, groundtruth_seq, fc=False, bl=True, parh=None,batch = 32, seq_len = 100,joints_num = 31):
    boneloss=0
    parh[0] = 0
    predicted_dual = out_seq.view(batch, seq_len, -1)
    expected_dual = groundtruth_seq.view(batch, seq_len, -1)

    if fc:
        predicted_dual = predicted_dual[:, :, 3:-2]
        expected_dual = expected_dual[:, :, 3:-2]
    else:
        predicted_dual = predicted_dual[:, :, 3:]
        expected_dual = expected_dual[:, :, 3:]
    predicted_dual = predicted_dual.reshape(batch * seq_len * joints_num, 8)
    expected_dual = expected_dual.reshape(batch * seq_len * joints_num, 8)

    translation_predicted = translation(predicted_dual).view(batch, seq_len, joints_num, 3)
    translation_expected = translation(expected_dual).view(batch, seq_len, joints_num, 3)
    differences = (translation_expected - translation_predicted).norm(dim=3)

    if bl:
        offset_loss = (get_offsets(predicted_dual.reshape(batch, seq_len, -1, 8), parh, joints_num).reshape(batch, seq_len, joints_num,
                                                                                                3) - get_offsets(
        expected_dual.reshape(batch, seq_len, -1, 8), parh, joints_num).reshape(batch, seq_len, joints_num, 3)).norm(dim=3)
        return torch.mean(differences + offset_loss.expand_as(differences))/3, 0
    else:
        return torch.mean(differences) / 3, 0



def out_dq_norm(predicted, batch_size, seq_len,fc,names=None,quats=False,nj=31):  # out (32,253000)
    if fc:
        if quats:
            dq = F.normalize(predicted.view(batch_size, seq_len, -1)[:, :, 3:3+4*nj].reshape(-1, 4)).view(batch_size,
                                                                                                             seq_len,
                                                                                                             -1)
        else:
            dq = dqnorm(predicted.view(batch_size, seq_len, -1)[:, :, 3:-2].reshape(-1, 8),force=True).view(batch_size, seq_len, -1)
            from code1.dualquats import is_unit
            is_unit(dq.reshape(-1,8))
        hip = predicted.view(batch_size, seq_len, -1)[:, :, :3]
        foot = predicted.view(batch_size, seq_len, -1)[:, :, -2:]
        norm  = torch.cat((torch.cat((hip, dq), dim=2), foot), dim=2)
    else:
        if quats:
            dq = F.normalize(predicted.view(batch_size, seq_len, -1)[:, :, 3:3+4*nj].reshape(-1, 4)).view(batch_size,
                                                                                                           seq_len, -1)
            assert torch.allclose(torch.sum(dq.reshape(-1, 100, nj, 4) ** 2, axis=-1), torch.ones((32, 100, nj)).cuda())
        else:
            dq = dqnorm(predicted.view(batch_size, seq_len, -1)[:, :, 3:].reshape(-1, 8),force=True).view(batch_size, seq_len, -1)
        hip = predicted.view(batch_size, seq_len, -1)[:, :, :3]
        norm = torch.cat((hip, dq), dim=2)
    assert torch.allclose(predicted.view(batch_size, seq_len, -1)[:, :, :3], norm[:, :, :3])
    if fc:
        assert torch.allclose(predicted.view(batch_size, seq_len, -1)[:, :, -2:], norm[:, :, -2:])

    return norm.view(batch_size,-1)
    
