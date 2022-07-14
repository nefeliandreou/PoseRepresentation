import os
import torch
from utils.bvh.rotation2xyz import *
import loadbvh
from utils.bvh import read_bvh
import numpy as np
from tqdm import tqdm
import os
from common.quaternion import qfix
from dualquats import qfix_dq
import quaternion
Quaternion = quaternion.quaternion

class allbvh2other(loadbvh):
    def __init__(self, in_folder_path, out_folder_path, order='zxy'):
        self.fp_in = in_folder_path
        self.dq_out = out_folder_path
        self.order = order

        if out_folder_path != "":
            os.makedirs(out_folder_path, exist_ok=True)

    def write_dq_to_bvh(self, bvh_filename, train_data, config_bvh):
        out_seq = k.skeleton2bvh(k.dqframes2skeleton(train_data))
        self.write_frames(self.fp_in + '/' + self.standard_bvh_file, bvh_filename, out_seq)

    def get_frame_format_string(self, bvh_filename):
        bvh_file = open(bvh_filename, "r")
        lines = bvh_file.readlines()
        bvh_file.close()
        l = [lines.index(i) for i in lines if 'MOTION' in i]
        data_end = l[0]
        # data_end = lines.index('MOTION\n')
        data_end = data_end + 2
        return lines[0:data_end + 1]

    def vector2string(self, data):
        s = ' '.join(map(str, data))

        return s

    def vectors2string(self, data):
        s = '\n'.join(map(self.vector2string, data))

        return s

    def write_frames(self, format_filename, out_filename, data):
        data = np.round(data, 6)
        format_lines = self.get_frame_format_string(format_filename)
        num_frames = data.shape[0]
        format_lines[len(format_lines) - 2] = "Frames:\t" + str(num_frames) + "\n"

        bvh_file = open(out_filename, "w")
        bvh_file.writelines(format_lines)
        bvh_data_str = self.vectors2string(data)
        bvh_file.write(bvh_data_str)
        bvh_file.close()

    def write_framesdq(self, out_filename, data):
        np.save(out_filename, data)

    def writeall(self):
        allfiles = [i for i in os.listdir(self.fp_in)]
        for file in tqdm(allfiles):
            self.standard_bvh_file = file
            self.bvh_file_in = self.fp_in + '/' + file
            self.bvh_file_out = 'bvh_reconstruct/' + str(file).split('.bvh')[0] + '2.bvh'
#             print(self.fp_in + '/' + str(file))
            p = loadbvh(bvh_file=self.fp_in + '/' + str(file), order=self.order)
            p()
            self.write_dq_to_bvh(self.bvh_file_out, p.framesDQC, p)

    def writealldq(self, q_fix=False):
        allfiles = [i for i in os.listdir(self.fp_in) if '.bvh' in i]
        for file in tqdm(allfiles):
            self.bvh_file_out = self.dq_out + '/' + str(file).split('.bvh')[0]
            p = loadbvh(bvh_file=self.fp_in + '/' + str(file), order=self.order)
            p()
            root = p.framesDQC[:, :3]
            dat = p.framesDQC[:, 3:].reshape(p.Nframes, 31, 8)
            if q_fix:
                self.write_framesdq(self.bvh_file_out,
                                    np.concatenate((root, qfix_dq(dat).reshape(p.Nframes, 31 * 8)), 1))
            else:
                self.write_framesdq(self.bvh_file_out, p.framesDQC)

    def writeallq(self, q_fix=False):
        allfiles = [i for i in os.listdir(self.fp_in) if i.endswith('.bvh')]
        for file in tqdm(allfiles):
            self.bvh_file_out = self.dq_out + '/' + str(file).split('.bvh')[0]
            p = loadbvh(bvh_file=self.fp_in + '/' + str(file), order=self.order)
            data = p.skeleton2q()
            root = data[:, :3]
            dat = data[:, 3:].reshape(p.Nframes, 31, 4)

            if q_fix:
                self.write_framesdq(self.bvh_file_out, np.concatenate((root, qfix(dat).reshape(p.Nframes, 31 * 4)), 1))
            else:
                self.write_framesdq(self.bvh_file_out, data)

    def writeallortho6D(self):
        allfiles = [i for i in os.listdir(self.fp_in) if 'bvh' in i]
        for file in tqdm(allfiles):
            self.bvh_file_out = self.dq_out + '/' + str(file).split('.bvh')[0]
            p = loadbvh(bvh_file=self.fp_in + '/' + str(file), order=self.order)
            data = p.skeleton2ortho6D(self.order)
            self.write_framesdq(self.bvh_file_out, data)

    def writeallpos(self):
        allfiles = [i for i in os.listdir(self.fp_in) if '.bvh' in i]

        for file in tqdm(allfiles):
            p = loadbvh(bvh_file=self.fp_in + '/' + str(file), order=self.order)
            get_pos('LeftFoot',p.data[100],p.non_end_bones,p.skeleton)

            for ff in range(4500):
                for nn in p.non_end_bones:
                    assert (np.max(np.abs(get_pos(nn,p.data[ff],p.non_end_bones[1:],p.skeleton)[:3].reshape(-1,3)-p.skeleton[nn]['transG'][:3,3,ff]))<1e-5)
    def writeallqpos(self,q_fix=True):
        allfiles = [i for i in os.listdir(self.fp_in) if i.endswith('.bvh')]
        for file in tqdm(allfiles):
            self.bvh_file_out = self.dq_out + '/' + str(file).split('.bvh')[0]
            p = loadbvh(bvh_file=self.fp_in + '/' + str(file), order=self.order)
            data = p.skeleton2q()
            root = data[:, :3]
            dat = data[:, 3:].reshape(p.Nframes, 31, 4)
            pos = p.exportpositions(outfolder=self.dq_out,save=False)[:,3:]
            if q_fix:
                data_f = np.concatenate((root, qfix(dat).reshape(p.Nframes, 31 * 4),pos), 1)
            )
            else:
                data_f = p.concatenate((data,pos), 1)
            self.write_framesdq(self.bvh_file_out, data_f)
         
    def quat2expmap(self,q):
        """
        Converts a quaternion to an exponential map
        Matlab port to python for evaluation purposes
        https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1
        Args
          q: 1x4 quaternion
        Returns
          r: 1x3 exponential map
        Raises
          ValueError if the l2 norm of the quaternion is not close to 1
        """
        if (np.abs(np.linalg.norm(q) - 1) > 1e-3):
            print(np.linalg.norm(q))
            raise (ValueError, "quat2expmap: input quaternion is not norm 1")

        sinhalftheta = np.linalg.norm(q[1:])
        coshalftheta = q[0]

        r0 = np.divide(q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps));
        theta = 2 * np.arctan2(sinhalftheta, coshalftheta)
        theta = np.mod(theta + 2 * np.pi, 2 * np.pi)

        if theta > np.pi:
            theta = 2 * np.pi - theta
            r0 = -r0

        r = r0 * theta
        return r
  
    def writeallaxanglepos(self, q_fix=True):
        
        self.aaLoss(torch.zeros(0))
        allfiles = [i for i in os.listdir(self.fp_in) if i.endswith('.bvh')]
        for file in tqdm(allfiles):
            self.bvh_file_out = self.dq_out + '/' + str(file).split('.bvh')[0]
            p = loadbvh(bvh_file=self.fp_in + '/' + str(file), order=self.order)
            data = p.skeleton2q()
            datafix =qfix(data[:,3:].reshape(p.Nframes,31,4))
            root = data[:, :3]
            ax = np.zeros((datafix.shape[0],datafix.shape[1],3))
            from scipy.spatial.transform import Rotation as R
            for idi,i in enumerate(datafix):
                for idj,j in enumerate(i):
                    r = R.from_quat(datafix[idi,idj])
                    axang = r.as_rotvec()
                    ax[idi,idj] = axang
            pos = p.exportpositions(outfolder=self.dq_out,save=False)[:, 3:]
            data_f = np.concatenate((root, ax.reshape(root.shape[0],-1), pos), 1)
            self.write_framesdq(self.bvh_file_out, data_f)


