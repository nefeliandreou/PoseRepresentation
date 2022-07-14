# from pylab import *
from pyquaternion import Quaternion

from utils.bvh import rotation2xyz as helper, read_bvh_hierarchy
from utils.bvh.rotation2xyz import *

"""
Code from: https://github.com/papagina/Auto_Conditioned_RNN_motion

Important: Always use the initialization method: initialize_bvh_definitions()
           before using the methods!
"""

standard_bvh_file = "/DATA/Projects/2020/lstm_pro/src/rhythmdata/skeleton_conf.bvh"
skeleton, non_end_bones = read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_file)
joint_names = list(skeleton.keys())
bvh_rotations_order = 'zxy'


def initialize_bvh_definitions(skeleton_file_name, rotations_order_of_bvh):
    global standard_bvh_file, skeleton, non_end_bones, joint_names, joint_index, bvh_rotations_order
    standard_bvh_file = skeleton_file_name
    skeleton, non_end_bones = read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_file)
    joint_names = list(skeleton.keys())
    sample_data = parse_frames(standard_bvh_file)  # Bvh rotations + root's translation
    joint_index = get_pos_joints_index(sample_data[0], non_end_bones, skeleton)
    bvh_rotations_order = rotations_order_of_bvh
    return np.array(joint_names)


def get_pos_joints_index(raw_frame_data, non_end_bones, skeleton):
    pos_dic = helper.get_skeleton_position(raw_frame_data, non_end_bones, skeleton)
    keys = OrderedDict()
    i = 0
    for joint in pos_dic.keys():
        keys[joint] = i
        i = i + 1
    return keys


def parse_frames(bvh_filename):
    """
    Parse frames from bvh file.
    Used in generate training data.
    Returns the data in shape of [#frames, #attributes]
    Attributes are translation of root & rotations of joints.
    """
    bvh_file = open(bvh_filename, "r")  # Load file
    lines = bvh_file.read().splitlines()  # Read lines without new line.
    bvh_file.close()  # Bye Bye file.
    l = [lines.index(i) for i in lines if 'MOTION' in i]  # Get the indices for the MOTION part.
    data_start = l[0]  # Skip MOTION, Frames, and Frame Rate
    first_frame = data_start + 3

    num_params = len(list(filter(None, lines[first_frame].split(' '))))
    num_frames = len(lines) - first_frame

    data = np.zeros((num_frames, num_params))

    for i in range(num_frames):
        line = lines[first_frame + i].split(' ')  # Split into number strings.
        line = line[0:len(line)]  # Validate.
        line = list(filter(None, line))  # Remove any empty strings.
        line_f = [float(e) for e in line]  # Convert to float.
        data[i, :] = line_f  # Save.

    return data


sample_data = parse_frames(standard_bvh_file)
joint_index = get_pos_joints_index(sample_data[0], non_end_bones, skeleton)


def get_frame_format_string(bvh_filename):
    bvh_file = open(bvh_filename, "r")
    lines = bvh_file.readlines()
    bvh_file.close()
    l = [lines.index(i) for i in lines if 'MOTION' in i]
    data_end = l[0]
    # data_end = lines.index('MOTION\n')
    data_end = data_end + 2
    return lines[0:data_end + 1]


def get_motion_center(bvh_data):
    """Calculated the center of the motion using the positions of each frame."""
    center = np.zeros(3)
    for frame in bvh_data:
        center = center + frame[0:3]
    center = center / bvh_data.shape[0]
    return center


def augment_train_frame_data(train_frame_data, T, axisR):
    """
    This function augments the training data, using the Translation T
    an angle axisR[3] to rotate around axisR[0:3]. Usually, we rotate around y axis, axisR[3] angle.

    Parameters:
        train_frame_data:
            The data in shape of [#frames, #attributes]
        T:
            The Translation of the dance.
            It is used also to normalize the position of each frame,
            by subtracting the average center of the dance.

        axisR:
            axisR[0] = x
            axisR[1] = y
            axisR[2] = z
            axisR[3] = degrees

    """
    hip_index = joint_index[joint_names[0]]
    hip_pos = train_frame_data[hip_index * 3: hip_index * 3 + 3]

    for i in range(int(len(train_frame_data) / 3)):
        if i != hip_index:
            train_frame_data[i * 3: i * 3 + 3] = train_frame_data[i * 3: i * 3 + 3] + hip_pos

    mat_r_augment = euler.axangle2mat(axisR[0:3], axisR[3])
    n = int(len(train_frame_data) / 3)
    for i in range(n):
        raw_data = train_frame_data[i * 3:i * 3 + 3]
        new_data = np.dot(mat_r_augment, raw_data) + T
        train_frame_data[i * 3:i * 3 + 3] = new_data

    hip_pos = train_frame_data[hip_index * 3: hip_index * 3 + 3]

    for i in range(int(len(train_frame_data) / 3)):
        if i != hip_index:
            train_frame_data[i * 3: i * 3 + 3] = train_frame_data[i * 3: i * 3 + 3] - hip_pos
    return train_frame_data


def augment_train_data(train_data, T, axisR):
    """Augments the data by Translating (T) and Rotating by axisR[3] around axis[0:3]"""
    result = list(map(lambda frame: augment_train_frame_data(frame, T, axisR), train_data))
    return np.array(result)


def get_one_frame_training_format_data(raw_frame_data, non_end_bones, skeleton):
    """
    Input a vector of data, with the first three data as translation and the rest the euler rotation
    output a vector of data, with the first three data as translation not changed and the rest to quaternions.
    note: the input data are in z, x, y sequence
    """
    pos_dic = helper.get_skeleton_position(raw_frame_data, non_end_bones, skeleton)
    new_data = np.zeros(len(pos_dic.keys()) * 3)
    i = 0
    hip_pos = pos_dic[joint_names[0]]
    # print hip_pos

    for joint in pos_dic.keys():
        if joint == joint_names[0]:
            new_data[i * 3:i * 3 + 3] = pos_dic[joint].reshape(3)
        else:
            new_data[i * 3:i * 3 + 3] = pos_dic[joint].reshape(3) - hip_pos.reshape(3)
        i = i + 1
    # print(new_data)
    new_data = new_data * 0.01
    return new_data


def get_training_format_data(raw_data, non_end_bones, skeleton):
    new_data = []
    for frame in raw_data:
        new_frame = get_one_frame_training_format_data(frame, non_end_bones, skeleton)
        new_data = new_data + [new_frame]
    return np.array(new_data)


def get_train_data(bvh_filename):
    """Returns the training data from a BVH.
    We calculate the center (avg position) of the bvh data, in order to augment the data."""
    data = parse_frames(bvh_filename)
    train_data = get_training_format_data(data, non_end_bones, skeleton)
    center = get_motion_center(train_data)  # get the avg position of the hip
    center[1] = 0.0  # don't center the height
    new_train_data = augment_train_data(train_data, -center, [0, 1, 0, 0.0])
    return new_train_data


def get_train_data_from_positions(bvh_filename):
    """Returns the training data from a BVH.
    We calculate the center (avg position) of the bvh data, in order to augment the data."""
    data = parse_frames(bvh_filename)
    train_data = get_training_format_data(data, non_end_bones, skeleton)
    center = get_motion_center(train_data)  # get the avg position of the hip
    center[1] = 0.0  # don't center the height
    new_train_data = augment_train_data(train_data, -center, [0, 1, 0, 0.0])
    return new_train_data


def change_hips_position_to_relative(dance):
    """
    We update the hips_x and hips_z, in a way that they show the relative translation and not the global position.
    :param dance:
    :return: The dance with 1 frame less.
    """
    result_dance = dance[:dance.shape[0] - 1].copy()
    dance_prev = dance[:dance.shape[0] - 1]
    dance_curr = dance[1:]
    diff = dance_curr - dance_prev
    result_dance[:, 0] = diff[:, 0]  # Hips_x
    result_dance[:, 2] = diff[:, 2]  # Hips_z

    # Rel root rot also:
    # if rotation:
    #     for i, (f_p, f_n) in enumerate(zip(dance[:-1], dance[1:])):
    #         q_prev = Quaternion(f_p[3], f_p[4], f_p[5], f_p[6])
    #         q_next = Quaternion(f_n[3], f_n[4], f_n[5], f_n[6])
    #         q_diff = q_next * q_prev.inverse  # todo ??? << check here, also in the save_bvh :)
    #         result_dance[i, 3:7] = [q_diff.w, q_diff.x, q_diff.y, q_diff.z]

    return result_dance


# # todo: new addition
# def change_hips_position_to_relative_rotation(frames):
#     updated_frames = frames[:-1].copy()
#     for i, (f_p, f_n) in enumerate(zip(frames, frames[1:-1])):
#         q_prev = Quaternion(f_p[3], f_p[4], f_p[5], f_p[6])
#         q_next = Quaternion(f_n[3], f_n[4], f_n[5], f_n[6])
#         q_diff = q_next * q_prev.inverse # todo ??? << check here, also in the save_bvh :)
#         updated_frames[i,3:7] = [q_diff.w, q_diff.x, q_diff.y, q_diff.z]
#     return updated_frames


def _write_frames(format_filename, out_filename, data):
    """
    Writes the frames in the bvh file.
    """
    format_lines = get_frame_format_string(format_filename)

    num_frames = data.shape[0]
    format_lines[len(format_lines) - 2] = "Frames:\t" + str(num_frames) + "\n"

    bvh_file = open(out_filename, "w")
    bvh_file.writelines(format_lines)
    bvh_data_str = vectors2string(data)
    bvh_file.write(bvh_data_str)
    bvh_file.close()


def _write_xyz_to_bvhn(xyz_motion, skeleton, non_end_bones, format_filename, output_filename="",
                       bvh_rotations_order='zxy'):
    """Prepares the motion to be written in the bvh file.
    Calls write_frames, to write the motion."""
    bvh_vec_length = len(non_end_bones) * 3 + 6
    if bvh_rotations_order == 'zxy':
        o = [2, 0, 1]
    elif bvh_rotations_order == 'zyx':
        o = [2, 1, 0]
    out_data = np.zeros([len(xyz_motion), bvh_vec_length])
    for i in range(1, len(xyz_motion)):
        positions = xyz_motion[i]
        rotation_matrices, rotation_angles = helper.xyz_to_rotations_debugn(skeleton, positions, joint_names[0],
                                                                            rotations_order=bvh_rotations_order)
        new_motion1 = helper.rotation_dic_to_vec(rotation_angles, non_end_bones, positions, joint_names[0], o)

        new_motion = np.array([round(a, 6) for a in new_motion1])
        new_motion[0:3] = new_motion1[0:3]

        out_data[i, :] = np.transpose(new_motion[:, np.newaxis])
    if output_filename != "":
        _write_frames(format_filename, output_filename, out_data)
    return out_data


def _write_xyz_to_bvh(xyz_motion, skeleton, non_end_bones, format_filename, output_filename="", rotations_order="zxy"):
    """Prepares the motion to be written in the bvh file.
    Calls write_frames, to write the motion."""
    bvh_vec_length = len(non_end_bones) * 3 + 6

    out_data = np.zeros([len(xyz_motion), bvh_vec_length])
    for i in range(1, len(xyz_motion)):
        positions = xyz_motion[i]
        rotation_matrices, rotation_angles = helper.xyz_to_rotations_debug(skeleton, positions, joint_names[0],
                                                                           rotations_order)
        if rotations_order == 'zxy':
            o = [2, 0, 1]
        elif rotations_order == 'zyx':
            o = [2, 1, 0]
        new_motion1 = helper.rotation_dic_to_vec(rotation_angles, non_end_bones, positions, joint_names[0], o)

        new_motion = np.array([round(a, 6) for a in new_motion1])
        new_motion[0:3] = new_motion1[0:3]

        out_data[i, :] = np.transpose(new_motion[:, np.newaxis])
    if output_filename != "":
        _write_frames(format_filename, output_filename, out_data)

    return out_data


def prepare_xyz_positions_from_prediction(train_data):
    """
    For debugging reasons.
    """
    seq_length = train_data.shape[0]
    xyz_motion = []
    for frame in range(seq_length):
        xyz_frame = []
        data = np.array([round(a, 6) for a in train_data[frame]])
        # position = data_vec_to_position_dic(data, skeleton)
        data = data * 100
        hip_pos = data[0:3]
        for i, joint in enumerate(joint_names):
            if joint != joint_names[0]:
                xyz_frame.extend(data[i * 3:i * 3 + 3] + hip_pos)
            else:
                xyz_frame.extend(data[i * 3:i * 3 + 3])

        xyz_motion.append(xyz_frame)
    return np.array(xyz_motion)


def write_traindata_positions_to_bvh(bvh_filename="", train_data=[], format_filename=None):
    """This function is called during at the end of a training step,
    to save the data into the bvh file."""
    seq_length = train_data.shape[0]
    xyz_motion = []
    if format_filename is None:
        format_filename = standard_bvh_file
    for i in range(seq_length):
        data = np.array([round(a, 6) for a in train_data[i]])
        position = data_vec_to_position_dic(data, skeleton)
        xyz_motion.append(position)
    return _write_xyz_to_bvh(xyz_motion, skeleton, non_end_bones, format_filename, bvh_filename, rotations_order='zxy')


def write_traindata_positions_to_bvhn(bvh_filename="", train_data=[], format_filename=None, bvh_rotations_order='zxy'):
    """This function is called during at the end of a training step,
    to save the data into the bvh file."""
    seq_length = train_data.shape[0]
    if format_filename != None:
        skeleton, non_end_bones = read_bvh_hierarchy.read_bvh_hierarchy(format_filename)

    xyz_motion = []
    if format_filename is None:
        format_filename = standard_bvh_file
    for i in range(seq_length):
        data = np.array([round(a, 6) for a in train_data[i]])
        position = data_vec_to_position_dic(data, skeleton)
        xyz_motion.append(position)
    return _write_xyz_to_bvh(xyz_motion, skeleton, non_end_bones, format_filename, bvh_filename, bvh_rotations_order)


def data_vec_to_position_dic(data, skeleton):
    # data=data*100
    hip_pos = data[joint_index[joint_names[0]] * 3:joint_index[joint_names[0]] * 3 + 3]
    positions = {}
    for joint in joint_index:
        positions[joint] = data[joint_index[joint] * 3:joint_index[joint] * 3 + 3]
    for joint in positions.keys():
        if joint == joint_names[0]:
            positions[joint] = positions[joint]
        else:
            positions[joint] = positions[joint] + hip_pos
    return positions


def data_vec_to_position_dicn(data, skeleton):
    data = data
    joint_index = {}
    i = 0
    for j in skeleton.keys():
        joint_index[j] = i
        i += 1
    # hip_pos = data[joint_index[joint_names[0]] * 3:joint_index[joint_names[0]] * 3 + 3]
    hip_pos = data[:3]
    positions = {}
    for joint in joint_index:
        if skeleton[joint]['channels'] == []:
            continue
        else:
            positions[joint] = data[joint_index[joint] * 3:joint_index[joint] * 3 + 3]
    for joint in positions.keys():
        if joint == joint_names[0]:
            positions[joint] = positions[joint]
        else:
            positions[joint] = positions[joint] + hip_pos
    return positions


def vector2string(data):
    s = ' '.join(map(str, data))
    return s


def vectors2string(data):
    s = '\n'.join(map(vector2string, data))
    return s
