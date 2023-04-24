from hashlib import new
from turtle import forward
import numpy as np
import hdf5pickle
import os

try:
    from .tilt_signal import Tilt2D
except:
    from tilt_signal import Tilt2D


def read_recovery(data_path: str, perserve_key=None, remove_key=None, if_flatten=False):
    """
    Args:
        data_path
        preserve_key: str
        remove_key: str
    """
    data = hdf5pickle.load(data_path)

    # below are three numpy array
    x_meas_vec = np.array(data["x_meas_vec"])
    y_meas_vec = np.array(data["y_meas_vec"])
    data_points = np.array(data["data_points"])

    if perserve_key is not None:
        mask = [perserve_key in name_ for name_ in data_points]
    else:
        mask = [True] * len(data_points)

    mask = np.array(mask, dtype=np.bool)

    if remove_key is not None:
        remove_mask = [remove_key not in name_ for name_ in data_points]
    else:
        remove_mask = [True] * len(data_points)
    remove_mask = np.array(remove_mask, dtype=np.bool)

    mask = np.logical_and(mask, remove_mask)

    x_meas_vec = x_meas_vec[mask]
    y_meas_vec = y_meas_vec[mask]
    data_points = data_points[mask]

    # flatten x and y if necessary

    if if_flatten:
        if x_meas_vec.ndim == 4:
            print(
                "Loading data: \n  => x_meas_vec has shape of {}".format(
                    x_meas_vec.shape
                ),
                "reshape it to [num_exp, num_marker, num_rows]",
            )
            num_signal, num_marker, num_frame, _ = x_meas_vec.shape
            x_meas_vec = x_meas_vec.reshape(num_signal, num_marker, -1)
            y_meas_vec = y_meas_vec.reshape(num_signal, num_marker, -1)

    return x_meas_vec, y_meas_vec, data_points


def read_repeat(data_path, gt_loc=None):
    data = hdf5pickle.load(data_path)

    # name is just the index of the repeat
    num_repeat = len(data["data_points"])

    # shape [num_repeat, num_marker, N]
    x_meas_vec = np.array(data["x_meas_vec"])
    y_meas_vec = np.array(data["y_meas_vec"])

    return (
        x_meas_vec,
        y_meas_vec,
    )


def filter_out_early_signals(signal_list_list, gt_list, num_clear=100):
    ret_list = []
    ret_gt_list = []
    for i, signal_list in enumerate(signal_list_list):
        save = True

        for signal in signal_list:

            x = (np.abs(np.array(signal[0])) > 1.5).astype(np.float32)

            start_x = np.nonzero(x)
            if len(start_x[0]) > 0:
                start_x = start_x[0][0]
            else:
                start_x = 100000

            y = (np.abs(np.array(signal[1])) > 5.0).astype(np.float32)
            start_y = np.nonzero(y)
            if len(start_y[0]) > 0:
                start_y = start_y[0][0]
            else:
                start_y = 100000

            if start_x < num_clear or start_y < num_clear:
                save = False
                break

        if save:
            ret_list.append(signal_list)
            ret_gt_list.append(gt_list[i])

    print(
        "Input {} signals.  Output {} signals".format(
            len(signal_list_list), len(ret_list)
        )
    )

    return ret_list, ret_gt_list


def filter_out_late_signals(signal_list_list, gt_list, num_clear=200):
    ret_list = []
    ret_gt_list = []
    for i, signal_list in enumerate(signal_list_list):
        save = True

        for signal in signal_list:

            x = (np.abs(np.array(signal[0])) > 0.8).astype(np.float32)

            start_x = np.nonzero(x)
            if len(start_x[0]) > 0:
                start_x = start_x[0][0]
            else:
                start_x = 100000

            y = (np.abs(np.array(signal[1])) > 1.0).astype(np.float32)
            start_y = np.nonzero(y)
            if len(start_y[0]) > 0:
                start_y = start_y[0][0]
            else:
                start_y = 100000

            start_time = min(start_x, start_y)
            valid_size = len(signal[0]) - start_time
            if valid_size < num_clear:
                save = False
                break

        if save:
            ret_list.append(signal_list)
            ret_gt_list.append(gt_list[i])

    print(
        "Remove late start signals. Input {} signals.  Output {} signals".format(
            len(signal_list_list), len(ret_list)
        )
    )

    return ret_list, ret_gt_list


def filter_out_early_signal_single(signal_list, gt_list, num_clear=100):
    ret_list = []
    ret_gt_list = []

    for i, signal in enumerate(signal_list):

        x = (np.abs(np.array(signal[0])) > 1.5).astype(np.float32)

        start_x = np.nonzero(x)
        if len(start_x[0]) > 0:
            start_x = start_x[0][0]
        else:
            start_x = 100000

        y = (np.abs(np.array(signal[1])) > 5.0).astype(np.float32)
        start_y = np.nonzero(y)
        if len(start_y[0]) > 0:
            start_y = start_y[0][0]
        else:
            start_y = 100000

        if start_x < num_clear or start_y < num_clear:
            continue

        ret_list.append(signal)
        ret_gt_list.append(gt_list[i])

    print(
        "Filter out early start signals!. Input {} signals.  Output {} signals".format(
            len(signal_list), len(ret_list)
        )
    )

    return ret_list, ret_gt_list


def filter_out_late_signal_single(signal_list, gt_list, num_clear=200):
    ret_list = []
    ret_gt_list = []

    for i, signal in enumerate(signal_list):

        signal_len = len(signal[0])
        x = (np.abs(np.array(signal[0])) > 1.5).astype(np.float32)

        start_x = np.nonzero(x)
        if len(start_x[0]) > 0:
            start_x = start_x[0][0]
        else:
            start_x = 100000

        y = (np.abs(np.array(signal[1])) > 2.0).astype(np.float32)
        start_y = np.nonzero(y)
        if len(start_y[0]) > 0:
            start_y = start_y[0][0]
        else:
            start_y = 100000

        start_time = min(start_x, start_y)
        valid_size = signal_len - start_time
        if valid_size < num_clear:
            continue

        ret_list.append(signal)
        ret_gt_list.append(gt_list[i])

    print(
        "=> Filter out late start signals. Input {} signals.  Output {} signals".format(
            len(signal_list), len(ret_list)
        )
    )

    return ret_list, ret_gt_list


def read_titls_simulation_single(txt_path):
    with open(txt_path, "r") as f:
        lines = f.readlines()

    lines = lines[9:]

    time_list = []
    x_tilt_list = []
    y_tilt_list = []

    for line in lines:
        str_list = line.split()

        time_list.append(float(str_list[0]))
        x_tilt_list.append(float(str_list[1]))
        y_tilt_list.append(float(str_list[2]))

    return time_list, x_tilt_list, y_tilt_list


def read_simulation_dir(dir_path):
    """
    marker-{i}.txt
    """

    angle_list = []
    time_list = []

    x_tilt_list_list = []
    y_tilt_list_list = []

    num_marker_file = len([a for a in os.listdir(dir_path) if a.endswith("txt")]) - 1
    # print(os.listdir(dir_path), num_marker_file)

    angle_file_path = os.path.join(dir_path, "angle.txt")

    assert os.path.exists(angle_file_path), "angle file not existed!!"

    with open(angle_file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        theta = float(line.strip())
        angle_list.append(theta)

    for i in range(num_marker_file):
        txt_path = os.path.join(dir_path, "marker-{}.txt".format(i + 1))

        time_list, x_tilt_list, y_tilt_list = read_titls_simulation_single(txt_path)

        x_tilt_list_list.append(x_tilt_list)
        y_tilt_list_list.append(y_tilt_list)

    return (
        np.array(angle_list),
        np.array(x_tilt_list_list),
        np.array(y_tilt_list_list),
        np.array(time_list),
    )


def read_simulation_raw(path):

    assert os.path.exists(path), "angle file not existed!!"

    with open(path, "r") as f:
        lines = f.readlines()

    lines = lines[9:]

    num_signal = (len(lines[1].split()) - 1) // 2
    # print(num_signal, lines[0])
    time_list = []
    x_tilt_list = []
    y_tilt_list = []
    for i in range(num_signal):
        x_tilt_list.append([])
        y_tilt_list.append([])

    for line in lines:
        str_list = line.split()

        time_list.append(float(str_list[0]))
        for i in range(num_signal):
            x_tilt_list[i].append(float(str_list[2 * i + 1]))
            y_tilt_list[i].append(float(str_list[2 * i + 2]))

    print(
        "Reading {} signals with {} time stampes.  Last time {} us".format(
            num_signal, len(time_list), time_list[-1]
        )
    )
    return np.array(time_list), np.array(x_tilt_list), np.array(y_tilt_list)


###### --------------------------------------------------------- ######
# For a processed data format

# save_dict = {
#     # save to raw signal ( no scalling applied
#     'signal_x': None,                 # [num_exp, num_marker, num_rows_from_camera]
#     'signal_y': None,                 # [num_exp, num_marker, num_rows_from_camera]
#     'roi_x': None,                    # [num_exp, num_marker, roi_window_size]
#     'roi_y': None,                    # [num_exp, num_marker, roi_window_size]
#     'roi_offset': None,               # int. indicating the offset w.r.t start of signal.
#     'optical_scaling': None,          # int
#     'predicted_ratios': None,         # [num_exp, num_marker, ]
#     'suggested_theta_cfg': None,      # suggested theta cfg for
#     'angle_mapping_cfg': None,        # List[ dict() ]
#     'marker_locations': None,         # [num_markers, 2]
#     'name_list': None,                # [num_exp, ].  List of str
#     'gt_locations_list': None,        # [num_exp, 2]
#     'gt_loc_list': None,              # [num_points, 2].  List of target points
#     'searched_start_index': None,     # [num_exp, num_marker]
# }


def generate_saving_format(
    all_signal_list,
    m_locations,
    _gt_locations_list,
    _gt_loc_list,
    name_list,
    optical_scalling,
    angle_mapping_cfg=None,
    suggested_theta_cfg=None,
    searched_start_index=None,
):

    save_dict = {}

    roi_cfg = {
        "if_normalize": True,
        "offset": 0,
        "thre": -0.03,
        "std_coe": 12.0,
        "smoothing_wsize": 4,
        "window_size": 180,
        "pre_size": 30,
        "safe_interval": 30,
        "offset": -20,
    }

    save_dict["roi_offset"] = roi_cfg["offset"]
    save_dict["suggested_theta_cfg"] = suggested_theta_cfg
    save_dict["angle_mapping_cfg"] = angle_mapping_cfg
    save_dict["marker_locations"] = m_locations
    save_dict["name_list"] = name_list
    save_dict["gt_locations_list"] = np.array(_gt_locations_list)
    save_dict["gt_loc_list"] = np.array(_gt_loc_list)

    save_dict["optical_scaling"] = optical_scalling

    save_dict["searched_start_index"] = searched_start_index

    signal_x_list, signal_y_list = [], []
    roi_x_list, roi_y_list = [], []
    pred_ratio_list = []

    for i, signal_list_ in enumerate(all_signal_list):
        signal_x_list.append([])
        signal_y_list.append([])
        roi_x_list.append([])
        roi_y_list.append([])
        pred_ratio_list.append([])

        for exp_ind, signal_ in enumerate(signal_list_):
            signal_x_list[i].append(signal_[0])
            signal_y_list[i].append(signal_[1] * optical_scalling)

            x_roi, y_roi = signal_.get_roi_renorm(**roi_cfg)
            roi_x_list[i].append(x_roi)
            roi_y_list[i].append(y_roi * optical_scalling)

            if suggested_theta_cfg is not None:
                pred_ratio_list[i].append(signal_.get_theta(**suggested_theta_cfg))

        for t_list in [signal_x_list, signal_y_list, roi_x_list, roi_y_list]:
            t_list[i] = np.stack(t_list[i], axis=0)

    for name, t_list in zip(
        ["signal_x", "signal_y", "roi_x", "roi_y"],
        [signal_x_list, signal_y_list, roi_x_list, roi_y_list],
    ):
        t_array = np.stack(t_list, axis=0)

        save_dict[name] = t_array

    if suggested_theta_cfg is not None:
        save_dict["predicted_ratios"] = np.array(pred_ratio_list)

    for name, value in save_dict.items():
        print(name, " -- type of -- {}".format(type(value)))
        if isinstance(value, np.ndarray):
            print("    {}: {}".format(name, value.shape))

    return save_dict


def load_from_processed(
    npz_path, len_first_clip=80, use_optical_scaling=True, shift_signal=True
):

    DataDict = np.load(npz_path, allow_pickle=True)

    # change to dict first
    keys = [
        "roi_offset",
        "suggested_theta_cfg",
        "marker_locations",
        "name_list",
        "gt_locations_list",
        "gt_loc_list",
        "optical_scaling",
        "signal_x",
        "signal_y",
        "roi_x",
        "roi_y",
        "predicted_ratios",
        "searched_start_index",
        "corner_locations",
        "height_list",
    ]

    dict_keys = ["angle_mapping_cfg"]
    NewDict = {}
    for key in keys:
        if key in DataDict:
            NewDict[key] = DataDict[key]
        else:
            NewDict[key] = None
    for key in dict_keys:
        if key in DataDict:
            NewDict[key] = DataDict[key].item()
        else:
            NewDict[key] = None
    # NewDict = {k_: DataDict[k_] for k_ in keys}
    DataDict = NewDict

    signal_x, signal_y = DataDict["signal_x"], DataDict["signal_y"]

    roi_x, roi_y = DataDict["roi_x"], DataDict["roi_y"]

    name_list = DataDict["name_list"]

    marker_locations = DataDict["marker_locations"]
    gt_locations_list = DataDict["gt_locations_list"]
    gt_loc_list = DataDict["gt_loc_list"]

    # optical_scaling = DataDict['optical_scaling']

    if DataDict["suggested_theta_cfg"] is not None:
        DataDict["suggested_theta_cfg"] = DataDict["suggested_theta_cfg"].item()

    all_signal_list = []

    optical_scaling = DataDict["optical_scaling"]
    # print(type(optical_scaling), optical_scaling)
    if optical_scaling == None or (not use_optical_scaling):
        optical_scaling = 1

    for i in range(signal_x.shape[0]):
        all_signal_list.append([])
        for marker_ind in range(signal_x.shape[1]):
            if shift_signal:
                all_signal_list[i].append(
                    Tilt2D(
                        [
                            signal_x[i][marker_ind],
                            signal_y[i][marker_ind] / optical_scaling,
                        ],
                        len_first_clip=len_first_clip,
                    ).shift_signal(
                        pre_size=40,
                        safe_interval=50,
                        std_coe=10.0,
                        forward_window_size=30,
                        smoothing_wsize=10,
                    )
                )
            else:
                all_signal_list[i].append(
                    Tilt2D(
                        [
                            signal_x[i][marker_ind],
                            signal_y[i][marker_ind] / optical_scaling,
                        ],
                        len_first_clip=len_first_clip,
                    )
                )

    print("=> Loading")
    print("Signal X of shape : {}".format(signal_x.shape))

    print("Names: ", name_list)

    return (
        all_signal_list,  # List[List[Tilt2D]].  [num_exp, num_marker, ...]
        marker_locations,
        gt_locations_list,
        gt_loc_list,
        name_list,
        DataDict,
    )


def load_npz_array_dict(npz_path: str):

    # not a dict now. conver to dict
    DataDict = np.load(npz_path, allow_pickle=True)
    NewDict = {}
    for key in DataDict.keys():
        NewDict[key] = DataDict[key]

    return NewDict


###### --------------------------------------------------------- ######
