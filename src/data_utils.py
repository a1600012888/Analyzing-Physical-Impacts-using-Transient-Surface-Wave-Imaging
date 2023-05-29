"""
Description: code for loading processed tilts signals 
Author: Tianyuan Zhang
"""

import numpy as np
import hdf5pickle
import os

try:
    from .tilt_signal import Tilt2D, AngleMapping
except:
    from tilt_signal import Tilt2D, AngleMapping


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
    npz_path,
    len_first_clip=80,
    use_optical_scaling=True,
    shift_signal=True,
    use_angle_mapping=True,
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

    if DataDict["angle_mapping_cfg"] is not None and use_angle_mapping:
        angle_map_cfg = DataDict["angle_mapping_cfg"]
        angle_mapping_obj = AngleMapping(DataDict["angle_mapping_cfg"])
    else:
        angle_mapping_obj = AngleMapping()

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
        angle_mapping_obj,
    )


def load_npz_array_dict(npz_path: str):
    # not a dict now. conver to dict
    DataDict = np.load(npz_path, allow_pickle=True)
    NewDict = {}
    for key in DataDict.keys():
        NewDict[key] = DataDict[key]

    return NewDict
