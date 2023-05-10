"""
Description: code for performing back projection given marker locations, signals, and camera calibration parameters
Author: Tianyuan Zhang
"""

import numpy as np
from scipy.optimize import least_squares
from typing import List, Tuple, Dict, Callable
import cv2
import os
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec

try:
    from .tilt_signal import Tilt2D
    from .plot_utils import plot_signals, plot_signals_roi, plot_points
    from .angle_utils import Name2AngleFunc

except:
    from tilt_signal import Tilt2D
    from plot_utils import plot_signals, plot_signals_roi, plot_points
    from angle_utils import Name2AngleFunc

import matplotlib.pyplot as plt
import copy
from tqdm import tqdm


class BackProjSolver(object):
    def __init__(
        self,
        marker_locations=None,
        marker_scaling=10,
        canvas_size=[200, 200],
        theta_cfg=dict(
            std_coe=20.0,
            smoothing_wsize=-1,
            forward_window_size=40,
            if_normalize=False,
            start_index=-1,
            offset=0,
            window_size=80,
            stability_window_size=50,
            pre_size=-1,
            safe_interval=10,
            weighting_cfg={
                "name": "poly",
                "params": [1.0],
            },  # Linear backprojection, no filtering
            # weighting_cfg=dict(name="exp"), # exponential tonemapping for each frame after backprojection
            drawing_cfg=dict(
                name="weighted_cone",
                fov=6,
                thickness=-1,
                weighting_by_scale=False,
                weighting_by_distance=False,
                weighting_by_undertanity=False,
                cone_weighting=0.04,
            ),
            global_start=False,
        ),
        angle_mapping_obj=None,
        corner_locations=None,
        marker_offset=np.array([0, 0]),
        marker_center=None,
    ):
        """
        Args:
            marker_locations: int array of shape [num_of_marker, 2]
                indicate the coordinates of markers.
            marker_scaling: int
                scaling factor for marker locations.
            canvas_size: int array of shape [2, ]
                [H, W], denote the size of the canvas.
                larger size will result in higher computational cost
            cornerl_locations: int array of shape [2, 2]
                default as None.  Used for visualize the borders of the material board.
                [x, y] coordinates for upper left and bottom right of the borders.
                in the same unit/frame as marker_locations
            marker_offset: int array of shape [2, ]
                used to shift the marker locations on the canvas
            marker_center: float array of shape [2, ]
                used to shift the marker locations on the canvas
            angle_mapping_obj:
                used to represent the camera and material caliberation
                it takes the ratio of y_tilts and x_tilts, output the angle for which to back project
            for weighting_cfg:
                name in [exp, poly, ]
                e.g. {name: 'poly', params: [2]} => x**2

            theta_cfg: dict
                collection of hyperparameters for
                    (1) detecting the arrival of the transient surface waves, as described in Eq(13) in the main paper
                    (2) extracting the stable intervals as defined in section 5 of the paper
                    (3) backprojections
                I list the most important parameters below. The rest of the parameters are not recommend to change.

                std_coe: float
                    used to determine the threshold for detecting the arrival of the transient surface waves
                smoothing_wsize: window size for gaussian smoothing
                    perform gaussian smoothing on the signal before detecting the arrival of the transient surface waves
                    -1 means no smoothing
                if_normalize:
                    normalize the signal so that the magnitude of the signal at the beginning frames are 0
                start_index: int
                    the index of the first frame to perform backprojection
                    -1 means using thresholding to determine the first frame automatically
                offset: int
                    the offset of the first frame to perform backprojection
                    e.g. if start_index = 10, offset = 5, then the first frame to perform backprojection is 15
                window_size: int
                    the size of the window to perform backprojection. (The length of stable interval period)
                global_start: bool, default False
                    if True the start frame for each marker is the same
        """

        # initialize the canvas
        self.canvas_size = np.array(canvas_size)
        self.canvas = np.zeros(canvas_size)

        # initialize the coordiates of all markers on the canvas
        self.marker_locations_original = marker_locations
        self.marker_scaling = marker_scaling

        self.marker_locations = self._init_marker_locations(
            marker_locations,
            marker_scaling,
            corner_locations,
            marker_offset,
            marker_center=marker_center,
        )
        self.marker_offset = marker_offset

        self.theta_cfg = theta_cfg

        # the weighting function for each frame, like filtered backprojection
        self.name2weightfunc = {
            "exp": exp_weighting_func,
            "exp_relu": exp_relu_weighting_func,
            "poly": poly_weighting_func,
            "poly_relu": poly_relu_weighting_func,
        }

        # self.corner_locations = corner_locations

        # account for the camera and material caliberation
        self.angle_mapping_obj = angle_mapping_obj

    def _init_marker_locations(
        self,
        marker_locations,
        marker_scalling,
        corner_locations=None,
        marker_offset=np.array([0, 0]),
        marker_center=None,
    ):

        canvas_center = self.canvas_size / 2
        self.canvas_center = canvas_center

        # [N, 2] => [2]
        if marker_center is None:
            marker_center = np.mean(marker_locations, axis=0)
        print(marker_center)

        # save this term to re-normalize the gt later!
        self.marker_center = marker_center

        marker_loc_normalized = (marker_locations - marker_center) * marker_scalling

        marker_locations_ret = marker_loc_normalized + canvas_center + marker_offset

        # transform the corner locations too
        if corner_locations is not None:
            corner_locations_norm = (corner_locations - marker_center) * marker_scalling
            self.corner_locations = (
                corner_locations_norm + canvas_center + marker_offset
            )
        else:
            self.corner_locations = None

        print("corner locations on canvas. tl. br")
        print(self.corner_locations)

        return marker_locations_ret

    def pred(
        self,
        signal_list,
        start_index_list=None,
        theta_cfg=None,
        auto_optimize=True,
        debug=False,
    ):
        """
        Args:
            signal_list: List[Tilt2D]
                list of signals measured by each marker
            start_index_list: List[int], default None
                if specificed, it denotes the arrival time of the transient surface waves for each marker
                if None, it will be determined automatically using thresholding
            theta_cfg: dict, default None
                if specificed, it will overwrite the default self.theta_cfg
            auto_optimize: bool, default True
            debug: bool, default False
                if True, it will print extra information for debugging

        Returns:
            pred: np.array of shape [2]
                the predicted location of the source of the transient surface wave
            ret_img: np.array of shape [H, W]
                the heatmap of the backprojection, has the same size as the canvas_size in init
        """

        # override the default theta_cfg
        _cfg = copy.deepcopy(self.theta_cfg)
        if theta_cfg is not None:
            for name, value in theta_cfg.items():
                _cfg[name] = value

        drawing_cfg = _cfg["drawing_cfg"]
        if _cfg["smoothing_wsize"] > 0:
            new_signal_list = []
            for _signal in signal_list:
                new_signal_list.append(
                    _signal.get_smoothed_signal(_cfg["smoothing_wsize"])
                )

            signal_list = new_signal_list

        if (
            "weighting_by_uncertainty" in _cfg["drawing_cfg"]
            and _cfg["drawing_cfg"]["weighting_by_uncertainty"]
        ):

            stavility_ind_offset = (_cfg["stability_window_size"] + 1) // 2
            # get uncertainty
            stability_list_list = []
            for signal_ in signal_list:
                rec_mse = signal_.get_stability(_cfg["stability_window_size"])

                stability_weighting = np.exp(rec_mse * -10.0)
                stability_list_list.append(stability_weighting)
        else:
            stability_list_list = None

        # get the arrival time of the transient surface wave using thresholding
        pred_start_index_list = []
        if start_index_list is None:
            start_index_list = [None] * len(signal_list)
        for i, _signal in enumerate(signal_list):

            start_index = start_index_list[i]
            # if start_index is not specified, use the thresholding method to determine it
            if start_index is None:
                start_index = _signal.get_start_index(
                    _cfg["std_coe"],
                    _cfg["smoothing_wsize"],
                    _cfg["forward_window_size"],
                    True,
                )
                pred_start_index_list.append(start_index)

        if start_index_list[0] is None:
            start_index_list = pred_start_index_list

        start_index_list = np.array(start_index_list)

        # if global_start is True, use the median of all start_index_list as the start_index for all markers
        if _cfg["global_start"]:
            start_index = np.median(start_index_list).astype(np.int32)
            start_index_list = [
                start_index,
                start_index,
                start_index,
                start_index,
                start_index,
            ]

        if auto_optimize:
            start_array = np.array(start_index_list)
            if start_array.max() - start_array.min() > 200:
                if debug:
                    print("Use global start index might be better, ", start_index_list)
                start_index = np.median(start_index_list).astype(np.int32)
                start_index_list = [
                    start_index,
                    start_index,
                    start_index,
                    start_index,
                    start_index,
                ]

        if debug:
            print(start_index_list)

        offset = _cfg["offset"]
        window_size = _cfg["window_size"]

        # extract params for weighting_cfg
        weighting_cfg = _cfg["weighting_cfg"]
        weighting_func_name = weighting_cfg["name"]
        weighting_param_config = {}
        for name, value in weighting_cfg.items():
            if name == "name":
                continue
            weighting_param_config[name] = value

        ret_img = np.zeros(self.canvas_size)

        # add normalized signal here
        tmp_new_signal_list = []
        if "pre_size" in _cfg and _cfg["pre_size"] > 0:
            pre_size = _cfg["pre_size"]
            safe_interval = _cfg["safe_interval"]
            for _signal, start_index in zip(signal_list, start_index_list):

                pre_region_x = _signal[0][
                    start_index - pre_size - safe_interval : start_index - safe_interval
                ]
                pre_region_y = _signal[1][
                    start_index - pre_size - safe_interval : start_index - safe_interval
                ]

                pre_x_mean = np.mean(pre_region_x)
                pre_y_mean = np.mean(pre_region_y)

                new_signal = Tilt2D([_signal[0] - pre_x_mean, _signal[1] - pre_y_mean])
                tmp_new_signal_list.append(new_signal)

            signal_list = tmp_new_signal_list

        # Begin looping over the stable ratio interval for backprojections (begin draw canvas)
        for i in range(offset, offset + window_size):

            ratio_list = []
            scale_list = []
            stability_list = []

            if_frame_ends = False
            for j in range(len(signal_list)):
                _signal = signal_list[j]
                frame_ind = start_index_list[j] + i
                if frame_ind > len(_signal[1]) - 1:
                    print("Frame ind exceds length, start list: ", start_index_list)
                    if_frame_ends = True
                    break
                y_tilt = _signal[1][frame_ind]
                x_tilt = _signal[0][frame_ind]
                ratio = y_tilt / x_tilt
                scale = np.sqrt(y_tilt**2 + x_tilt**2)
                ratio_list.append(ratio)

                scale_list.append(scale)
                if stability_list_list is not None:
                    stability_list.append(
                        stability_list_list[j][frame_ind - stavility_ind_offset]
                    )
                else:
                    stability_list = None

            if if_frame_ends:
                break

            ratio_list = self.angle_mapping_obj(ratio_list)

            # backprojection for one frame of measurement
            tmp_img = self._draw_one_frame(
                ratio_list,
                self.marker_locations,
                self.canvas_size,
                scale_list,
                stability_list,
                drawing_cfg,
            )

            tmp_img = self.name2weightfunc[weighting_func_name](
                tmp_img, **weighting_param_config
            )
            ret_img = ret_img + tmp_img

        # argmax to get the location of the prediction
        pred_yx = np.array(np.unravel_index(np.argmax(ret_img), ret_img.shape))
        pred_xy = np.array([pred_yx[1], pred_yx[0]])

        pred_xy = (
            pred_xy - self.marker_offset - self.canvas_center
        ) / self.marker_scaling + self.marker_center
        return pred_xy, ret_img

    def _draw_one_frame(
        self,
        ratio_list,
        marker_locations,
        canvas_size,
        scale_list,
        stability_list,
        drawing_cfg,
    ):
        """
        Perform backprojection for one frame of measurement

        Args:
            ratio_list: the ratio between y_tilt and x_tilt for each marker.
                the ratio is calibrated to account for cameras and material anisotropy
                thus, the ratio is used as the prediction of direction towards the gt location,
                as described in Eq (8), (16) in the main paper.
            marker_locations: the location of the markers in the canvas
            canvas_size: the size of the canvas
            scale_list, and stability_list: the weightinf factor for each marker. By default it is all the same for each marker.
                if you want to account for the uncertainty and nosiness of each measured signal, you can specify
                the values of scale_list.
            drawing_cfg: the configuration for drawing the backprojection.
                with the following options: ["line", "cone", "weighted_cone"]
                line: draw a line passing each marker location with the slope specified by the ratio_list
                cone: draw a cone passing (centering at) each marker location with the slope specified by the ratio_list
                weighted_cone: draw a cone passing (centering at) each marker location with the slope specified by the ratio_list
                    The cone is weighted by the distance to the apex line of the cone.

                weighted_cone is the recommended option.

        """
        # extract params for weighting_cfg
        drawing_func_name = drawing_cfg["name"]
        drawing_param_config = {}
        for name, value in drawing_cfg.items():
            if name == "name":
                continue
            drawing_param_config[name] = value

        if drawing_func_name == "line":
            ret_img = self._line_fill(
                ratio_list,
                marker_locations,
                canvas_size,
                scale_list,
                **drawing_param_config,
            )
        elif drawing_func_name == "cone":
            ret_img = self._cone_fill(
                ratio_list,
                marker_locations,
                canvas_size,
                scale_list,
                stability_list,
                **drawing_param_config,
            )
        elif drawing_func_name == "weighted_cone":
            ret_img = self._weighted_cone_fill(
                ratio_list,
                marker_locations,
                canvas_size,
                scale_list,
                stability_list,
                **drawing_param_config,
            )
        else:
            raise NotImplementedError

        return ret_img

    def _line_fill(
        self,
        ratio_list,
        marker_locations,
        canvas_size,
        scale_list,
        thickness,
        weighting_by_scale=True,
        weighting_by_distance="sqrt",
    ):
        """
        Draw a line passing each marker location with the slope specified by the ratio_list
        Called by _draw_one_frame if drawing_cfg["name"] == "line"

        Args:
            thickness: the thickness of the line, passed to cv2.line
        """

        ret_img = np.zeros(canvas_size)

        cords_map = np.mgrid[: canvas_size[0], : canvas_size[1]].transpose(0, 2, 1)

        for i in range(len(ratio_list)):
            ratio = ratio_list[i]
            marker_loc = marker_locations[i]
            scale_ = scale_list[i]

            y_x_eq_0 = marker_loc[1] - ratio * marker_loc[0]
            y_x_eq_bound = (
                ratio * canvas_size[1] + marker_loc[1] - ratio * marker_loc[0]
            )

            tmp_img = np.zeros(canvas_size)

            cv2.line(
                tmp_img,
                [0, int(y_x_eq_0)],
                [canvas_size[1], int(y_x_eq_bound)],
                thickness=thickness,
                color=1,
            )

            if weighting_by_scale is not False:
                if weighting_by_scale == "sqrt":
                    tmp_img = tmp_img * np.sqrt(scale_)
                elif weighting_by_scale == "log":
                    tmp_img = tmp_img * np.log(scale_ + 1)
                else:
                    raise NotImplementedError

            if weighting_by_distance is not False:
                distance_map = cords_map - marker_loc[:, np.newaxis, np.newaxis]
                distance_map = np.sqrt(np.sum(distance_map**2, axis=0))

                canvas_length = np.sqrt(canvas_size[0] ** 2 + canvas_size[1] ** 2)
                distance_map = distance_map / canvas_length * 10.0

                if weighting_by_distance == "sqrt":
                    distance_map = np.sqrt(distance_map)
                    tmp_img = tmp_img * distance_map
                elif weighting_by_distance == "linear":
                    tmp_img = tmp_img * distance_map
                elif weighting_by_distance == "log":
                    tmp_img = tmp_img * np.log(distance_map + 1)
                else:
                    raise NotImplementedError

            ret_img = ret_img + tmp_img

        return ret_img

    def _cone_fill(
        self,
        ratio_list,
        marker_locations,
        canvas_size,
        scale_list,
        stability_list,
        fov=10,
        thickness=4,
        weighting_by_scale=False,
        weighting_by_distance=False,
        weighting_by_uncertainty=True,
    ):
        """
        Draw a cone passing (centering at) each marker location with the slope specified by the ratio_list
        Called by _draw_one_frame if drawing_cfg["name"] == "cone"

        Args:
            fov: the field of view of the cone, in degree
            thickness: not used
        """
        ret_img = np.zeros(canvas_size)

        # cords_map = np.mgrid[: canvas_size[0], : canvas_size[1]].transpose(0, 2, 1)

        half_angle = np.deg2rad(fov / 2)
        for i in range(len(ratio_list)):
            ratio = ratio_list[i]
            marker_loc = marker_locations[i]
            scale_ = scale_list[i]
            tmp_img = np.zeros(canvas_size)

            arc_theta = np.arctan(ratio)
            ratio_1 = np.tan(arc_theta + half_angle)
            ratio_2 = np.tan(arc_theta - half_angle)

            if ratio_1 * ratio_2 > 0:
                # same sign!
                y_x_eq_0_r1 = (marker_loc[1] - ratio_1 * marker_loc[0]).astype(np.int32)
                y_x_eq_bound_r1 = (
                    ratio_1 * canvas_size[1] + marker_loc[1] - ratio_1 * marker_loc[0]
                ).astype(np.int32)
                y_x_eq_0_r2 = (marker_loc[1] - ratio_2 * marker_loc[0]).astype(np.int32)
                y_x_eq_bound_r2 = (
                    ratio_2 * canvas_size[1] + marker_loc[1] - ratio_2 * marker_loc[0]
                ).astype(np.int32)

                cv2.drawContours(
                    tmp_img,
                    [
                        np.array(
                            [
                                marker_loc.astype(np.int32),
                                [0, y_x_eq_0_r1],
                                [0, y_x_eq_0_r2],
                            ],
                            dtype=np.int32,
                        )
                    ],
                    0,
                    1,  # color
                    -1,
                )
                cv2.drawContours(
                    tmp_img,
                    [
                        np.array(
                            [
                                marker_loc.astype(np.int32),
                                [canvas_size[1], y_x_eq_bound_r1],
                                [canvas_size[1], y_x_eq_bound_r2],
                            ],
                            dtype=np.int32,
                        )
                    ],
                    0,
                    1,
                    -1,
                )
            else:
                x_y_eq_0_r1 = (marker_loc[0] - marker_loc[1] / ratio_1).astype(np.int32)
                x_y_eq_bound_r1 = (
                    marker_loc[0] - marker_loc[1] / ratio_1 + canvas_size[0] / ratio_1
                ).astype(np.int32)
                x_y_eq_0_r2 = (marker_loc[0] - marker_loc[1] / ratio_2).astype(np.int32)
                x_y_eq_bound_r2 = (
                    marker_loc[0] - marker_loc[1] / ratio_2 + canvas_size[0] / ratio_2
                ).astype(np.int32)

                cv2.drawContours(
                    tmp_img,
                    [
                        np.array(
                            [
                                marker_loc.astype(np.int32),
                                [x_y_eq_0_r1, 0],
                                [x_y_eq_0_r2, 0],
                            ],
                            dtype=np.int32,
                        )
                    ],
                    0,
                    1,  # color
                    -1,
                )
                cv2.drawContours(
                    tmp_img,
                    [
                        np.array(
                            [
                                marker_loc.astype(np.int32),
                                [x_y_eq_bound_r1, canvas_size[0]],
                                [x_y_eq_bound_r2, canvas_size[0]],
                            ],
                            dtype=np.int32,
                        )
                    ],
                    0,
                    1,  # color
                    -1,
                )

            if weighting_by_scale is not False:
                if weighting_by_scale == "sqrt":
                    tmp_img = tmp_img * np.sqrt(scale_)
                elif weighting_by_scale == "log":
                    tmp_img = tmp_img * np.log(scale_ + 1)
                else:
                    raise NotImplementedError
            if weighting_by_uncertainty is not False:
                tmp_img = tmp_img * stability_list[i]

            ret_img = ret_img + tmp_img

        return ret_img

    def _weighted_cone_fill(
        self,
        ratio_list,
        marker_locations,
        canvas_size,
        scale_list,
        stability_list,
        fov=10,
        thickness=4,
        weighting_by_scale=False,
        weighting_by_distance=False,
        weighting_by_uncertainty=True,
        cone_weighting=5.0,
    ):
        """
        Draw a weighted cone with a given fov and ratio, the apex line is specifed by marker_locations and ratio_list
        Called by _draw_one_frame if drawing_cfg["name"] == "weighted_cone"

        Args:
            fov: the field of view of the cone, in degree
            thickness: not used
        """
        ret_img = np.zeros(canvas_size)

        # cords_map = np.mgrid[: canvas_size[0], : canvas_size[1]].transpose(0, 2, 1)

        half_angle = np.deg2rad(fov / 2)

        # [x, y] => [2, H, W]
        # cords_map = np.mgrid[: canvas_size[0], : canvas_size[1]].transpose(0, 2, 1)
        cords_map = np.mgrid[: canvas_size[0], : canvas_size[1]][
            ::-1, :, :
        ]  # .transpose(0, 1, 2)

        for i in range(len(ratio_list)):
            ratio = ratio_list[i]
            marker_loc = marker_locations[i]
            scale_ = scale_list[i]
            tmp_img = np.zeros(canvas_size)

            arc_theta = np.arctan(ratio)
            ratio_1 = np.tan(arc_theta + half_angle)
            ratio_2 = np.tan(arc_theta - half_angle)

            if ratio_1 * ratio_2 > 0:
                # same sign!
                y_x_eq_0_r1 = (marker_loc[1] - ratio_1 * marker_loc[0]).astype(np.int32)
                y_x_eq_bound_r1 = (
                    ratio_1 * canvas_size[1] + marker_loc[1] - ratio_1 * marker_loc[0]
                ).astype(np.int32)
                y_x_eq_0_r2 = (marker_loc[1] - ratio_2 * marker_loc[0]).astype(np.int32)
                y_x_eq_bound_r2 = (
                    ratio_2 * canvas_size[1] + marker_loc[1] - ratio_2 * marker_loc[0]
                ).astype(np.int32)

                cv2.drawContours(
                    tmp_img,
                    [
                        np.array(
                            [
                                marker_loc.astype(np.int32),
                                [0, y_x_eq_0_r1],
                                [0, y_x_eq_0_r2],
                            ],
                            dtype=np.int32,
                        )
                    ],
                    0,
                    1,  # color
                    -1,
                )
                cv2.drawContours(
                    tmp_img,
                    [
                        np.array(
                            [
                                marker_loc.astype(np.int32),
                                [canvas_size[1], y_x_eq_bound_r1],
                                [canvas_size[1], y_x_eq_bound_r2],
                            ],
                            dtype=np.int32,
                        )
                    ],
                    0,
                    1,
                    -1,
                )
            else:
                x_y_eq_0_r1 = (marker_loc[0] - marker_loc[1] / ratio_1).astype(np.int32)
                x_y_eq_bound_r1 = (
                    marker_loc[0] - marker_loc[1] / ratio_1 + canvas_size[0] / ratio_1
                ).astype(np.int32)
                x_y_eq_0_r2 = (marker_loc[0] - marker_loc[1] / ratio_2).astype(np.int32)
                x_y_eq_bound_r2 = (
                    marker_loc[0] - marker_loc[1] / ratio_2 + canvas_size[0] / ratio_2
                ).astype(np.int32)

                cv2.drawContours(
                    tmp_img,
                    [
                        np.array(
                            [
                                marker_loc.astype(np.int32),
                                [x_y_eq_0_r1, 0],
                                [x_y_eq_0_r2, 0],
                            ],
                            dtype=np.int32,
                        )
                    ],
                    0,
                    1,  # color
                    -1,
                )
                cv2.drawContours(
                    tmp_img,
                    [
                        np.array(
                            [
                                marker_loc.astype(np.int32),
                                [x_y_eq_bound_r1, canvas_size[0]],
                                [x_y_eq_bound_r2, canvas_size[0]],
                            ],
                            dtype=np.int32,
                        )
                    ],
                    0,
                    1,  # color
                    -1,
                )

            if weighting_by_scale is not False:
                if weighting_by_scale == "sqrt":
                    tmp_img = tmp_img * np.sqrt(scale_)
                elif weighting_by_scale == "log":
                    tmp_img = tmp_img * np.log(scale_ + 1)
                else:
                    raise NotImplementedError
            if weighting_by_uncertainty is not False and stability_list is not None:
                tmp_img = tmp_img * stability_list[i]

            line_dis_map = (
                cords_map[0, ...] * ratio
                - cords_map[1, ...]
                + marker_loc[1]
                - marker_loc[0] * ratio
            )
            line_dis_map = np.abs(line_dis_map) / np.sqrt(1 + ratio**2) + 2e-1
            distance_weight_map = np.exp(-1.0 * cone_weighting * line_dis_map)
            # print(line_dis_map.shape, tmp_img.shape, cords_map.shape, "debug")
            tmp_img = tmp_img * distance_weight_map
            ret_img = ret_img + tmp_img

        return ret_img

    def pred_and_plot(
        self,
        signal_list,
        start_index_list=None,
        theta_cfg=None,
        gt_loc=None,
        save_path=None,
        color_map_range=None,
        auto_optimize=True,
        debug=False,
    ):
        """
        The same as the pred function, but add one extra step to plot the result

        """
        # shift gt
        if not isinstance(gt_loc, np.ndarray):
            gt_loc = np.array(gt_loc)

        img_gt_loc = (
            (gt_loc - self.marker_center) * self.marker_scaling
            + self.canvas_center
            + self.marker_offset
        )

        pred_xy_real_cor, ret_img = self.pred(
            signal_list,
            start_index_list,
            theta_cfg,
            auto_optimize=auto_optimize,
            debug=debug,
        )
        pred_yx = np.unravel_index(np.argmax(ret_img), ret_img.shape)
        pred_xy = np.array([pred_yx[1], pred_yx[0]])

        plt.figure()

        for i, mxy in enumerate(self.marker_locations):
            label = "marker-{}".format(i)
            m_color = "white"
            plt.scatter(mxy[0], mxy[1], color=m_color, label=label, marker="s")
            plt.text(mxy[0], mxy[1], s=label, color=m_color)

        plt.scatter(
            img_gt_loc[0], img_gt_loc[1], color="red", label="knock", marker="P"
        )
        plt.text(img_gt_loc[0], img_gt_loc[1], color="red", s="knock")

        if self.corner_locations is not None:
            rec_color = np.max(ret_img) * 0.8
            cv2.rectangle(
                ret_img,
                self.corner_locations[0].astype(np.int32),
                self.corner_locations[1].astype(np.int32),
                color=rec_color,
                thickness=3,
            )

        plt.scatter(pred_xy[0], pred_xy[1], color="orange", marker="*")
        plt.text(pred_xy[0], pred_xy[1], color="orange", s="pred")
        if color_map_range is None:
            plt.imshow(ret_img)
        else:
            plt.imshow(ret_img, vmin=color_map_range[0], vmax=color_map_range[1])
        ax = plt.gca()
        ax.invert_yaxis()

        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)

        return pred_xy_real_cor, ret_img

    def _crop_shift_signal(self, signal_list, start_index_list):
        """
        Starting from the second start_index_list. Crop and shift the signal

        Args:
            start_index_list: List(list()). shape [num_knock, num_marker]
            signal_list: List(Tilt2D)
        Outs:
            signal_list.  copy then shift
        """

        new_signal_list = [signal_.clone() for signal_ in signal_list]
        signal_list = new_signal_list
        for i in range(1, len(start_index_list)):
            for j in range(len(signal_list)):
                start_ind = start_index_list[i][j]
                signal_list[j][0][start_ind:] = (
                    signal_list[j][0][start_ind:] - signal_list[j][0][start_ind]
                )
                signal_list[j][1][start_ind:] = (
                    signal_list[j][1][start_ind:] - signal_list[j][1][start_ind]
                )

        return signal_list

    def get_video_multiple(
        self,
        signal_list,
        start_index_list,
        theta_cfg=None,
        gt_loc_list=None,
        save_path=None,
        color_map_range=None,
        pred_xy_list=None,
    ):
        """
        Args:
            start_index_list: List(list()). shape [num_knock, num_marker]
            signal_list: List(Tilt2D)
            theta_cfg:
                'window_size'
                'running_window_size'
                drawing_cfg,
            gt_loc_list: List(List).  shape of (num_knock, 2)
        """
        _cfg = copy.deepcopy(self.theta_cfg)
        if theta_cfg is not None:
            for name, value in theta_cfg.items():
                _cfg[name] = value
        save_dir = os.path.dirname(save_path)
        max_img_save_path = save_path[:-4] + "max_pred.png"
        tmp_save_dir = os.path.join(save_dir, "tmp")
        if not os.path.exists(tmp_save_dir):
            os.mkdir(tmp_save_dir)

        if start_index_list is None:
            start_index_list = [[]]
            for i, _signal in enumerate(signal_list):

                start_index = _signal.get_start_index(
                    _cfg["std_coe"], _cfg["smoothing_wsize"]
                )
                start_index_list[0].append(start_index)

            if _cfg["global_start"]:
                start_index = np.median(np.array(start_index_list[0])).astype(np.int32)
                start_index_list[0] = [
                    start_index,
                    start_index,
                    start_index,
                    start_index,
                    start_index,
                ]

        if len(start_index_list) > 1:
            signal_list = self._crop_shift_signal(signal_list, start_index_list)

        # shift gt

        img_gt_loc_list = []
        for gt_loc in gt_loc_list:
            if not isinstance(gt_loc, np.ndarray):
                gt_loc = np.array(gt_loc)
            img_gt_loc = (
                (gt_loc - self.marker_center) * self.marker_scaling
                + self.canvas_center
                + self.marker_offset
            )
            img_gt_loc_list.append(img_gt_loc)

        window_size = _cfg["window_size"]
        running_window_size = _cfg["running_window_size"]
        step = _cfg["step"]

        _cfg["window_size"] = 1
        img_queue = np.zeros((running_window_size, *self.canvas_size))

        _start_index_list = start_index_list[0]

        save_path_list = []
        max_score = 0
        max_pred = None
        max_frame_ind = 0

        for frame_ind in range(window_size):
            tmp_start_index_list = [frame_ind + _s for _s in _start_index_list]
            pred_xy_real_cor, single_img = self.pred(
                signal_list, tmp_start_index_list, _cfg
            )

            img_queue[frame_ind % running_window_size] = single_img

            avg_img = np.sum(img_queue, axis=0)
            pred_yx = np.array(np.unravel_index(np.argmax(avg_img), avg_img.shape))
            pred_xy = np.array([pred_yx[1], pred_yx[0]])
            pred_score = np.max(avg_img)

            is_max = False
            if pred_score > max_score:
                max_score = pred_score
                max_pred = pred_xy
                max_frame_ind = np.median(np.array(tmp_start_index_list))
                is_max = True

            if frame_ind % step == (step - 1):
                fig = plt.figure(tight_layout=True, figsize=(9, 6))
                gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
                ax0 = fig.add_subplot(gs[0, :])
                ax1 = fig.add_subplot(gs[1, :])

                plot_x = list(range(len(signal_list[0][0])))
                ax0.plot(plot_x, signal_list[0][0], label="x-tilts")
                ax0.plot(plot_x, signal_list[0][1], label="y-tilts")
                ax0.axvline(
                    x=frame_ind + _start_index_list[0], color="cyan", linestyle="dotted"
                )
                ax0.legend()

                for i, mxy in enumerate(self.marker_locations):
                    label = "marker-{}".format(i)
                    m_color = "white"
                    ax1.scatter(mxy[0], mxy[1], color=m_color, label=label, marker="s")
                    ax1.text(mxy[0], mxy[1], s=label, color=m_color)

                for img_gt_loc in img_gt_loc_list:
                    ax1.scatter(
                        img_gt_loc[0],
                        img_gt_loc[1],
                        color="red",
                        label="knock",
                        marker="P",
                    )
                    ax1.text(img_gt_loc[0], img_gt_loc[1], color="red", s="knock")

                if self.corner_locations is not None:
                    rec_color = np.max(avg_img) * 0.8
                    cv2.rectangle(
                        avg_img,
                        self.corner_locations[0].astype(np.int32),
                        self.corner_locations[1].astype(np.int32),
                        color=rec_color,
                        thickness=3,
                    )

                ax1.scatter(pred_xy[0], pred_xy[1], color="orange", marker="*")
                ax1.text(pred_xy[0], pred_xy[1], color="orange", s="pred")
                if pred_xy_list is not None:
                    for pred_xy_ in pred_xy_list:
                        ax1.scatter(pred_xy_[0], pred_xy_[1], color="cyan", marker="*")
                        ax1.text(pred_xy_[0], pred_xy_[1], color="cyan", s="pred-max")
                elif max_pred is not None:
                    ax1.scatter(max_pred[0], max_pred[1], color="cyan", marker="*")
                    ax1.text(max_pred[0], max_pred[1], color="cyan", s="pred-max")
                if color_map_range is None:
                    ax1.imshow(avg_img)
                else:
                    ax1.imshow(
                        avg_img, vmin=color_map_range[0], vmax=color_map_range[1]
                    )
                # ax = plt.gca()
                ax1.invert_yaxis()

                tmp_save_path = os.path.join(tmp_save_dir, "{}.png".format(frame_ind))
                print(tmp_save_path)
                save_path_list.append(tmp_save_path)
                plt.savefig(tmp_save_path)
                if is_max:
                    # max_file_path = os.path.join(tmp_save_dir, "max_score.png")
                    os.system("cp {} {}".format(tmp_save_path, max_img_save_path))

        import imageio

        writer = imageio.get_writer(save_path, fps=30)

        for impath in save_path_list:
            writer.append_data(imageio.imread(impath))
        writer.close()
        return save_path_list, max_pred

    def pred_and_plot_polish(
        self,
        signal_list,
        start_index_list=None,
        theta_cfg=None,
        gt_loc=None,
        save_path=None,
        color_map_range=None,
        auto_optimize=True,
        debug=False,
    ):
        """
        Similar to pred_and_plot, but with polish ploting confirguration
        """
        # shift gt
        if not isinstance(gt_loc, np.ndarray):
            gt_loc = np.array(gt_loc)

        img_gt_loc = (
            (gt_loc - self.marker_center) * self.marker_scaling
            + self.canvas_center
            + self.marker_offset
        )

        pred_xy_real_cor, ret_img = self.pred(
            signal_list,
            start_index_list,
            theta_cfg,
            auto_optimize=auto_optimize,
            debug=debug,
        )
        pred_yx = np.unravel_index(np.argmax(ret_img), ret_img.shape)
        pred_xy = np.array([pred_yx[1], pred_yx[0]])

        plt.figure()

        for i, mxy in enumerate(self.marker_locations):
            label = "marker-{}".format(i)
            m_color = "white"
            plt.scatter(mxy[0], mxy[1], color=m_color, label=label, marker="s")
            plt.text(mxy[0], mxy[1], s=label, color=m_color)

        plt.scatter(
            img_gt_loc[0], img_gt_loc[1], color="red", label="knock", marker="P"
        )
        plt.text(img_gt_loc[0], img_gt_loc[1], color="red", s="knock")

        if self.corner_locations is not None:
            rec_color = np.max(ret_img) * 0.8
            cv2.rectangle(
                ret_img,
                self.corner_locations[0].astype(np.int32),
                self.corner_locations[1].astype(np.int32),
                color=rec_color,
                thickness=3,
            )

        plt.scatter(pred_xy[0], pred_xy[1], color="orange", marker="*")
        plt.text(pred_xy[0], pred_xy[1], color="orange", s="pred")
        if color_map_range is None:
            plt.imshow(ret_img)
        else:
            plt.imshow(ret_img, vmin=color_map_range[0], vmax=color_map_range[1])
        ax = plt.gca()
        ax.invert_yaxis()

        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)

        return pred_xy_real_cor, ret_img, self.marker_locations, pred_xy, img_gt_loc


def exp_weighting_func(single_frame, params=None):
    tmp_img = np.exp(single_frame - 1.0) - 1.0
    return tmp_img


def exp_relu_weighting_func(single_frame, params=None):
    tmp_img = np.exp(single_frame - 1.0) - 1.0
    # add relu
    tmp_img[tmp_img < 0] = 0
    return tmp_img


def poly_weighting_func(single_frame, params=[2]):
    tmp_img = single_frame ** params[0]

    tmp_img = single_frame - 1.0

    return tmp_img


def poly_relu_weighting_func(single_frame, params=[2]):
    tmp_img = single_frame ** params[0]

    tmp_img = single_frame - 1.0

    # add relu
    tmp_img[tmp_img < 0] = 0
    return tmp_img
