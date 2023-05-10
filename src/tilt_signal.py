"""
Description: code for basic signal processing of tilt signal 
    A tilt signal is a 2D signal, which is a 2D vector of time, denoting tilt in x and y directions. (shape [2, num_of_frames])
Author: Tianyuan Zhang
"""

from typing import List, Tuple, Dict, Callable
import numpy as np
from copy import deepcopy
import pickle

try:
    from .angle_utils import Name2AngleFunc
except:
    from angle_utils import Name2AngleFunc


class Tilt2D(object):
    def __init__(
        self, tilts, mean=None, std=None, len_first_clip=80, cropped=False
    ) -> None:
        """
        Args:
            tilts (np.ndarray) of shape [2, num_of_frames]
                denotes tilts in x and y directions.

            mean: float.  represent pre-computed mean value
            std: float. represent pre-computed std value.
                If no pre-computed! It will be computed using the first clip of the signal

            cropped: bool. default False
                If True, first frame of the signal denote the the arrival time of the signal is the.

        """

        # [N,1]
        self.tilt_x = tilts[0]
        self.tilt_y = tilts[1]

        self.mean = mean
        self.std = std
        self.cropped = cropped

        # first clip of the signal to compute mean and std
        self.len_first_clip = len_first_clip

    def clone(
        self,
    ):
        signal_ = Tilt2D(
            np.stack([deepcopy(self.tilt_x), deepcopy(self.tilt_y)], axis=0),
            len_first_clip=self.len_first_clip,
        )

        return signal_

    def pad(
        self,
    ):
        """
        pad the beggning of the signal with 30 frames of the first clip
        """
        new_x = np.concatenate(
            [
                deepcopy(self.tilt_x[:30]),
                deepcopy(self.tilt_x[:30]),
                deepcopy(self.tilt_x[:30]),
                self.tilt_x,
            ]
        )
        new_y = np.concatenate(
            [
                deepcopy(self.tilt_y[:30]),
                deepcopy(self.tilt_y[:30]),
                deepcopy(self.tilt_y[:30]),
                self.tilt_y,
            ]
        )

        ret_signal = Tilt2D([new_x, new_y])

        return ret_signal

    def __getitem__(self, ind):
        if ind == 0:
            return self.tilt_x
        elif ind == 1:
            return self.tilt_y

        raise NotImplementedError

    def __add__(self, other):
        """
        Args: other:  np.array of shape [2, N]. or Tilt2D
        """

        tilt_x = self.tilt_x + other[0]
        tilt_y = self.tilt_y + other[1]

        ret_signal = Tilt2D(np.stack([tilt_x, tilt_y], axis=0))

        return ret_signal

    def get_smoothed_signal(self, window_size=5) -> np.ndarray:
        """
        Using guassian kernel to smooth the signal
        Args:
            window_size: int. the size of the guassian kernel equals to window_size * 2 + 1
        Outputs:
            smoothed_titls Tilt2D.
                A new Tilt2D object denotes the gaussian smoothed signal
        """

        kernel1 = np.arange(1, window_size // 2 + 1)

        kernel2 = np.arange(1, window_size // 2 + 1)[::-1]

        kernel = np.concatenate([kernel1, kernel2])

        kernel = kernel / np.sum(kernel)

        eff_kernel_size = len(kernel)
        crop_size = (eff_kernel_size - 1) // 2
        # print(kernel, window_size)
        ret_x = np.convolve(kernel, self.tilt_x)[crop_size:]
        ret_y = np.convolve(kernel, self.tilt_y)[crop_size:]

        ret_signal = np.stack([ret_x, ret_y], axis=0)
        ret_signal = Tilt2D(ret_signal)
        return ret_signal

    def get_gradient(self, smoothing_wsize=-1):
        """
        Get the gradient of the signal, which is the difference between two consecutive frames
        Args:
            smoothing_wsize: int.
                perform gaussian smoothing with window size of smoothing_wsize * 2 + 1 before computing the gradient
                if -1, no smoothing
        Outputs:
            signal_: Tilt2D.
                A new Tilt2D object denotes the gradient of the signal
        """
        if smoothing_wsize > 0:
            signal_ = self.get_smoothed_signal(smoothing_wsize)
        else:
            signal_ = self.clone()

        signal_.tilt_x = signal_.tilt_x[1:] - signal_.tilt_x[:-1]
        signal_.tilt_y = signal_.tilt_y[1:] - signal_.tilt_y[:-1]

        return signal_

    def forward_conv(self, forward_window_size=40, if_abs=True):
        kernel_ = np.arange(1, forward_window_size + 1)[::-1]

        kernel_ = kernel_ / np.sum(kernel_)

        tilt_x = self.tilt_x
        tilt_y = self.tilt_y
        if if_abs:
            tilt_x = np.abs(self.tilt_x)
            tilt_y = np.abs(self.tilt_y)

        ret_x = np.convolve(kernel_, tilt_x)[forward_window_size - 1 :]
        ret_y = np.convolve(kernel_, tilt_y)[forward_window_size - 1 :]

        ret_signal = np.stack([ret_x, ret_y], axis=0)
        ret_signal = Tilt2D(ret_signal, len_first_clip=self.len_first_clip)

        return ret_signal

    def edge_conv(self, forward_window_size=40, if_abs=True):

        kernel_ = np.arange(1, forward_window_size + 1)[::-1]

        kernel_ = kernel_ / np.sum(kernel_)

        kernel_2 = -1.0 * np.arange(1, forward_window_size + 1)[::-1]

        kernel_2 = kernel_2 / np.sum(np.abs(kernel_2))

        kernel = np.concatenate([kernel_, kernel_2])

        tilt_x = self.tilt_x
        tilt_y = self.tilt_y
        if if_abs:
            tilt_x = np.abs(self.tilt_x)
            tilt_y = np.abs(self.tilt_y)

        ret_x = np.convolve(kernel, tilt_x, mode="valid")
        ret_y = np.convolve(kernel, tilt_y, mode="valid")

        ret_signal = np.stack([ret_x, ret_y], axis=0)
        ret_signal = Tilt2D(ret_signal, len_first_clip=self.len_first_clip)

        return ret_signal

    def compute_mean_std(self, motion_1d):
        """
        Args:
            motion_1d: np.array of shape [N, ]
        """

        motion1d_window = motion_1d[: self.len_first_clip]

        ret_mean = np.mean(motion1d_window)
        ret_std = np.std(motion1d_window - ret_mean)

        return ret_mean, ret_std

    def get_roi_start(self, y, std_coe=4.0, mean=None, std=None):
        """
        Using threshold to get the arrival time of the transient surface waves
            Described in Eq (13) of the main paper
        Args:
            y (np.ndarray) of shape [N, ]
            std_coe: float, the threshold is std_coe * std of the signal
        Outs:
            start_index
        """
        if mean is None or std is None:
            mean_value, std_value = self.compute_mean_std(y)
            thres = std_value * std_coe
        else:
            mean_value = mean
            thres = std * std_coe

        # print(thres)
        thres = 0.6 * thres + 1.0 * 0.4

        y_larger = (np.abs(y - mean_value) > thres).astype(np.float32)

        # non zero return a tuple
        # print(y_larger, thres)
        if len(np.nonzero(y_larger)[0]) > 0:
            start_index = np.nonzero(y_larger)[0][0]
        else:
            start_index = len(y) - 1
        # print(y_larger.shape, start_index.shape, np.nonzero(y_larger))

        return start_index

    def get_start_index(
        self,
        std_coe=15.0,
        smoothing_wsize=-1,
        forward_window_size=40,
        mag_only=True,
    ):
        """
        A warpper function to get the arrival time of the transient surface waves
        This function groups multiple steps:
            1. smoothing  (optional), calling self.get_smoothed_signal
            2. high pass filtering  (optional) self.forward_conv
            3. thresholding to get the arrival time self.get_roi_start
        Args:
            std_coe: for threshold
            smoothing_wsize: size of the smoothing window size.
                if smoothing_wsize < 0, means no smoothing
            forward_window_size: int
                The window size high pass filter has is 2 * forward_window_size
            mag_only: bool
                If True, only use the magnitude of the signal for thresholding
        """

        if self.cropped:
            print("Signal been cropped!")
            return 0

        signal = Tilt2D([self.tilt_x, self.tilt_y])

        if smoothing_wsize > 0:
            signal = signal.get_smoothed_signal(smoothing_wsize)
            raw_signal = signal

        if mag_only:
            mag_signal = np.sqrt(signal.tilt_x**2 + signal.tilt_y**2)
            signal = Tilt2D([mag_signal, deepcopy(mag_signal)])

        if forward_window_size > 0:
            # change to edge conv
            signal = signal.edge_conv(forward_window_size, if_abs=True)

        if self.mean is None or self.std is None:

            start_index_x = self.get_roi_start(signal[0], std_coe)
            start_index_y = self.get_roi_start(signal[1], std_coe)
        else:
            start_index_x = self.get_roi_start(
                signal[0], std_coe, self.mean[0], self.std[0]
            )
            start_index_y = self.get_roi_start(
                signal[1], std_coe, self.mean[1], self.std[1]
            )

        error_xy = np.abs(start_index_x - start_index_y)
        if error_xy > 100000:
            print(
                "roi for tilt_x and tilt_y is not matched! roi_x: {} roi_y: {}".format(
                    start_index_x, start_index_y
                )
            )
        assert error_xy < 3000, "roi for tilt_x and tilt_y is not matched!"

        if error_xy < 10:
            start_index = (start_index_x + start_index_y) // 2
        else:
            # look at magnitude
            mag_x = np.abs(signal[0][start_index_x])
            mag_y = np.abs(signal[1][start_index_y])
            if mag_x > mag_y:
                start_index = start_index_x
            else:
                start_index = start_index_y

            # start_index = min(start_index_x, start_index_y)

            # set to y
            start_index = start_index_y

        if forward_window_size > 0:
            start_index = start_index + forward_window_size

        return start_index

    def get_roi(
        self,
        std_coe: float = 10.0,
        smoothing_wsize: int = -1,
        if_normalize: bool = False,
        start_index: int = -1,
        offset: int = 0,
        thre: float = -1.0,
        window_size: int = 100,
    ):
        """
        Get the stable ratio interval of the signal, which is used for backprojections.
        This function groups multiple steps together:
            1. determing the arrival time of the transient surface waves by calling self.get_start_index
            2. crop the signal start from the arrival time and parameter:offset,  with a window size specifed by parameter:window_size
        Args:
            std_coe: for threshold
            smoothing_wsize: size of the smoothing window size.
                if smoothing_wsize < 0, means no smoothing
            start_index:
                -1: unspecified. if unspecifed, will compute by
                call self.get_start_index
            offset: add offset to the start index
            thre: threshould very small motions.
                -1: does not so threshouling
            window_size: window_size of the RoI for averaing the theta
        """

        # first filter out signals!
        signal = Tilt2D([self.tilt_x, self.tilt_y])

        # smooth the signal
        if smoothing_wsize > 0:
            signal = signal.get_smoothed_signal(smoothing_wsize)

        tilt_x, tilt_y = signal.tilt_x, signal.tilt_y

        if start_index is None or start_index < 0:
            start_index = self.get_start_index(std_coe=std_coe, smoothing_wsize=-1)
        if if_normalize:
            # normalize
            if self.mean is None or self.std is None:
                mean_x, _ = self.compute_mean_std(tilt_x)
                mean_y, _ = self.compute_mean_std(tilt_y)
            else:
                mean_x, mean_y = self.mean[0], self.mean[1]

            tilt_x = tilt_x - mean_x
            tilt_y = tilt_y - mean_y

        start_index = start_index + offset
        x_roi = tilt_x[start_index : start_index + window_size]
        y_roi = tilt_y[start_index : start_index + window_size]

        # do thresholing
        if thre > 0:
            filter_mask = np.abs(x_roi) > thre

            x_roi = x_roi[filter_mask]
            y_roi = y_roi[filter_mask]

        return x_roi, y_roi

    def get_roi_renorm(
        self,
        std_coe: float = 10.0,
        smoothing_wsize: int = -1,
        forward_window_size: int = 40,
        if_normalize: bool = False,
        start_index: int = -1,
        offset: int = 0,
        thre: float = -1.0,
        window_size: int = 100,
        pre_size: int = -1,
        safe_interval: int = 20,
        **kwargs,
    ):
        """
        A warpper function for self.get_roi,
        which will add one extra step to renormalize the signal
        Args:
            std_coe: for threshold
            smoothing_wsize: size of the smoothing window size.
                if smoothing_wsize < 0, means no smoothing
            start_index:
                -1: unspecified. if unspecifed, will compute by
                call self.get_start_index
            offset: add offset to the start index
            thre: threshould very small motions.
                -1: does not so threshouling
            window_size: window_size of the RoI for averaing the theta
        """

        # first filter out signals!
        signal = Tilt2D([self.tilt_x, self.tilt_y])

        # smooth the signal
        if smoothing_wsize > 0:
            signal = signal.get_smoothed_signal(smoothing_wsize)

        tilt_x, tilt_y = signal.tilt_x, signal.tilt_y

        if start_index is None or start_index < 0:
            start_index = self.get_start_index(
                std_coe=std_coe,
                smoothing_wsize=smoothing_wsize,
                forward_window_size=forward_window_size,
            )
        if if_normalize:
            # normalize
            if self.mean is None or self.std is None:
                mean_x, _ = self.compute_mean_std(tilt_x)
                mean_y, _ = self.compute_mean_std(tilt_y)
            else:
                mean_x, mean_y = self.mean[0], self.mean[1]

            tilt_x = tilt_x - mean_x
            tilt_y = tilt_y - mean_y

        if pre_size != -1 and if_normalize:
            pre_x_roi = tilt_x[
                start_index - safe_interval - pre_size : start_index - safe_interval
            ]
            pre_y_roi = tilt_y[
                start_index - safe_interval - pre_size : start_index - safe_interval
            ]

            pre_mean_x = pre_x_roi.mean()
            pre_mean_y = pre_y_roi.mean()

            tilt_x = tilt_x - pre_mean_x
            tilt_y = tilt_y - pre_mean_y

        start_index = start_index + offset
        x_roi = tilt_x[start_index : start_index + window_size]
        y_roi = tilt_y[start_index : start_index + window_size]

        # do thresholing
        if thre > 0:
            filter_mask = np.abs(x_roi) > thre

            x_roi = x_roi[filter_mask]
            y_roi = y_roi[filter_mask]

        return x_roi, y_roi

    def get_theta(
        self,
        std_coe: float = 10.0,
        smoothing_wsize: int = -1,
        forward_window_size: int = 40,
        if_normalize: bool = False,
        start_index: int = -1,
        offset: int = 0,
        thre: float = -1.0,
        window_size: int = 100,
        angle_config=dict(
            name="median",
        ),
        pre_size: int = 30,
        safe_interval: int = 20,
        search_range=None,
    ):
        """
        This function performs the following steps:
        1. get the stable ratio interval of the tilt_x and tilt_y
        2. using the stable ratio interval to compute a robust estimate of the ratio between y_tilt/x_tilt,
            basically a robust estimate of \theta_y / \theta_x in Eq (7) of the paper
        Args:
            std_coe: for threshold
            smoothing_wsize: size of the smoothing window size.
                if smoothing_wsize < 0, means no smoothing
            start_index:
                -1: unspecified. if unspecifed, will compute by
                call self.get_start_index
            offset: add offset to the start index
            thre: threshould very small motions.
                -1: does not so threshouling
            window_size: window_size of the RoI for averaing the theta
            search_range: If None, no search.
                if [start, end, step_size], then search!
        """

        if search_range is not None and len(search_range) == 3:

            return self.get_theta_search(
                std_coe,
                smoothing_wsize,
                forward_window_size,
                if_normalize,
                start_index,
                offset,
                thre,
                window_size,
                angle_config,
                pre_size,
                safe_interval,
                search_range,
            )

        x_roi, y_roi = self.get_roi_renorm(
            std_coe,
            smoothing_wsize,
            forward_window_size,
            if_normalize,
            start_index,
            offset,
            thre,
            window_size,
            pre_size,
            safe_interval,
        )

        # signal filterd!
        func_name = angle_config["name"]
        param_config = {}
        for name, value in angle_config.items():
            if name == "name":
                continue
            param_config[name] = value

        # this one can don't return theta
        theta, error = Name2AngleFunc[func_name](x_roi, y_roi, **param_config)
        return theta, error

    def get_cropped_signal(
        self, start_index, window_size=100, offset=0, normalize=False
    ):

        tilt_x, tilt_y = self.tilt_x, self.tilt_y

        start_index = start_index + offset

        x_roi = tilt_x[start_index : start_index + window_size]
        y_roi = tilt_y[start_index : start_index + window_size]

        if normalize:
            if self.mean is None or self.std is None:
                mean_x, _ = self.compute_mean_std(tilt_x)
                mean_y, _ = self.compute_mean_std(tilt_y)
            else:
                mean_x, mean_y = self.mean[0], self.mean[1]

            x_roi = x_roi - mean_x
            y_roi = y_roi - mean_y

            x_max = np.max(np.abs(x_roi))
            y_max = np.max(np.abs(y_roi))

            norm_max = max(x_max, y_max)

            x_roi = x_roi / norm_max
            y_roi = y_roi / norm_max

        ret_signal = Tilt2D(
            np.stack([x_roi, y_roi], axis=0),
            cropped=True,
        )

        return ret_signal

    def get_theta_search(
        self,
        std_coe: float = 10.0,
        smoothing_wsize: int = -1,
        forward_window_size: int = 40,
        if_normalize: bool = False,
        start_index: int = -1,
        offset: int = 0,
        thre: float = -1.0,
        window_size: int = 100,
        angle_config=dict(name="median"),
        pre_size: int = 30,
        safe_interval: int = 20,
        search_range=[-10, 50, 2],
    ):
        """
        This function is an advanced version of self.get_theta.
        It searches the best hyper-parameters, like window_size,
        for the best robust estimate of the angle.
        Args:
            window_size: default as int.  if window_size as a list, e.g. [10, 20]
                then performing searching!
        """

        search_space = list(range(*search_range))
        search_min = min(search_space)
        search_max = max(search_space)

        if isinstance(window_size, list):
            assert (
                len(window_size) == 2
            ), "format for search space of window size is wrong"
            window_size_search_space = window_size
            window_size = window_size[1]
        else:
            window_size_search_space = None
        # first filter out signals!
        signal = Tilt2D([self.tilt_x, self.tilt_y])

        # smooth the signal
        if smoothing_wsize > 0:
            signal = signal.get_smoothed_signal(smoothing_wsize)

        tilt_x, tilt_y = signal.tilt_x, signal.tilt_y

        if start_index is None or start_index < 0:
            start_index = self.get_start_index(
                std_coe=std_coe,
                smoothing_wsize=smoothing_wsize,
                forward_window_size=forward_window_size,
            )
        if if_normalize:
            # normalize
            if self.mean is None or self.std is None:
                mean_x, _ = self.compute_mean_std(tilt_x)
                mean_y, _ = self.compute_mean_std(tilt_y)
            else:
                mean_x, mean_y = self.mean[0], self.mean[1]

            tilt_x = tilt_x - mean_x
            tilt_y = tilt_y - mean_y

        if pre_size != -1 and if_normalize:
            pre_x_roi = tilt_x[
                start_index - safe_interval - pre_size : start_index - safe_interval
            ]
            pre_y_roi = tilt_y[
                start_index - safe_interval - pre_size : start_index - safe_interval
            ]

            pre_mean_x = pre_x_roi.mean()
            pre_mean_y = pre_y_roi.mean()

            tilt_x = tilt_x - pre_mean_x
            tilt_y = tilt_y - pre_mean_y

        start_index = start_index + offset
        min_start = start_index + search_min
        max_start = start_index + search_max
        x_roi = tilt_x[min_start : max_start + window_size]
        y_roi = tilt_y[min_start : max_start + window_size]

        # do thresholing
        if thre > 0:
            filter_mask = np.abs(x_roi) > thre

            x_roi = x_roi[filter_mask]
            y_roi = y_roi[filter_mask]

        # Prepare param for angle func
        func_name = angle_config["name"]
        param_config = {}
        for name, value in angle_config.items():
            if name == "name":
                continue
            param_config[name] = value

        search_max_ind = len(x_roi) - window_size
        min_error = 10000
        best_ret = None
        best_start_index = None
        best_window_size = None
        for start_ind in range(search_max_ind + 1):

            if window_size_search_space is not None:
                for window_size in range(
                    window_size_search_space[0], window_size_search_space[1] + 1, 2
                ):

                    tmp_x_roi = x_roi[start_ind : start_ind + window_size]
                    tmp_y_roi = y_roi[start_ind : start_ind + window_size]

                    mean_y = np.mean(np.abs(tmp_y_roi))

                    theta, error = Name2AngleFunc[func_name](
                        tmp_x_roi, tmp_y_roi, **param_config
                    )
                    error = error / mean_y

                    if error < min_error:
                        min_error = error
                        best_ret = theta
                        best_start_index = start_ind + min_start
                        best_window_size = window_size
            else:
                tmp_x_roi = x_roi[start_ind : start_ind + window_size]
                tmp_y_roi = y_roi[start_ind : start_ind + window_size]

                mean_y = np.mean(np.abs(tmp_y_roi))

                theta, error = Name2AngleFunc[func_name](
                    tmp_x_roi, tmp_y_roi, **param_config
                )
                error = error / mean_y

                if error < min_error:
                    min_error = error
                    best_ret = theta
                    best_start_index = start_ind + min_start
                    best_window_size = window_size

        # print(best_window_size)
        return best_ret, (min_error, best_window_size, best_start_index)

    def shift_signal(
        self,
        std_coe: float = 8.0,
        smoothing_wsize: int = -1,
        forward_window_size: int = -1,
        start_index: int = None,
        pre_size: int = 30,
        safe_interval: int = 50,
        just_start: bool = False,
        # **kwargs,
    ):
        """
        Substract the mean of the signal and return the shifted signal

        Args:
            std_coe: for threshold
            smoothing_wsize: size of the smoothing window size.
                if smoothing_wsize < 0, means no smoothing
            start_index:
                -1: unspecified. if unspecifed, will compute by
                call self.get_start_index
            offset: add offset to the start index
            thre: threshould very small motions.
                -1: does not so threshouling
            window_size: window_size of the RoI for averaing the theta
        """

        # first filter out signals!
        signal = Tilt2D([self.tilt_x, self.tilt_y])
        prev_mean_x = self.tilt_x[:50].mean()
        prev_mean_y = self.tilt_y[:50].mean()

        signal.tilt_x = signal[0] - prev_mean_x
        signal.tilt_y = signal[1] - prev_mean_y

        if just_start:
            prev_mean_x = signal.tilt_x[:pre_size].mean()
            prev_mean_y = signal.tilt_y[:pre_size].mean()

            signal.tilt_x = signal[0] - prev_mean_x
            signal.tilt_y = signal[1] - prev_mean_y

            return signal

        crop_signal = Tilt2D([self.tilt_x[20:], self.tilt_y[20:]])
        # smooth the signal
        if smoothing_wsize > 0:

            signal = signal.get_smoothed_signal(smoothing_wsize)

        if start_index is None:
            start_index = (
                crop_signal.get_start_index(
                    std_coe, smoothing_wsize, forward_window_size
                )
                + 20
            )

        if start_index < (pre_size + safe_interval):
            pre_size = start_index - safe_interval

        prev_start_seg_x = signal[0][
            start_index - pre_size - safe_interval : start_index - safe_interval
        ]
        prev_mean_x = np.mean(prev_start_seg_x)

        prev_start_seg_y = signal[1][
            start_index - pre_size - safe_interval : start_index - safe_interval
        ]
        prev_mean_y = np.mean(prev_start_seg_y)

        signal.tilt_x = signal[0] - prev_mean_x
        signal.tilt_y = signal[1] - prev_mean_y

        return signal

    def get_stability(self, window_size=50):
        """
        Compute the stability of the ratio tilts_y / tilts_x for each frame.
        The stability metric is computed within a window of size window_size.
        This stability metric can be used to search for the stable time interval.

        However, by default, we don't perform searching. But we still visualize the stability metric
        for better understanding the data.
        """

        half_window_size = window_size // 2

        start_ind = (window_size + 1) // 2

        end_ind = len(self.tilt_y) - start_ind

        ret_list = []

        y_div_x = self.tilt_y / self.tilt_x

        norm_signal = np.sqrt(self.tilt_x**2 + self.tilt_y**2 + 1e-3)

        normalized_x = self.tilt_x / norm_signal
        normalized_y = self.tilt_y / norm_signal

        for i in range(start_ind, end_ind):
            # cropped_ratio = y_div_x[i - half_window_size : i + half_window_size]
            median_ratio = y_div_x[i]

            cropped_norm_x = normalized_x[i - half_window_size : i + half_window_size]
            cropped_norm_y = normalized_y[i - half_window_size : i + half_window_size]
            # cropped_norm_x = normalized_x[i]
            # cropped_norm_y = normalized_y[i]

            # median_ratio = np.median(cropped_ratio)

            reconstruct_mse = np.mean(
                (
                    (cropped_norm_x * median_ratio - cropped_norm_y)
                    / np.sqrt(1 + median_ratio**2)
                )
                ** 2
            )

            ret_list.append(reconstruct_mse)

        return np.array(ret_list)


def linear_angle_mapping(scaling, ratio):
    """
    Args:
        scaling: float  ratio / scaling => tan_theta
        ratio:   float
    """

    return ratio / scaling


def rotation_angle_mapping(params, ratio):
    """
    Args:
        params: [scaling, alpha]
        ratio:  float
    """
    theta_p_alpha = np.arctan(ratio / params[0])

    ret_tan_theta = np.tan(theta_p_alpha - params[1])

    return ret_tan_theta


class AngleMapping(object):
    """
    A class for acounting the camera caliberation and material anisotropy
    """

    def __init__(self, cfg=dict(name="linear", params=[1.0]), num_marker=10) -> None:
        """
        Args:
            cfg['params']: List()
        """

        if len(cfg["params"]) == 1:
            # copy params into params list. All markers has the same angle mapping
            self.params_list = cfg["params"] * num_marker
        else:
            self.params_list = cfg["params"]

        self.name = cfg["name"]

        self.name2func = {
            "linear": linear_angle_mapping,
            "rotation": rotation_angle_mapping,
        }

    def ratio2directions(self, ratio_list, marker_ind_list=None):
        ret_list = []
        if marker_ind_list is None:
            for i in range(len(ratio_list)):
                ret_list.append(
                    self.name2func[self.name](self.params_list[i], ratio_list[i])
                )

            return ret_list

        assert len(marker_ind_list) >= len(
            ratio_list
        ), "marker ind not specified in correct length"

        for i in range(len(marker_ind_list)):
            ret_list.append(
                self.name2func[self.name](
                    self.params_list[marker_ind_list[i]], ratio_list[i]
                )
            )

        return ret_list

    def __call__(self, ratio_list, marker_ind_list=None):
        return self.ratio2directions(ratio_list, marker_ind_list)

    def save(self, file_path):

        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    def load(self, file_path):

        with open(file_path, "rb") as f:
            read_obj = pickle.load(f)

        return read_obj
