import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Callable
import numpy as np
import copy

try:
    from .tilt_signal import Tilt2D
except:
    from tilt_signal import Tilt2D


# collections of visually pleasing colors
class ColorMap(object):
    def __init__(self) -> None:

        self.color_mapping = [
            np.array([4, 157, 217]) / 255.0,
            np.array([191, 4, 54]) / 255.0,
            np.array([140, 140, 136]) / 255.0,
            np.array([0, 0, 0]) / 255.0,
            np.array([224, 133, 250]) / 255.0,
            np.array([26, 85, 153]) / 255.0,  # 5. Capri blue
            np.array([77, 115, 67]) / 255.0,  # 6.
            np.array([76, 0, 9]) / 255.0,  # 7. Bordeaux red
            np.array([0, 149, 182]) / 255.0,  # 8. Bondi blue
            np.array([32, 64, 40]) / 255.0,
            np.array([129, 216, 208]) / 255.0,  # 10. Tiffany blue
            np.array([128, 0, 32]) / 255.0,  # 11. Burgundy red
            np.array([143, 75, 40]) / 225.0,  # 12. Mummy brown
            np.array([0, 49, 83]) / 255.0,  # 13. Berlin blue
            np.array([230, 159, 0]) / 255.0,  # 14. Normal orange
            np.array([250, 127, 111]) / 255.0,  # 15. orange pink. from hzw
            np.array([255, 190, 122]) / 255.0,  # 16. orange yellow. from hzw
            np.array([(130, 176, 210)]) / 255.0,  # 17. drak sky blue. from hzw
            np.array([231, 218, 210]) / 255.0,  # 18 warm gray. from hzw
            np.array([116, 195, 101])/255.0,  # 19 light mint green
        ]

    def __len__(
        self,
    ):
        return len(self.color_mapping)

    def __getitem__(self, ind):

        ind = ind % len(self.color_mapping)

        return self.color_mapping[ind]


color_mapping = ColorMap()


def plot_signals(
    signal_list: List[Tilt2D],
    start_index_list: List[int] = None,
    label_list=None,
    window_size=30,
    if_double_plot=True,
):

    num_signal = len(signal_list)

    if if_double_plot:
        fig, axs = plt.subplots(2)
        fig.suptitle("Vertically stacked X-Y titls")

        for ind_signal in range(num_signal):
            x = list(range(signal_list[ind_signal][0].shape[-1]))
            if label_list is not None:
                label = label_list[ind_signal]
            else:
                label = "marker-{}".format(ind_signal)
            for fig_ind in range(2):

                axs[fig_ind].plot(
                    x,
                    signal_list[ind_signal][fig_ind],
                    color=color_mapping[ind_signal],
                    label=label,
                )
        plt.legend()

        plt.show()

    # plot start time
    if start_index_list is not None:
        assert num_signal == len(start_index_list), "start_index_list has wrong length"

        fig, axs = plt.subplots(2)
        fig.suptitle("Vertically stacked X-Y titls -- With ROI indicator")
        for ind_signal in range(num_signal):
            s = start_index_list[ind_signal]
            e = s + window_size
            x = list(range(signal_list[ind_signal][0].shape[-1]))
            print("=> start frame for marker-{} is: {}".format(ind_signal, s))
            if label_list is not None:
                label = label_list[ind_signal]
            else:
                label = "marker-{}".format(ind_signal)
            for fig_ind in range(2):
                axs[fig_ind].plot(
                    x,
                    signal_list[ind_signal][fig_ind],
                    color=color_mapping[ind_signal],
                    label=label,
                )
                axs[fig_ind].axvline(s, color=color_mapping[ind_signal], linestyle="--")
                axs[fig_ind].axvline(e, color=color_mapping[ind_signal], linestyle="--")

        plt.legend()

        plt.show()


def plot_signals_roi(
    signal_list: List[Tilt2D],
    start_index_list: List[int],
    label_list=None,
    window_size=50,
):

    num_signal = len(signal_list)

    assert num_signal == len(start_index_list), "start_index_list has wrong length"

    tmp_nparray = np.array(start_index_list).astype(np.int32)
    start = np.min(tmp_nparray) - 80
    start = max(0, start)
    end = np.max(tmp_nparray) + window_size + 10

    fig, axs = plt.subplots(2)
    fig.suptitle("Vertically stacked X-Y titls - Cropped ROI")

    for ind_signal in range(num_signal):
        # TODO, add start index
        x = list(range(signal_list[ind_signal][0].shape[-1]))[start:end]
        s = start_index_list[ind_signal]
        if label_list is not None:
            label = label_list[ind_signal]
        else:
            label = "m-{}".format(ind_signal)

        for fig_ind in range(2):
            axs[fig_ind].plot(
                x,
                signal_list[ind_signal][fig_ind][start:end],
                color=color_mapping[ind_signal],
                label=label,
            )
            axs[fig_ind].axvline(s, color=color_mapping[ind_signal], linestyle="--")
    plt.legend()

    plt.show()


def plot_signals_smoothing_pyramid(
    signal_list: List[Tilt2D],
    window_size_list: List[int] = [5, 10, 20, 40, 80],
):
    num_signal = len(signal_list)

    for smooth_size in window_size_list:
        fig, axs = plt.subplots(2)
        fig.suptitle(
            "Vertically stacked X-Y titls. Smoothing with kernel size --- {}".format(
                2 * smooth_size + 1
            )
        )

        for ind_signal in range(num_signal):
            signal_ = signal_list[ind_signal].get_smoothed_signal(smooth_size)
            x = list(range(signal_[0].shape[-1]))
            for fig_ind in range(2):
                axs[fig_ind].plot(
                    x,
                    signal_[fig_ind],
                    color=color_mapping[ind_signal],
                    label="marker-{}".format(ind_signal),
                )
        plt.legend()
        plt.show()


# plot markers


def plot_points(
    marker_locations: List[List[int]],
    arrow_list: List[List[float]] = None,
    source_location=None,
    pred_location=None,
    label_list=None,
    save_path=None,
):
    """
    Args:

    """

    plt.figure()
    plt.title("2D points plane")

    plt.xlabel("X axis")
    plt.ylabel("Y axis")

    for i, mxy in enumerate(marker_locations):
        if label_list is not None:
            label = label_list[i]
        else:
            label = "marker-{}".format(i)
        plt.scatter([mxy[0]], mxy[1], color=color_mapping[i], label=label)

        # add text for markers
        plt.text(mxy[0], mxy[1], s=label, color=color_mapping[i])

        if arrow_list is not None:
            arrowxy = arrow_list[i]
            plt.arrow(mxy[0], mxy[1], arrowxy[0], arrowxy[1], color=color_mapping[i])

    if source_location is not None:
        plt.scatter([source_location[0]], [source_location[1]], label="GT", color="red")

    if pred_location is not None:
        if isinstance(pred_location, list):
            pred_x = [_[0] for _ in pred_location]
            pred_y = [_[1] for _ in pred_location]
            plt.scatter(
                pred_x, pred_y, label="Pred", color="blue", alpha=0.5, marker="*"
            )
        else:
            plt.scatter(
                [pred_location[0]],
                [pred_location[1]],
                label="Pred",
                color="blue",
                marker="*",
            )

    # plt.legend()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def plot_points_batched(
    marker_locations: List[List[int]],
    arrow_lists=None,
    source_locations=None,
    pred_locations=None,
    label_list=None,
):
    """
    Args:

    source_locations: list of gt locations. shape of [N, 2]
    pred_locations: list of pred locations. shape of [N, 2]
    """

    plt.figure()
    plt.title("2D points plane")

    plt.xlabel("X axis")
    plt.ylabel("Y axis")

    for i, mxy in enumerate(marker_locations):
        if label_list is not None:
            label = label_list[i]
        else:
            label = "marker-{}".format(i)
        plt.scatter([mxy[0]], mxy[1], color="blue", label=label)

        # add text for markers
        plt.text(mxy[0], mxy[1], s=label, color="blue")

    if arrow_lists is not None:
        for i in range(len(arrow_lists)):
            mxy = marker_locations[i]
            arrowxy = arrow_lists[i]
            plt.arrow(mxy[0], mxy[1], arrowxy[0], arrowxy[1], color=color_mapping[i])

    if source_locations is not None:
        for i in range(len(source_locations)):
            plt.scatter(
                [source_locations[i][0]],
                [source_locations[i][1]],
                label="GT",
                color=color_mapping[i],
            )

    if pred_locations is not None:
        for i in range(len(pred_locations)):
            plt.scatter(
                [pred_locations[i][0]],
                [pred_locations[i][1]],
                label="pred",
                color=color_mapping[i],
                marker="*",
            )

    # plt.legend()

    plt.show()


def plot_signal_with_correct_theta(signal_list, gt_list):

    num_signal = len(signal_list)
    for i in range(signal_list):

        signal = signal_list[i]
        # float
        gt_theta = gt_list[i]

        x_motion = signal[0]
        y_motion = signal[1]
        norm_ = np.sqrt(x_motion ** 2 + y_motion ** 2 + 1e-8)

        sine_theta = y_motion / norm_
        cosine_theta = x_motion / norm_


def plot_target_ymotion(
    signal,
    gt_theta,
    start_index=None,
    color=None,
    error_thres=10.0 / 180.0 * np.pi,
    title="Y-motion with X-motion * tan(theta)",
):
    """
    gt_theta in np.pi space.

    red region means exact tan theta

    green region represents theta with 180 difference.
        (same tan theta, but different theta)
    """

    if signal.mean is None or signal.std is None:
        mean_x, _ = signal.compute_mean_std(signal[0])
        mean_y, _ = signal.compute_mean_std(signal[1])
    else:
        mean_x, mean_y = signal.mean[0], signal.mean[1]

    pred_y = np.tan(gt_theta) * (signal[0] - mean_x) + mean_y

    theta_list = np.arctan2(signal[1] - mean_y, signal[0] - mean_x)

    accurate_regions = np.abs(theta_list - gt_theta) < error_thres
    # add invert
    accurate_regions_inverse = (
        np.abs((theta_list - np.pi) % (np.pi * 2.0) - gt_theta) < error_thres
    )

    fig, axs = plt.subplots(2)
    fig.suptitle(title)

    x_list = np.array(list(range(len(signal[0]))))
    axs[0].plot(x_list, signal[0], color=color)
    # axs[0].plot(x_list, theta_list)

    axs[1].plot(x_list, signal[1], label="y-motion", color=color)
    axs[1].plot(x_list, pred_y, color="red", label="x * tan", linestyle="--")

    if start_index is not None:
        axs[1].axvline(start_index, color="cyan", linestyle="-.")
        accurate_regions = np.logical_and(accurate_regions, x_list >= start_index)
        accurate_regions_inverse = np.logical_and(
            accurate_regions_inverse, x_list >= start_index
        )

    axs[1].fill_between(
        x_list, *axs[1].get_ylim(), where=accurate_regions, facecolor="red", alpha=0.2
    )
    axs[1].fill_between(
        x_list,
        *axs[1].get_ylim(),
        where=accurate_regions_inverse,
        facecolor="green",
        alpha=0.2
    )

    axs[0].fill_between(
        x_list, *axs[0].get_ylim(), where=accurate_regions, facecolor="red", alpha=0.2
    )
    axs[0].fill_between(
        x_list,
        *axs[0].get_ylim(),
        where=accurate_regions_inverse,
        facecolor="green",
        alpha=0.2
    )

    plt.legend()
    plt.show()


def plot_tan_theta(
    signal_list, theta_cfg, gt_tan_list=None, label_list=None, start_index_list=None
):

    pred_tan_list = []
    pred_start_list = []

    gt_color = "green"
    for i, signal_ in enumerate(signal_list):
        if start_index_list is None:
            start_index = signal_.get_start_index(
                theta_cfg["std_coe"], theta_cfg["smoothing_wsize"]
            )
        else:
            start_index = start_index_list[i]

        pred_start_list.append(start_index)
        tan_theta, _error = signal_.get_theta(start_index=start_index, **theta_cfg)
        pred_tan_list.append(tan_theta)

    fig, axs = plt.subplots(2)
    fig.suptitle("Vertically stacked X-Y titls")

    for ind_signal in range(len(signal_list)):
        x = list(range(signal_list[ind_signal][0].shape[-1]))
        if label_list is not None:
            label = label_list[ind_signal]
        else:
            label = "marker-{}".format(ind_signal)
        for fig_ind in range(2):

            axs[fig_ind].plot(
                x,
                signal_list[ind_signal][fig_ind],
                color=color_mapping[ind_signal],
                label=label,
            )

            if fig_ind == 1:
                x_motion = np.array(signal_list[ind_signal][0])
                # print(pred_tan_list)
                pred_y = x_motion * pred_tan_list[ind_signal]

                axs[1].plot(
                    x,
                    pred_y,
                    color=color_mapping[ind_signal],
                    label=label + "-pred",
                    linestyle="dotted",
                )
                axs[1].axvline(
                    pred_start_list[ind_signal], color="cyan", linestyle="-."
                )
                axs[1].axvline(
                    pred_start_list[ind_signal] + theta_cfg["window_size"],
                    color="cyan",
                    linestyle="-.",
                )
                if gt_tan_list is not None:

                    x_motion = np.array(signal_list[ind_signal][0])
                    gt_y = x_motion * gt_tan_list[ind_signal]

                    axs[1].plot(
                        x,
                        gt_y,
                        color=gt_color,
                        label=label + "-gt",
                        linestyle="--",
                    )
    plt.legend()

    plt.show()


def plot_tan_theta_all(
    signal_list,
    theta_cfg,
    gt_tan_list=None,
    start_index_list=None,
    label_list=None,
    save_path=None,
):

    num_signal = len(signal_list)
    pred_tan_list = []
    pred_start_list = []

    for i, signal_ in enumerate(signal_list):
        if start_index_list is None or start_index_list[0] is None:
            start_index = signal_.get_start_index(
                theta_cfg["std_coe"], theta_cfg["smoothing_wsize"]
            )
        else:
            start_index = start_index_list[i]

        pred_start_list.append(start_index)
        tan_theta, _error = signal_.get_theta(start_index=start_index, **theta_cfg)
        pred_tan_list.append(tan_theta)

    fig, axs = plt.subplots(nrows=2, ncols=num_signal, figsize=(20, 4), sharex=True)
    fig.suptitle("X-Y motions & Pred tan(theta) & GT tan(theta) in green")

    gt_color = "green"

    for ind_signal in range(len(signal_list)):
        x = list(range(signal_list[ind_signal][0].shape[-1]))
        if label_list is not None:
            label = label_list[ind_signal]
        else:
            label = "marker-{}".format(ind_signal)

        y_motion = np.array(signal_list[ind_signal][1])

        y_min = np.min(y_motion)
        y_max = np.max(y_motion)

        axs[1][ind_signal].set_ylim([y_min * 1.5, y_max * 1.5])
        for fig_ind in range(2):

            axs[fig_ind][ind_signal].plot(
                x,
                signal_list[ind_signal][fig_ind],
                color=color_mapping[ind_signal],
                label=label,
            )

            if fig_ind == 1:
                x_motion = np.array(signal_list[ind_signal][0])
                # print(pred_tan_list)
                pred_y = x_motion * pred_tan_list[ind_signal]

                axs[1][ind_signal].plot(
                    x,
                    pred_y,
                    color=color_mapping[ind_signal],
                    label=label + "-pred",
                    linestyle="dotted",
                )
                # mark roi
                axs[1][ind_signal].axvline(
                    pred_start_list[ind_signal], color="cyan", linestyle="-."
                )
                axs[1][ind_signal].axvline(
                    pred_start_list[ind_signal] + theta_cfg["window_size"],
                    color="cyan",
                    linestyle="-.",
                )
                if gt_tan_list is not None:

                    x_motion = np.array(signal_list[ind_signal][0])
                    gt_y = x_motion * gt_tan_list[ind_signal]
                    gt_y = np.clip(gt_y, a_min=-100, a_max=100)
                    axs[1][ind_signal].plot(
                        x,
                        gt_y,
                        # color=color_mapping[ind_signal],
                        color=gt_color,
                        label=label + "-gt",
                        linestyle="--",
                    )

    # only add lengend at the final subplot
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_tan_theta_roi(
    signal_list,
    theta_cfg,
    gt_tan_list=None,
    label_list=None,
    start_index_list=None,
    offset=-30,
    plot_window_size=200,
    pred_tan_list=None,
):

    pred_tan_list_ = []
    pred_start_list = []

    gt_color = "green"
    for i, signal_ in enumerate(signal_list):
        if start_index_list is None:
            start_index = signal_.get_start_index(
                theta_cfg["std_coe"], theta_cfg["smoothing_wsize"]
            )
        else:
            start_index = start_index_list[i]

        pred_start_list.append(start_index)

        # might have a bug here
        tan_theta, _error = signal_.get_theta(start_index=start_index, **theta_cfg)
        pred_tan_list_.append(tan_theta)

    if pred_tan_list is None:
        pred_tan_list = pred_tan_list_()

    # change signal to roi signal
    roi_signal_list = []
    roi_plot_theta_cfg = copy.deepcopy(theta_cfg)
    roi_plot_theta_cfg["offset"] = offset
    roi_plot_theta_cfg["window_size"] = plot_window_size
    for i, signal_ in enumerate(signal_list):
        tmp_theta_cfg = copy.deepcopy(roi_plot_theta_cfg)
        tmp_theta_cfg["start_index"] = pred_start_list[i]
        roi_signal = signal_.get_roi_renorm(**tmp_theta_cfg)
        roi_signal_list.append(roi_signal)

    pred_start_list = [int(-1 * offset)] * len(pred_start_list)
    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle("Vertically stacked X-Y titls")

    for ind_signal in range(len(signal_list)):
        x = list(range(roi_signal_list[ind_signal][0].shape[-1]))
        if label_list is not None:
            label = label_list[ind_signal]
        else:
            label = "marker-{}".format(ind_signal)
        for fig_ind in range(2):

            if fig_ind == 0:
                abs_sine = roi_signal_list[ind_signal][1] / (
                    np.sqrt(
                        roi_signal_list[ind_signal][0] ** 2
                        + roi_signal_list[ind_signal][1] ** 2
                    )
                )
                abs_sine = np.abs(abs_sine)

                axs[fig_ind].plot(
                    x,
                    # roi_signal_list[ind_signal][fig_ind],
                    abs_sine,
                    color=color_mapping[ind_signal],
                    label=label,
                )
                axs[0].axvline(
                    pred_start_list[ind_signal], color="cyan", linestyle="-."
                )
                axs[0].axvline(
                    pred_start_list[ind_signal] + theta_cfg["window_size"],
                    color="cyan",
                    linestyle="-.",
                )

            if fig_ind == 1:
                axs[fig_ind].plot(
                    x,
                    roi_signal_list[ind_signal][fig_ind],
                    color=color_mapping[ind_signal],
                    label=label,
                )

                x_motion = np.array(roi_signal_list[ind_signal][0])
                # print(pred_tan_list)
                pred_y = x_motion * pred_tan_list[ind_signal]

                axs[1].plot(
                    x,
                    pred_y,
                    color=color_mapping[ind_signal],
                    label=label + "-pred",
                    linestyle="dotted",
                )
                axs[1].axvline(
                    pred_start_list[ind_signal], color="cyan", linestyle="-."
                )
                axs[1].axvline(
                    pred_start_list[ind_signal] + theta_cfg["window_size"],
                    color="cyan",
                    linestyle="-.",
                )
                if gt_tan_list is not None and 0:

                    x_motion = np.array(roi_signal_list[ind_signal][0])
                    gt_y = x_motion * gt_tan_list[ind_signal]

                    axs[1].plot(
                        x,
                        gt_y,
                        color=gt_color,
                        label=label + "-gt",
                        linestyle="--",
                    )
    plt.legend()

    plt.show()


def plot_pointers(signal_list, marker_locations, gt_loc):
    pass


def plot_points_fusion(
    marker_locations,
    arrow_lists=None,
    source_locations=None,
    pred_locations=None,
    label_list=None,
):
    """
    Args:

    source_locations: list of gt locations. shape of [N, 2]
    pred_locations: list of pred locations. shape of [N, T+1, 2]
        first is the fused prediction
    """
    plt.rcParams["figure.dpi"] = 75
    plt.figure()
    plt.title("2D points plane. Unit as cm")

    plt.xlabel("X axis")
    plt.ylabel("Y axis")

    for i, mxy in enumerate(marker_locations):
        if label_list is not None:
            label = label_list[i]
        else:
            label = "marker-{}".format(i)
        plt.scatter([mxy[0]], mxy[1], color="blue", label=label)

        # add text for markers
        plt.text(mxy[0], mxy[1], s=label, color="blue")

    if arrow_lists is not None:
        for i in range(len(arrow_lists)):
            mxy = marker_locations[i]
            arrowxy = arrow_lists[i]
            plt.arrow(mxy[0], mxy[1], arrowxy[0], arrowxy[1], color=color_mapping[i])

    if source_locations is not None:
        for i in range(len(source_locations)):
            plt.scatter(
                [source_locations[i][0]],
                [source_locations[i][1]],
                label="GT",
                color=color_mapping[i],
            )

    if pred_locations is not None:
        for i in range(len(pred_locations)):
            x_list = [pred_locations[i][j][0] for j in range(len(pred_locations[i]))]
            y_list = [pred_locations[i][j][1] for j in range(len(pred_locations[i]))]
            mean_x = np.mean(np.array(x_list))
            mean_y = np.mean(np.array(y_list))
            plt.scatter(
                [mean_x],
                [mean_y],
                label="avg pred",
                color=color_mapping[i],
                marker="*",
            )
            plt.scatter(
                [x_list],
                [y_list],
                label="pred",
                color=color_mapping[i],
                marker="^",
                alpha=0.3,
            )

    # plt.legend()

    plt.show()


def plot_list(y_list, label_list=None, x_axis=None):

    if x_axis is None:
        x_list = list(range(len(y_list[0])))
    else:
        x_list = x_axis

    plt.figure()
    for i, y in enumerate(y_list):
        if label_list is None:
            plt.plot(x_list, y)
        else:
            plt.plot(x_list, y, label=label_list[i])
    if label_list is not None:
        plt.legend()

    plt.show()
