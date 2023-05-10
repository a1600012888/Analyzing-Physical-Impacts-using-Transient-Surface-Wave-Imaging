"""
Description: helper functions for visualing measured tilts signals 
Author: Tianyuan Zhang
"""

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
            np.array([116, 195, 101]) / 255.0,  # 19 light mint green
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
