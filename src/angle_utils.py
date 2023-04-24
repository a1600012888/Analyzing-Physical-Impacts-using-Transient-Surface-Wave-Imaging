import numpy as np
from pyparsing import line
from scipy.optimize import least_squares


def get_normalized_squared_error(x_roi, y_roi, tan_theta):
    y_max = np.max(np.abs(y_roi))
    x_max = np.max(np.abs(x_roi))
    norm_ = max(x_max, y_max)

    x_roi = x_roi / norm_
    y_roi = y_roi / norm_

    y_fit = x_roi * tan_theta

    l2_error = y_roi - y_fit

    line_distance = l2_error * (np.abs(np.cos(np.arctan(tan_theta))))

    squared_error = line_distance**2

    ret_error = np.mean(squared_error) * 10.0
    # print("uncertainty", ret_error, tan_theta)

    return ret_error


def get_median(x_roi, y_roi):
    norm_ = np.sqrt(x_roi**2 + y_roi**2 + 1e-8)

    sine_theta = np.median(y_roi / norm_)
    cosine_theta = np.median(x_roi / norm_)

    theta = np.arctan2(sine_theta, cosine_theta)

    return theta


def get_mean(x_roi, y_roi):
    norm_ = np.sqrt(x_roi**2 + y_roi**2 + 1e-8)

    sine_theta = np.mean(y_roi / norm_)
    cosine_theta = np.mean(x_roi / norm_)

    theta = np.arctan2(sine_theta, cosine_theta)

    return theta


def get_median_post(x_roi, y_roi):
    norm_ = np.sqrt(x_roi**2 + y_roi**2 + 1e-8)

    sine_theta = y_roi / norm_
    cosine_theta = x_roi / norm_

    theta = np.arctan2(sine_theta, cosine_theta)

    theta = np.median(theta)
    return theta


def get_median_tan(x_roi, y_roi):

    # clip x_roi

    sign_x = np.sign(x_roi)
    abs_x = np.abs(x_roi)
    abs_x = np.clip(abs_x, a_min=1e-5, a_max=None)
    x_roi = abs_x * sign_x
    tan_theta = y_roi / x_roi

    median_tan_theta = np.median(tan_theta)

    # error = get_normalized_squared_error(x_roi, y_roi, median_tan_theta)
    error = np.abs(y_roi - x_roi * median_tan_theta).mean()
    # print(error)

    return median_tan_theta, error


def get_line_residual_2d(line_param, x_list, y_list):
    """
    Args:
        x_list: np.ndarray of shape [N,1]
        y_list: np.ndarray of shape [N,1]
        line_param: np.ndarray of shape [2, ]
            y = x * line_param[0] + line_param[2]
    """

    y_pred = line_param[0] * x_list + line_param[1]
    residual = np.abs(y_pred - y_list)

    return residual


def get_tan_softl1(x_roi, y_roi, params=None):
    """
    x_roi [N,]
    y_roi [N,]
    """
    # y = k * x + m
    line_param = np.array([1.0, 0.0])
    lst_l1 = least_squares(
        get_line_residual_2d,
        line_param,
        loss="soft_l1",
        f_scale=0.5,
        args=(x_roi, y_roi),
    )

    line_param = lst_l1.x
    ret_error = lst_l1.cost / x_roi.shape[0]

    #
    return line_param[0], ret_error


def get_line_residual_nobias_2d(line_param, x_list, y_list):
    """
    Args:
        x_list: np.ndarray of shape [N,1]
        y_list: np.ndarray of shape [N,1]
        line_param: np.ndarray of shape [2, ]
            y = x * line_param[0] + line_param[2]
    """

    y_pred = line_param[0] * x_list
    residual = np.abs(y_pred - y_list)

    return residual


def get_tan_softl1_nobias(x_roi, y_roi, params=[1.0]):
    """
    x_roi [N,]
    y_roi [N,]
    """
    # y = k * x + m
    line_param = np.array([1.0])
    lst_l1 = least_squares(
        get_line_residual_nobias_2d,
        line_param,
        loss="soft_l1",
        f_scale=0.5,
        args=(x_roi, y_roi),
    )

    line_param = lst_l1.x

    # compute mean abs relative error
    residual_y = y_roi - x_roi * line_param[0]

    abs_rel = np.abs(residual_y / (np.abs(y_roi) + 1e-3))
    ret_error = np.mean(abs_rel)

    # using searched error
    ret_error = lst_l1.cost / x_roi.shape[0]
    return line_param[0] / params[0], ret_error


Name2AngleFunc = {
    "median": get_median,
    "mean": get_mean,
    "median_post": get_median_post,
    "median_tan": get_median_tan,
    "soft_l1": get_tan_softl1,
    "soft_l1_nobias": get_tan_softl1_nobias,
}
