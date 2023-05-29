import numpy as np
from tqdm import tqdm
import cv2
import time
import traceback

# might remove below two libs if we don't use graph cut to smooth the results
import maxflow.fastmin as fm
from scipy.spatial import distance_matrix


def compute_CAM2_translations_v2(
    video_pp_cam2, functions, run_opt, W_matrix=None, debug=0
):
    """
    compute the shifts between global-shutter frames by calling
        cross-correlation functions (specified in functions) on ajacent global-shutter frames
    Args:
        video_pp_cam2: global-shutter video in shape of [num_frames_gs, H, W]
        functions: cross-correlation function
    """
    n_ref_frames = video_pp_cam2.shape[0]
    all_reference_shifts = []
    succ_frames = np.zeros((n_ref_frames,), dtype=bool)

    print("----- Computing shifts -----")

    if debug:
        video_pp_cam2_debug = []
        video_pp_cam2_debug.append(video_pp_cam2[0])
    else:
        video_pp_cam2_debug = None

    M_total = np.eye(2, 3, dtype=np.float32)
    N_funcs = len(functions)
    succ_frames[0] = True
    all_reference_shifts.append([0, 0])

    prev = 0
    for i in tqdm(range(1, n_ref_frames)):
        for function in functions:
            reference_shift, M, ret = function(
                video_pp_cam2[prev].copy(), video_pp_cam2[i].copy(), W_matrix
            )
            if ret:
                break
            else:
                print(f"{function.__name__} failed for frame {i}")

        if ret == 1:
            succ_frames[i] = True
            all_reference_shifts.append(reference_shift)
            prev = i
            M_total[:, -1] += M[:, -1]
            if debug:
                video_pp_cam2_debug.append(
                    cv2.warpAffine(
                        video_pp_cam2[i].copy(), M_total, video_pp_cam2[0].shape[::-1]
                    )
                )
        else:
            if debug:
                video_pp_cam2_debug.append(video_pp_cam2[i])
            print(f"frame {i} failed")

    return np.array(all_reference_shifts).squeeze(), video_pp_cam2_debug, succ_frames


def find_frame_translation_phase_corr(frame_1, frame_2, inputMask):
    def diff(im):
        grad_x = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=3)
        grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
        return grad

    frame_1 = diff(frame_1)
    frame_2 = diff(frame_2)
    h, w = frame_1.shape
    hannW = cv2.createHanningWindow((w, h), cv2.CV_64F)

    # pad width
    def get_pad_size(w):
        two_power = 2 ** np.ceil(np.log2(w))
        left_pad = int((two_power - w) // 2)
        right_pad = int(two_power - w - left_pad)
        return left_pad, right_pad

    left_pad, right_pad = get_pad_size(w)
    up_pad, down_pad = get_pad_size(h)

    def pad(frame):
        return np.pad(frame, ((up_pad, down_pad), (left_pad, right_pad)))

    frame_1 = pad(frame_1)
    frame_2 = pad(frame_2)
    hannW = pad(hannW)

    shift, corr_val = cv2.phaseCorrelate(frame_2, frame_1, hannW)
    ret = 1
    if corr_val < 0.1:
        ret = 0

    tx, ty = shift[0], shift[1]
    M = np.hstack((np.eye(2), np.array([[tx], [ty]])))
    return shift, M, ret


def find_closets_reference_frames(RS_timestamp, ref_timestamps, K):
    RS_timestamp_ms = (
        RS_timestamp[3] + RS_timestamp[2] * 1000 + RS_timestamp[1] * 1000 * 60
    )
    ref_timestamps = (
        ref_timestamps[:, 3]
        + ref_timestamps[:, 2] * 1000
        + ref_timestamps[:, 1] * 1000 * 60
    )
    diff = np.abs(RS_timestamp_ms - 8 - ref_timestamps)
    idx_sorted = np.argsort(diff)
    return idx_sorted[:K]


def normalize_tensor_for_corr(A):  # (2, 984, 270)
    N_non_zero = A.shape[-1]
    mean = np.sum(A, axis=-1) / N_non_zero  # (2, 984)
    A = A - mean[:, :, None]
    std = np.sqrt(np.sum((A) ** 2, axis=-1) / N_non_zero)
    A = A / (std[:, :, None] + np.finfo(float).eps)
    return A, N_non_zero


def compute_coarse_xy_using_phase_correlation(
    frame, ref, cam_0_pad, W_margin, y_range=20, x_range=20, Lambda=1000
):
    def unpad(img):
        if cam_0_pad > 0:
            return img[cam_0_pad:-cam_0_pad]
        else:
            return img

    frame, N_non_zero = normalize_tensor_for_corr(frame[None, ..., W_margin:-W_margin])
    ref, N_non_zero = normalize_tensor_for_corr(ref[None, ..., W_margin:-W_margin])

    frame, ref = frame.squeeze(), ref.squeeze()

    def zpad(img, n_pads=1):
        signal_len = img.shape[1]
        img = np.pad(img, ((0, 0), (0, signal_len * n_pads)))
        return img

    frame = zpad(frame, n_pads=1)
    ref = zpad(ref, n_pads=1)

    # hannW        = np2cp(np.hanning(frame.shape[1]))
    frame_F = np.fft.fft(frame)  # *hannW) # (944, 216)
    ref_F_conj = np.conj(np.fft.fft(ref))  # (984, 216)

    y_shift_vec = np.arange(-y_range, y_range)  # y_range = 100
    N_shifts = len(y_shift_vec)
    ref_F_conj_mat = np.zeros((N_shifts,) + frame_F.shape, dtype=frame_F.dtype)

    for i in range(N_shifts):
        ref_F_conj_mat[i] = unpad(np.roll(ref_F_conj, y_shift_vec[i], axis=0))

    # R            = frame_F * ref_F_conj_mat / cp.abs(frame_F * ref_F_conj_mat + np.finfo(complex).eps)
    R = frame_F * ref_F_conj_mat / N_non_zero
    r = np.abs(np.fft.ifft(R))  # (40, 944, 216)

    max_y = np.max(r, axis=-1)  # (40, 944)

    y, d, score_y = graph_cut_solution_single_axis(max_y.T, y_shift_vec, Lambda=Lambda)
    N_rows = frame_F.shape[0]

    # x-axis:
    h, w = frame.shape
    assert (w // 2 - x_range) >= 0
    assert (w // 2 + x_range + 1) <= w

    all_max_F = r[d, np.arange(N_rows)]
    all_max_F = np.fft.fftshift(all_max_F, axes=-1)  # (944, 216)

    r_crop = [w // 2 - x_range, w // 2 + x_range + 1]
    all_max_F = all_max_F[..., r_crop[0] : r_crop[1]]

    x_shift_vec = np.arange(w) - w // 2
    x_shift_vec = x_shift_vec[r_crop[0] : r_crop[1]]

    x, d, score_x = graph_cut_solution_single_axis(
        all_max_F, x_shift_vec, Lambda=Lambda
    )

    return x[:, None], y[:, None]


def graph_cut_solution_single_axis(X, y_shifts, Lambda=1):
    # X is (944,200)
    N_rows = X.shape[0]
    V = distance_matrix(y_shifts[:, None], y_shifts[:, None]) ** 2

    d0 = np.argmax(X, axis=1)
    # lower D is better
    D = (1 - X) * Lambda
    d = fm.aexpansion_grid(D, V, labels=d0)
    y = y_shifts[d]

    score = X[np.arange(N_rows), d]

    return y, d, score


def run_recovery(
    frame_index_list, video_pp, video_pp_cam2, all_reference_shifts, run_opt
):
    """
    run recovery for a list of frames.
    For recovery of each frame, it contains the following steps:
    1. For each rolling-shutter frame, find the closest reference global frame.
    2. For each pair of (rolling-shutter frame, global-shutter frame).
        We loop of each row of the rolling-shutter frame, and compute the phase correlation
        between the row and the global-shutter frame to get a 2D shift vector.
    3. Since the global shutter frame is also moving, we need to account for it by translate the recovered shifts
        using all_reference_shifts
    Args:
        frame_index_list: list of frame indices of rolling-shutter frames to recover
        video_pp: rolling-shutter video in shape of [num_frames_rs, H, W]
        video_pp_cam2: global-shutter video in shape of [num_frames_gs, H, W]
        all_reference_shifts: list of shifts of global-shutter frames
        run_opt: dict of options
    """
    N_ref_frames = 1

    # final recovery for each frames
    x_vec, y_vec = [], []

    start_time = time.time()

    idx_cam2_vec = np.zeros(
        (
            len(
                frame_index_list,
            )
        ),
        dtype=int,
    )

    j = -1
    try:
        for frame_idx in tqdm(frame_index_list):  # <-------------- FOR LOOP
            j += 1

            # CREATE the REFERENCE FRAME

            RS_timestamp = run_opt["time_stamps"][0][frame_idx]
            use_frames = find_closets_reference_frames(
                RS_timestamp, run_opt["ref_time_stamps"], K=N_ref_frames
            )

            idx_cam2_vec[j] = use_frames[0]

            reference_frame = video_pp_cam2[use_frames][0]

            frame_RS = video_pp[
                frame_idx : frame_idx + 1
            ].copy()  # <-------- maybe remove copy

            # COMPUTE THE SHIFT
            W_margin = 25

            cam_0_pad = run_opt["cam_0_pad"]
            y_range = run_opt["recovery"]["motion_model_y_range"][1]
            x_range = run_opt["recovery"]["motion_model_x_range"][1]

            phase_corr_function = compute_coarse_xy_using_phase_correlation

            x, y = phase_corr_function(
                frame_RS[0],
                reference_frame,
                cam_0_pad,
                W_margin,
                y_range=y_range,
                x_range=x_range,
                Lambda=run_opt["recovery"]["graph_cut_Lambda"][0],
            )

            x_vec.append(x)
            y_vec.append(y)

        x_vec = np.array(x_vec, dtype=np.float64).squeeze()
        y_vec = np.array(y_vec, dtype=np.float64).squeeze()

        end_time = time.time()
        print("Done recovering {:0.2f} min".format((end_time - start_time) / 60))

        # apply per-frame shifts
        for i in range(1, len(frame_index_list)):
            shift = -all_reference_shifts[
                idx_cam2_vec[i - 1] + 1 : idx_cam2_vec[i] + 1
            ].sum(axis=0)

            x_vec[i:] += shift[0]
            y_vec[i:] += shift[1]

    except Exception:
        traceback.print_exc()
        pass

    return x_vec, y_vec
