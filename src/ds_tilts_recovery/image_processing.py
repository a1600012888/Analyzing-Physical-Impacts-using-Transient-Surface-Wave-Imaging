import numpy as np
import cv2
import traceback


FONT = cv2.FONT_HERSHEY_SIMPLEX


def adjustGamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def convert_frame_to_uint8(frame):
    # frame is debayered
    if frame.dtype == "uint8":
        return frame
    else:
        return np.floor_divide(frame, 256).astype("uint8")


def debayer_sequence(seq, mode="BGR"):
    if mode == "BGR":
        mode = cv2.COLOR_BAYER_BG2BGR
    else:
        mode = cv2.COLOR_BAYER_BG2RGB

    seq_color = np.zeros((seq.shape[0], seq.shape[1], seq.shape[2], 3), dtype=seq.dtype)
    for i in range(seq.shape[0]):
        seq_color[i] = cv2.cvtColor(seq[i], mode)
    return seq_color


def display_16bit_BG(frame, is_out_RGB=1, gamma=1.4):
    frame = convert_frame_to_uint8(frame)
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
    if is_out_RGB:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = adjustGamma(frame, gamma)
    return frame


def bgr2rgb(img):
    return img[..., ::-1].copy()


def combine_frames_for_show(frames, gamma_val, use_grid):
    frame_show = frames[0]
    H, W = frames[0].shape[0:2]

    try:
        for i in range(1, len(frames)):
            if frames[i].shape[0] == frame_show.shape[0]:
                frame_show = np.concatenate((frame_show, frames[i]), axis=1)
            else:
                Hn, Wn = frames[i].shape[0:2]

                frame_to_add = cv2.resize(frames[i], None, fx=H / Hn, fy=H / Hn)

                frame_show = np.concatenate((frame_show, frame_to_add), axis=1)

    except Exception:
        traceback.print_exc()
        pass

    frame_show = adjustGamma(frame_show, gamma=gamma_val)
    if len(frames) > 1:
        cv2.line(
            frame_show,
            (frames[0].shape[1], 0),
            (frames[0].shape[1], frames[0].shape[0]),
            (0, 0, 255),
            1,
        )
        cv2.line(
            frame_show,
            (frames[0].shape[1] + frames[1].shape[1], 0),
            (frames[0].shape[1] + frames[1].shape[1], frames[0].shape[0]),
            (0, 0, 255),
            1,
        )

    # -------- Draw grid

    if use_grid:
        H, W = frames[0].shape[0:2]
        width = 1
        for i in range(len(frames)):
            cv2.line(
                frame_show,
                (W * i, H // 2),
                (W * (i + 1), H // 2),
                (255, 255, 255),
                width,
            )
            cv2.line(
                frame_show,
                (W * i + W // 2, 0),
                (W * i + W // 2, H),
                (255, 255, 255),
                width,
            )
    if gamma_val != 1:
        cv2.putText(
            frame_show,
            "g={:.1f}".format(gamma_val),
            (frame_show.shape[1] - 150, 20),
            FONT,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return frame_show
