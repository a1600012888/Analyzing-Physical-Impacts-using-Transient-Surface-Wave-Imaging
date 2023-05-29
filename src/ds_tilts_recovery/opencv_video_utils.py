import numpy as np
import cv2

import traceback
from .image_processing import adjustGamma

FONT = cv2.FONT_HERSHEY_SIMPLEX


class videoPlayer:
    def __init__(self, frame_func, N_frames, resize_factor=None):
        self.frame_func = frame_func
        self.data_counter = 0
        self.resize_factor = resize_factor
        self.gamma = 1
        self.N_frames = N_frames

        # -------- Loop control

    def loop_control(self, key):
        if key == 13:
            return 1
        if key == 43:
            self.data_counter = (self.data_counter + 1) % self.N_frames
        if key == 45:
            self.data_counter = (self.data_counter - 1) % self.N_frames
        if key == 40:
            self.gamma -= 0.1
        if key == 41:
            self.gamma += 0.1
        return 0

    def additional_loop_control(self, key):
        return 0

    def correct_gamma_show(self, img):
        img = img.astype("float32") ** (1 / self.gamma)
        if self.gamma != 1:
            cv2.putText(
                img,
                "g={:.1f}".format(self.gamma),
                (img.shape[1] - 150, 20),
                FONT,
                0.75,
                (1, 1, 1),
                2,
                cv2.LINE_AA,
            )
        return img

    def get_frame_show(self):
        return self.frame_func(self.data_counter)

    def play_video(self, move_window=1, show_frame_number=True):
        try:
            cv2.namedWindow("frame")
            if move_window:
                cv2.moveWindow("frame", 1920 * 2 + 5, 0)

            while True:
                frame_show = self.get_frame_show()

                if self.resize_factor is not None:
                    frame_show = cv2.resize(
                        frame_show, None, fx=self.resize_factor, fy=self.resize_factor
                    )

                frame_show = self.correct_gamma_show(frame_show)
                color = (255, 255, 255) if frame_show.dtype == "uint8" else (1, 1, 1)
                if show_frame_number:
                    cv2.putText(
                        frame_show,
                        "frame {}".format(self.data_counter),
                        (10, 20),
                        FONT,
                        0.75,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow("frame", frame_show)
                key = cv2.waitKey()
                # if key!=-1:
                #    print(key)
                self.additional_loop_control(key)
                if self.loop_control(key):
                    break

        except Exception:
            traceback.print_exc()
            print("Closing camera thread.")
            # cleanup
        cv2.destroyAllWindows()

    def export_video(self, filename, is_flip_RGB=0, gamma=1.0, FPS=30):
        IMG_H, IMG_W = self.get_frame_show().shape[:2]
        fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
        video = cv2.VideoWriter(
            filename,
            fourcc,  # cv2.VideoWriter_fourcc('X','2','6','4'),
            FPS,
            (IMG_W, IMG_H),
        )
        self.gamma = gamma

        for i in range(self.N_frames):
            self.data_counter = i
            frame_show = self.get_frame_show()
            if is_flip_RGB:
                frame_show = frame_show[:, :, ::-1]

            if gamma != 1:
                frame_show = self.correct_gamma_show(frame_show)
            # frame is [0,1]
            if frame_show.dtype != "uint8":
                frame_show = (frame_show.clip(0, 1) * 255).astype("uint8")
            if frame_show.ndim == 2:
                frame_show = np.stack(((frame_show,) * 3), axis=-1)
            video.write(frame_show)
        video.release()
        print("done exporting video to {}".format(filename))


class videoPlayerRecording(videoPlayer):
    def __init__(self, frame_func, N_frames, resize_factor=None):
        videoPlayer.__init__(self, frame_func, N_frames, resize_factor=resize_factor)
        self.is_draw_ROI = 0
        self.apply_rotation = 1
        self.gamma = 2

    def additional_loop_control(self, key):
        if key == ord("a"):
            self.is_draw_ROI = not self.is_draw_ROI
        if key == ord("1"):  # numpad 1
            self.apply_rotation = not self.apply_rotation
        return 0

    def get_frame_show(self):
        return self.frame_func(self.data_counter, self.is_draw_ROI, self.apply_rotation)

    def correct_gamma_show(self, img):
        img = adjustGamma(img, self.gamma)
        if self.gamma != 1:
            cv2.putText(
                img,
                "g={:.1f}".format(self.gamma),
                (img.shape[1] - 150, 20),
                FONT,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return img
