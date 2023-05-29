import numpy as np

def load_recording(loaded_filename):
    data = np.load(loaded_filename, allow_pickle=True)
    frame_recording = data["frame_recording"]
    experiment_name = (
        np.array2string(data["experiment_name"]).replace('"', "").replace("'", "")
    )

    try:
        run_opt = data["run_opt"].item()
    except:
        print("error loading run_opt")
        pass

    print_parameters(run_opt)

    print("\nLoaded {} data samples".format(frame_recording[0].shape[0]))

    # samp_freq = run_opt['linescan_sampling_freq'] if run_opt['is_linescan_mode'] else run_opt['capture_params']['frame_rate_eff'][0]
    return frame_recording, run_opt, experiment_name


def print_parameters(run_opt):
    def print_table(opt_dict, do_not_print_keys, spacing=30):
        print(
            "| Parameter".ljust(spacing) + "|".ljust(2) + "Value".ljust(spacing), end=""
        )
        print(
            "| Parameter".ljust(spacing) + "|".ljust(2) + "Value".ljust(spacing) + "|"
        )
        print(
            "| -------------".ljust(spacing)
            + "|".ljust(2)
            + ":-------------:".ljust(spacing),
            end="",
        )
        print(
            "| -------------".ljust(spacing)
            + "|".ljust(2)
            + ":-------------:".ljust(spacing)
            + "|"
        )
        column = 0
        for key, value in opt_dict.items():
            if key not in do_not_print_keys:
                if type(value) == list and len(value) == 1:
                    value = value[0]
                if type(value) == np.ndarray and value.size == 1:
                    value = value[0]

                print("| {}".format(key).ljust(spacing), end="")
                print("|".ljust(2), end="")
                print("{}".format(value).ljust(spacing), end="")
                if column == 1:
                    print("|")
                column = not column
        if column == 1:
            print("|".ljust(spacing) + "|".ljust(spacing + 2) + "|")

    def print_in_table(opt_dict, print_keys, spacing=30):
        print(
            "| Parameter".ljust(spacing) + "|".ljust(2) + "Value".ljust(spacing), end=""
        )
        print(
            "| Parameter".ljust(spacing) + "|".ljust(2) + "Value".ljust(spacing) + "|"
        )
        print(
            "| -------------".ljust(spacing)
            + "|".ljust(2)
            + ":-------------:".ljust(spacing),
            end="",
        )
        print(
            "| -------------".ljust(spacing)
            + "|".ljust(2)
            + ":-------------:".ljust(spacing)
            + "|"
        )
        column = 0
        for key in print_keys:
            value = opt_dict[key]
            if type(value) == list and len(value) == 1:
                value = value[0]
            if type(value) == np.ndarray and value.size == 1:
                value = value[0]

            print("| {}".format(key).ljust(spacing), end="")
            print("|".ljust(2), end="")
            print("{}".format(value).ljust(spacing), end="")
            if column == 1:
                print("|")
            column = not column
        if column == 1:
            print("|".ljust(spacing) + "|".ljust(spacing + 2) + "|")

    def print_flat(opt_dict, keys):
        for key in keys:
            try:
                print(f"{key}:".ljust(20) + f"{opt_dict[key]}")
            except:
                print(f"{key}:".ljust(20) + "None")

    print("Capture parameters:")
    print_keys = ["exposure_eff", "frame_rate_eff"]
    print_in_table(run_opt["capture_params"], print_keys, spacing=30)

    print("\nExperiment:")
    # do_not_print_keys = ['signal_gt','capture_params','frame_DC','recovery','two_cam_homography_M','post_processing','map_x','map_y']+dir_keys
    # print_table(run_opt,do_not_print_keys,spacing=30)
    dir_keys = ["output_dir", "recovery_dir", "export_sound_dir"]
    print_flat(run_opt, dir_keys)