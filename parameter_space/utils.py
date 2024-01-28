import os


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


if __name__ == "__main__":
    # make_dir("directions")

    for i in range(64):
        # make_dir(f"directions/direction{i}")
        for amplitude in [10, 50, 100, 200, 500]:
            # make_dir(f"directions/direction{i}/amplitude{amplitude}")
            try:
                os.rmdir(f"directions/direction{i}/amplitude{amplitude}")
            except:
                continue

