from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import config as c


class Visualizer:
    def __init__(self, loss_labels):
        self.loss_labels = list(loss_labels)
        self.counter = 1
        self.output_dir = Path(getattr(c, "IMAGE_PATH_demo", "runtime/demo"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def update_losses(self, losses, *args):
        line = f"{self.counter:03d}"
        for loss in losses:
            line += f"\t{float(loss):.4f}"
        print(line)
        self.counter += 1

    def update_images(self, *img_list):
        if not img_list:
            return None
        panels = []
        for image in img_list:
            array = np.asarray(image)
            if array.ndim == 3 and array.shape[0] in {1, 3}:
                array = np.transpose(array, (1, 2, 0))
            array = np.clip(array, 0.0, 1.0)
            panels.append(array)
        canvas = np.concatenate(panels, axis=1)
        path = self.output_dir / f"{self.counter:04d}.png"
        plt.imsave(path, canvas)
        return canvas

    def update_hist(self, data):
        return np.asarray(data)

    def update_running(self, running):
        print(f"running={bool(running)}")

    def close(self):
        plt.close("all")


visualizer = Visualizer(c.loss_names)


def show_loss(losses, logscale=False):
    del logscale
    visualizer.update_losses(losses)


def show_imgs(*imgs):
    return visualizer.update_images(*imgs)


def show_hist(data):
    return visualizer.update_hist(data)


def signal_start():
    visualizer.update_running(True)


def signal_stop():
    visualizer.update_running(False)


def close():
    visualizer.close()
