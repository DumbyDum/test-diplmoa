from __future__ import annotations

import unittest

import numpy as np

from omniguard.editing import build_editor
from omniguard.settings import RuntimeSettings


class EditingTests(unittest.TestCase):
    def test_opencv_editors_produce_image(self) -> None:
        settings = RuntimeSettings()
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[..., 1] = 120
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 255

        for editor_name in ("opencv-telea", "opencv-ns"):
            editor = build_editor(settings, editor_name=editor_name)
            result = editor.edit(image, mask, "test prompt")
            self.assertEqual(result.image.shape, image.shape)
            self.assertEqual(result.backend_name, editor_name)


if __name__ == "__main__":
    unittest.main()
