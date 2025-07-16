"""
 Copyright (c) 2024, Deep Representation Learning Research Group, Seoul National University.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import sys
import os
import math
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from pipeline_processor.record import *
from .fps_extractor import *
from .gridview_generator import *
from .video_validation import *
from enum import Enum


class FpsDataProcessor:
    def __init__(
        self,
        calculate_max_row=lambda x: round(math.sqrt(x)),
        save_option="image",
        frame_fixed_number=6,
    ):
        self.calculate_max_row = calculate_max_row
        self.frame_fixed_number = frame_fixed_number
        self.save_option = save_option

    def process(self, video_path, ts=None):
        fps_extractor = FpsExtractor(video_path)
        grid_view_creator = GridViewCreator(self.calculate_max_row)

        try:
            print("START PROCESS")
            rlt_fps_extractor = fps_extractor.save_data_based_on_option(
                "numpy",
                frame_fixed_number=self.frame_fixed_number,
                ts=ts,
            )
            print("HAHA")
            rlt_grid_view_creator = grid_view_creator.post_process_based_on_options(
                self.save_option, rlt_fps_extractor
            )
            print("END PROCESS")
        except Exception as e:
            print(f"Exception: {e} on {video_path}")
            return -1

        return rlt_grid_view_creator

def main():
    video_name = "L01_V001_480p_sub1.mp4"
    video_path = os.path.join("/workspace/data/Sub_video/L01", video_name)
    print(f"Video path: {video_path}")
    fps_data_processor = FpsDataProcessor(
        save_option=SaveOption.IMAGE,
        frame_fixed_number=6,
    )
    print(vars(fps_data_processor))
    print(f"Video path: {video_path}")
    rlt = fps_data_processor.process(video_path)
    if rlt != -1:
        rlt.save(f"./example/imagegrid_sample/{video_name.split('.')[0]}.jpg")
    else:
        print(f"Failed to process {video_path}")

if __name__ == "__main__":
    main()