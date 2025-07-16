"""
 Copyright (c) 2024, Deep Representation Learning Research Group, Seoul National University.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import uuid
import math
import os
import time
from tqdm import tqdm

from model_processor.llava2_model_processor import Llava157BProcessor, LlavaNeXTVideo7BProcessor
from vision_processor.fps_gridview_processor import FpsDataProcessor

class Llava15_7B_Pipeline:
    def __init__(
        self,
        model_name,
        path_qa,
        path_video_file_format,
        dir="./llava_pipeline_result/",
        video_mode="single",
    ):
        self.model_name = model_name
        self.path_qa = path_qa
        self.path_dir = dir
        self.path_result = dir
        self.path_video_file_format = path_video_file_format
        self.video_mode = video_mode
        self.error_video_name = []
        self.load_model()

    def load_model(self):
        print("[DEBUG] Initializing Llava15_7BProcessor")
        self.model = Llava157BProcessor(self.model_name)
        self.model.load_model()
        print("[DEBUG] Model loaded in LlavaPipeline")

    def set_component(
        self,
        user_prompt,
        frame_fixed_number=6,
        func_user_prompt=lambda prompt, row: prompt % row,
        calculate_max_row=lambda x: round(math.sqrt(x)),
    ):
        if not hasattr(self, "model"):
            raise AttributeError("Model is not loaded. Please call load_model()")

        self.frame_fixed_number = frame_fixed_number
        self.user_prompt = user_prompt
        self.func_user_prompt = func_user_prompt
        self.calculate_max_row = calculate_max_row

        self.fps_data_processor = FpsDataProcessor(
            save_option="image",
            calculate_max_row=self.calculate_max_row,
            frame_fixed_number=self.frame_fixed_number,
        )

        extra_dir = f"ffn={self.frame_fixed_number}/"
        self._make_directory(extra_dir)

    def do_pipeline(self, qa_text=None):
        # print("start pipeline")
        start_time = time.time()  # Record start time

        question_id = str(uuid.uuid4())
        video_path = self.path_video_file_format
        ts = None

        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            self.error_video_name.append(video_path)
            return None, None, None

        try:
            image_data = self.fps_data_processor.process(video_path, ts)
            # print("[DEBUG]: IMAGE PROCESSED")
            if image_data == -1:
                print(f"Failed to process video: {video_path}")
                self.error_video_name.append(video_path)
                return None, None, None

            answer = self.model.inference(
                user_prompt=self.func_user_prompt(self.user_prompt, qa_text),
                raw_image=image_data,
            )
            # print("[DEBUG]: ANSWER PROCESSED")
            extracted_answer = self.model.extract_answers()
            # print("[DEBUG]: ANSWER EXTRACTED")
            result_file_path = self.write_result_file(question_id, extracted_answer)

            end_time = time.time()  # Record end time
            response_time = end_time - start_time  # Calculate response time in seconds
            return extracted_answer, result_file_path, response_time

        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            self.error_video_name.append(video_path)
            return None, None, None

    def write_result_file(self, question_id, answer, extension=".txt"):
        file_path = self._make_file_path(question_id, extension)
        with open(file_path, "w") as file:
            file.write(answer)
        return file_path

    def _make_file_path(self, question_id, extension=".txt"):
        return os.path.join(self.path_result, f"result_{question_id}{extension}")

    def _make_directory(self, extra_dir):
        self.path_result = os.path.join(self.path_dir, extra_dir)
        os.makedirs(self.path_result, exist_ok=True)