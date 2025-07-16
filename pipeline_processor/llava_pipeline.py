"""
 Copyright (c) 2024, Deep Representation Learning Research Group, Seoul National University.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import uuid
import math
import pandas as pd
import os
import sys
from tqdm import tqdm
import glob

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model_processor.llava2_model_processor import Llava2Processor
from vision_processor.fps_gridview_processor import FpsDataProcessor
from vision_processor.fps_extractor import FpsExtractor
from .record import *
from enum import Enum

class SaveOption(Enum):
    BYTES = "bytes"
    FILE = "file"
    BASE64 = "base64"
    NUMPY = "numpy"
    IMAGE = "image"

class LlavaPipeline:
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
        self.make_video_file_list()
        self.load_model()

    def make_video_file_list(self):
        self._load_qa_file()
        if self.video_mode == "single":
            # Use only the specified video
            video_name = os.path.splitext(os.path.basename(self.path_video_file_format))[0]
            self.df_qa = self.df_qa[self.df_qa["video_name"] == video_name]
            if self.df_qa.empty:
                raise ValueError(f"No QA entries found for video: {video_name}")
            self.df_qa["path_video"] = self.path_video_file_format
        else:  # video_mode == "folder"
            # Get all video files in the directory
            video_extensions = ["*.avi", "*.mp4", "*.mkv", "*.webm", "*.gif"]
            video_paths = []
            for ext in video_extensions:
                video_paths.extend(glob.glob(os.path.join(self.path_video_file_format, ext)))
            video_names = [os.path.splitext(os.path.basename(p))[0] for p in video_paths]
            if not video_names:
                raise FileNotFoundError(f"No valid video files found in directory: {self.path_video_file_format}")
            # Filter QA DataFrame to include only specified videos
            self.df_qa = self.df_qa[self.df_qa["video_name"].isin(video_names)]
            if self.df_qa.empty:
                raise ValueError(f"No QA entries found for videos in directory: {self.path_video_file_format}")
            self.df_qa["path_video"] = self.df_qa.apply(
                lambda x: next((p for p in video_paths if os.path.splitext(os.path.basename(p))[0] == x["video_name"]), None),
                axis=1
            )
            self.df_qa = self.df_qa.dropna(subset=["path_video"])  # Remove rows with no matching video

    def load_model(self):
        self.model = Llava2Processor(self.model_name)
        self.model.load_model()

    def set_component(
        self,
        user_prompt,
        frame_fixed_number=6,
        func_user_prompt=lambda prompt, row: prompt % (row["question"]),
        calculate_max_row=lambda x: round(math.sqrt(x)),
    ):
        if not hasattr(self, "model"):
            raise AttributeError("Model is not loaded. Please call load_model()")

        self.frame_fixed_number = frame_fixed_number
        self.user_prompt = user_prompt
        self.func_user_prompt = func_user_prompt
        self.calculate_max_row = calculate_max_row

        self.fps_data_processor = FpsDataProcessor(
            save_option=SaveOption.IMAGE,
            calculate_max_row=self.calculate_max_row,
            frame_fixed_number=self.frame_fixed_number,
        )

        extra_dir = "ffn=%s/" % (str(self.frame_fixed_number))
        self._make_directory(extra_dir)

    def do_pipeline(self):
        print("start pipeline")

        for idx, row in tqdm(self.df_qa.iterrows(), total=len(self.df_qa)):
            question_id = str(row["question_id"])
            video_path = row["path_video"]
            ts = row.get("ts", None)
            video_extensions = ["avi", "mp4", "mkv", "webm", "gif"]

            if not os.path.exists(video_path):
                base_video_path, _ = os.path.splitext(video_path)
                for ext in video_extensions:
                    temp_path = f"{base_video_path}.{ext}"
                    if os.path.exists(temp_path):
                        video_path = temp_path
                        self.df_qa.at[idx, "path_video"] = video_path
                        break
                else:
                    print(f"Video not found: {video_path}")
                    self.error_video_name.append(video_path)
                    continue

            if not os.path.exists(self._make_file_path(question_id)):
                try:
                    image_data = self.fps_data_processor.process(video_path, ts)
                    if image_data == -1:
                        print(f"Failed to process video: {video_path}")
                        self.error_video_name.append(video_path)
                        continue
                    answer = self.model.inference(
                        user_prompt=self.func_user_prompt(self.user_prompt, row),
                        raw_image=image_data,
                    )
                    extracted_answer = self.model.extract_answers()
                    self.write_result_file(question_id, extracted_answer)
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")
                    self.error_video_name.append(video_path)
                    continue

        return self.merge_qa_and_answer()

    def write_result_file(self, question_id, answer, extension=".txt"):
        file_path = self._make_file_path(question_id, extension)
        with open(file_path, "w") as file:
            file.write(answer)

    def _make_file_path(self, question_id, extension=".txt"):
        return os.path.join(self.path_result, question_id + extension)

    def _load_qa_file(self):
        try:
            self.df_qa = pd.read_csv(self.path_qa, index_col=0)
        except Exception as e:
            print(e)
            raise Exception("not valid qa files")

    def _make_directory(self, extra_dir):
        self.path_result = os.path.join(self.path_dir, extra_dir)
        os.makedirs(self.path_result, exist_ok=True)

    def merge_qa_and_answer(self):
        print("start merge_qa_and_answer")

        self.df_qa["pred"] = None
        path_merged = os.path.join(self.path_result, "result.csv")

        if not os.path.exists(path_merged):
            for idx, row in self.df_qa.iterrows():
                question_id = str(row["question_id"])
                file_path = self._make_file_path(question_id)
                if os.path.exists(file_path):
                    try:
                        with open(file_path, "r") as file:
                            file_contents = file.read()
                        self.df_qa.loc[idx, "pred"] = file_contents
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                        continue

            self.df_qa.to_csv(path_merged)
        else:
            self.df_qa = pd.read_csv(path_merged, index_col=0)

        return self.df_qa, path_merged