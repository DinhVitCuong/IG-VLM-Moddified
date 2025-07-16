"""
 Copyright (c) 2024, Deep Representation Learning Research Group, Seoul National University.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import sys
import os
import time
from io import BytesIO
import argparse
import re
import uuid
import glob
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from vision_processor.fps_gridview_processor import *
from pipeline_processor.llava_pipeline import *
from evaluation.gpt3_evaluation_utils import *
from enum import Enum

class SaveOption(Enum):
    BASE64 = "base64"
    IMAGE = "image"
    FILE = "file"
    
def infer_and_eval_model(args):
    path_qa = args.path_qa_pair_csv
    path_video = args.path_video
    video_mode = args.video_mode
    path_result_dir = args.path_result
    llm_size = args.llm_size
    api_key = args.api_key

    # Validate video input based on video_mode
    if video_mode == "single":
        if not os.path.isfile(path_video):
            raise FileNotFoundError(f"Video file not found: {path_video}")
        video_paths = [path_video]
    else:  # video_mode == "folder"
        if not os.path.isdir(path_video):
            raise NotADirectoryError(f"Video directory not found: {path_video}")
        video_extensions = ["*.avi", "*.mp4", "*.mkv", "*.gif", "*.webm"]
        video_paths = []
        for ext in video_extensions:
            video_paths.extend(glob.glob(os.path.join(path_video, ext)))
        if not video_paths:
            raise FileNotFoundError(f"No valid video files found in directory: {path_video}")

    model_name, user_prompt = get_llava_and_prompt(llm_size)
    frame_fixed_number = 6

    print("loading [%s]" % (model_name))

    # Process each video
    all_df_merged = []
    for video_path in video_paths:
        print(f"Processing video: {video_path}")
        llavaPipeline = LlavaPipeline(
            model_name,
            path_qa,
            video_path,  # Pass single video path
            dir=path_result_dir,
            video_mode=video_mode,
        )
        llavaPipeline.set_component(
            user_prompt,
            frame_fixed_number=frame_fixed_number,
        )
        df_merged, path_df_merged = llavaPipeline.do_pipeline()
        all_df_merged.append(df_merged)
        print(f"llava prediction result for {video_path}: {path_df_merged}")

    # Combine results if processing multiple videos
    if len(all_df_merged) > 1:
        df_merged = pd.concat(all_df_merged, ignore_index=True)
        path_df_merged = os.path.join(path_result_dir, f"combined_result_{uuid.uuid4()}.csv")
        df_merged.to_csv(path_df_merged)
    elif all_df_merged:
        df_merged = all_df_merged[0]
        path_df_merged = os.path.join(path_result_dir, f"result_{uuid.uuid4()}.csv")
        df_merged.to_csv(path_df_merged)

    print("llava prediction result: " + path_df_merged)
    # print("start gpt3-evaluation")

    # gpt3_dir = os.path.join(path_result_dir, "results_gpt3_evaluation")

    # Correctness Information Evaluation for Text Generation Performance
    # df_qa, path_merged = eval_gpt3(
    #     df_merged, gpt3_dir, api_key, gpt_eval_type=EvaluationType.CORRECTNESS
    # )

    """
    Text Generation Benchmark has five evaluations. CI(Correctness Information), DO(Detailed Orientation), CU(Context Understanding), TU(Temporal Understanding), CO(Consistency) 
    
    In case of DO, Please use the following. 
    # df_qa, path_merged = eval_gpt3(df_merged, gpt3_dir, api_key, gpt_eval_type=EvaluationType.DETAILED_ORIENTATION)
    
    In case of CU, Please use the following. 
    # df_qa, path_merged = eval_gpt3(df_merged, gpt3_dir, api_key, gpt_eval_type=EvaluationType.CONTEXT)
    
    In case of TU, Please use the following. 
    # df_qa, path_merged = eval_gpt3(df_merged, gpt3_dir, api_key, gpt_eval_type=EvaluationType.TEMPORAL)
    
    In case of CO, we need to evaluate llava twice on two questions. Then, Please use the following
    # df_qa, path_merged = eval_gpt3_consistency(df_merged1, df_merged2, gpt3_dir, api_key)
    
    """

def get_llava_and_prompt(llm_size):
    if llm_size in ["7b", "13b"]:
        prompt = "USER: <image>\nThe provided image arranges keyframes from a video in a grid view. Answer concisely with overall content and context of the video, highlighting any significant events, characters, or objects that appear throughout the frames. Question: %s? \nASSISTANT:\nAnswer: In the video,"
        model_name = "llava-hf/llava-1.5-7b-hf"
    else:
        prompt = "<|im_start|>system\nAnswer the question. <|im_end|>\n<|im_start|>user\n<image>\nThe provided image arranges keyframes from a video in a grid view. Answer concisely with overall content and context of the video, highlighting any significant events, characters, or objects that appear throughout the frames. Question: %s? <|im_end|>\n<|im_start|>assistant\nAnswer: In the video,"
        model_name = "llava-hf/llava-1.5-34b-hf"
    return model_name, prompt

def validate_llm_size(type_llm_size):
    if type_llm_size not in {"7b", "13b", "34b"}:
        raise argparse.ArgumentTypeError(f"No valid LLM size.")
    return type_llm_size

def validate_video_path(filename):
    pattern = r"\.(avi|mp4|mkv|gif|webm)$"
    if not re.search(pattern, filename) and not os.path.isdir(filename):
        raise argparse.ArgumentTypeError(
            f"No valid video path. For single mode, include %s and a video extension (e.g., /tmp/%s.mp4). For folder mode, provide a directory."
        )
    return filename

def validate_video_mode(mode):
    if mode not in {"single", "folder"}:
        raise argparse.ArgumentTypeError("video_mode must be 'single' or 'folder'.")
    return mode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA v1.6 with IG-VLM")
    parser.add_argument(
        "--path_qa_pair_csv",
        type=str,
        required=True,
        help="path of question and answer. It should be csv files",
    )
    parser.add_argument(
        "--path_video",
        type=validate_video_path,
        required=True,
        help="path to a single video file (single mode) or a directory of videos (folder mode).",
    )
    parser.add_argument(
        "--video_mode",
        type=validate_video_mode,
        default="single",
        help="Mode to process videos: 'single' for one video file, 'folder' for all videos in a directory.",
    )
    parser.add_argument(
        "--path_result",
        type=str,
        required=True,
        help="path of output directory",
    )
    parser.add_argument(
        "--llm_size",
        type=validate_llm_size,
        default="7b",
        help="You can choose llm size of LLaVA. 7b | 13b | 34b",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="api key for gpt-3 evaluation",
    )
    args = parser.parse_args()

    infer_and_eval_model(args)