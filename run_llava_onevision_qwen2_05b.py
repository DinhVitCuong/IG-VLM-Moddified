"""
 Copyright (c) 2024, Deep Representation Learning Research Group, Seoul National University.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
import argparse
import torch
import re

from pipeline_processor.llava2_pipeline import Llava_Onevision_QWEN2_05BPipeline

def infer_model(video_path, llm_size, path_result_dir):
    # Clear GPU memory to prevent CUDA out of memory
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Validate video path
    if not os.path.isfile(video_path) or not re.search(r"\.(avi|mp4|mkv|gif|webm)$", video_path):
        raise FileNotFoundError(f"Invalid video file: {video_path}")

    # Get model name and prompt
    model_name, user_prompt = get_llava_and_prompt(llm_size)
    print(f"loading [{model_name}]")

    # Initialize LlavaPipeline
    llava_pipeline = Llava_Onevision_QWEN2_05BPipeline(
        model_name=model_name,
        path_qa=None,
        path_video_file_format=video_path,
        dir=path_result_dir,
        video_mode="single",
    )

    # Set pipeline components
    llava_pipeline.set_component(
        user_prompt=user_prompt,
        frame_fixed_number=6,
        func_user_prompt=lambda prompt, row: prompt % row,
    )

    # Interactive loop for user questions
    print("Enter your question about the video (press Ctrl+C to exit):")
    while True:
        try:
            qa_text = input("Question: ").strip()
            if not qa_text:
                print("Please enter a valid question.")
                continue

            # Process video and get result with response time
            answer, result_file_path, response_time = llava_pipeline.do_pipeline(qa_text=qa_text)
            if answer is not None:
                print(f"Answer: {answer}")
                print(f"Response time: {response_time:.2f} seconds")
                print(f"Prediction result saved at: {result_file_path}")
                # Append response time to the .txt file
                with open(result_file_path, "a") as file:
                    file.write(f"\nResponse time: {response_time:.2f} seconds")
            else:
                print("Failed to process video or generate answer")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error processing question: {e}")
            continue

def get_llava_and_prompt(llm_size):
    if llm_size in ["7b", "13b"]:
        prompt = "USER: <image>\nThe provided image arranges keyframes from a video in a grid view. Answer concisely with overall content and context of the video, highlighting any significant events, characters, or objects that appear throughout the frames. Question: %s? \nASSISTANT:\nAnswer: In the video,"
        model_name = f"llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    else:
        prompt = "<|im_start|>system\nAnswer the question. <|im_end|>\n<|im_start|>user\n<image>\nThe provided image arranges keyframes from a video in a grid view. Answer concisely with overall content and context of the video, highlighting any significant events, characters, or objects that appear throughout the frames. Question: %s? <|im_end|>\n<|im_start|>assistant\nAnswer: In the video,"
        model_name = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    return model_name, prompt

def validate_llm_size(type_llm_size):
    if type_llm_size not in {"7b", "13b", "34b"}:
        raise argparse.ArgumentTypeError(f"No valid LLM size. Choose 7b, 13b, or 34b.")
    return type_llm_size

def validate_video_path(filename):
    if not os.path.isfile(filename) or not re.search(r"\.(avi|mp4|mkv|gif|webm)$", filename):
        raise argparse.ArgumentTypeError(f"Invalid video file. Must be a valid video file (avi, mp4, mkv, gif, webm).")
    return filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA v1.6 for video question answering")
    parser.add_argument(
        "--video_path",
        type=validate_video_path,
        required=True,
        help="Path to the video file (e.g., /path/to/video.mp4)",
    )
    parser.add_argument(
        "--path_result",
        type=str,
        required=True,
        help="Path to output directory for results",
    )
    parser.add_argument(
        "--llm_size",
        type=validate_llm_size,
        default="7b",
        help="LLaVA model size: 7b, 13b, or 34b",
    )
    args = parser.parse_args()

    infer_model(args.video_path, args.llm_size, args.path_result)