"""
 Copyright (c) 2024, Deep Representation Learning Research Group, Seoul National University.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
import time as HEHE
import argparse
import torch
import re
import json

# from pipeline_processor.llava2_pipeline import Llava_Onevision_QWEN2_05BPipeline
from transformers import AutoProcessor, AutoModelForImageTextToText

def infer_model(processor, model, video_path, path_result_dir, question=None, ffn=6):
    # Clear GPU memory to prevent CUDA out of memory
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Validate video path
    if not os.path.isfile(video_path) or not re.search(r"\.(avi|mp4|mkv|gif|webm)$", video_path):
        raise FileNotFoundError(f"Invalid video file: {video_path}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "path": f"{video_path}"},
                {"type": "text", "text": f"{question}"}
            ]
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    start_time = HEHE.time()

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    end_time = HEHE.time()
    response_time = end_time - start_time

    answer = generated_texts[0]
    print(answer)

    return answer, response_time

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
    model_path = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        # _attn_implementation="flash_attention_2"
    ).to("cuda")
    print("MODEL LOADED! ")
    parser = argparse.ArgumentParser(description="LLaVA v1.6 for video question answering")
    parser.add_argument(
        "--video_path",
        type=validate_video_path,
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
    parser.add_argument(
        "--bench_json",
        type=str,
        help="Path to bench.json file for batch processing",
    )
    args = parser.parse_args()

    if args.bench_json:
        # Bench mode
        with open(args.bench_json, 'r') as f:
            bench_data = json.load(f)
        
        results = []
        for key, entry in bench_data.items():
            video_path = entry["video_path"]
            question = entry["question"]
            answer, time = infer_model(processor, model, video_path, args.path_result, question=question)
            results.append({
                "id": key,
                "video_path": video_path,
                "question": question,
                "answer": answer.split("\nAssistant:")[1] if answer is not None else "Failed",
                "time": time if time is not None else "N/A"
            })
        
        # Save results
        output_file = os.path.join(args.path_result, "bench_results_SmolVLM2_05B.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Bench results saved to {output_file}")
    elif args.video_path:
        # Interactive mode
        infer_model(args.video_path, args.llm_size, args.path_result)
    else:
        parser.error("Either --video_path or --bench_json must be provided")