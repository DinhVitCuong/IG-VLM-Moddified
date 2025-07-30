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
from transformers import AutoTokenizer, AutoModel
import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def infer_model_1(model, video_path, question=None, ffn=6):


    pixel_values, num_patches_list = load_video(video_path, num_segments=15, max_num=1)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    prompt = video_prefix + f'{question}'

    start_time = HEHE.time()
    # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{prompt}
    response, history = model.chat(tokenizer, pixel_values, prompt, generation_config,
                                num_patches_list=num_patches_list, history=None, return_history=True)
    print(f'User: {prompt}\nAssistant: {response}')

    end_time = HEHE.time()
    response_time = end_time - start_time

    answer = response
    print(answer)

    return answer, response_time


def infer_model_2(model, video_path, question=None, ffn=6):


    pixel_values, num_patches_list = load_video(video_path, num_segments=15, max_num=1)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])

    prompt = f'{question}'
    start_time = HEHE.time()
    response, history = model.chat(tokenizer, pixel_values, prompt, generation_config,
                                num_patches_list=num_patches_list, history=history, return_history=True)
    print(f'User: {prompt}\nAssistant: {response}')

    end_time = HEHE.time()
    response_time = end_time - start_time

    answer = response
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
    path = 'OpenGVLab/InternVL3-1B'
    device_map = split_model('OpenGVLab/InternVL3-1B')
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        # use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    generation_config = dict(max_new_tokens=1024, do_sample=True)


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
        
        results_1 = []
        results_2 = []
        for key, entry in bench_data.items():
            video_path = entry["video_path"]
            question = entry["question"]
            answer_1, time_1 = infer_model_1(model, video_path, question=question)
            answer_2, time_2 = infer_model_1(model, video_path, question=question)
            results_1.append({
                "id": key,
                "video_path": video_path,
                "question": question,
                "answer": answer_1 if answer_1 is not None else "Failed",
                "time": time_1 if time_1 is not None else "N/A"
            })
            results_2.append({
                "id": key,
                "video_path": video_path,
                "question": question,
                "answer": answer_2 if answer_2 is not None else "Failed",
                "time": time_2 if time_2 is not None else "N/A"
            })
        
        # Save results
        output_file = os.path.join(args.path_result, "bench_results_InternVL_1B_sequence_frame.json")
        with open(output_file, 'w') as f:
            json.dump(results_1, f, indent=4)
        output_file = os.path.join(args.path_result, "bench_results_InternVL_1B_video.json")
        with open(output_file, 'w') as f:
            json.dump(results_2, f, indent=4)
        print(f"Bench results saved to {output_file}")
    elif args.video_path:
        # Interactive mode
        infer_model_1(args.video_path, args.llm_size, args.path_result)
    else:
        parser.error("Either --video_path or --bench_json must be provided")