"""
 Copyright (c) 2024, Deep Representation Learning Research Group, Seoul National University.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

# from .base_model_inference import *
# import math
# import re
# from io import BytesIO
# import torch
# import torch.nn.functional as F
# from llava.model.builder import load_pretrained_model
# from llava.constants import (
#     IMAGE_TOKEN_INDEX,
#     DEFAULT_IM_START_TOKEN,
#     DEFAULT_IM_END_TOKEN,
#     IMAGE_PLACEHOLDER,
# )
# from llava.conversation import conv_templates
# from llava.mm_utils import (
#     process_images,
#     tokenizer_image_token,
#     get_model_name_from_path,
# )

from .base_model_inference import *
import math
import re
from io import BytesIO
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor, AutoModelForCausalLM, LlavaOnevisionForConditionalGeneration


class Llava157BProcessor:
    def __init__(self, model_name, local_save_path=""):
        self.model_name = model_name
        self.local_save_path = local_save_path
        self.processor = None
        self.model = None
        self.raw_image = None
        self.user_prompt = None
        self.max_new_tokens = None
        self.do_sample = None
        self.temperature = None
        self.result_rext = None

    def load_model(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name, use_fast=False)  # Changed to use_fast=False to avoid do_pad
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map="cuda",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

    def inference(self, *args, **kwargs):
        self._extract_arguments(**kwargs)
        # print("[DEBUG] kwargs:", kwargs)
        # print("[DEBUG] START INFERENCE")
        # print(f"Type of raw_image: {type(self.raw_image)}")
        # print(f"User prompt: {self.user_prompt}")
        # self.raw_image = Image.open("/workspace/IG-VLM/example/imagegrid_sample/L01_V001_480p_sub1.jpg")
        inputs = self.processor(
            images=self.raw_image,
            text=self.user_prompt,
            return_tensors="pt",
            padding=False
        ).to(self.model.device, dtype=torch.float16)
        # print("[DEBUG] INPUT PROCESSED")
        # print(f"Processor inputs: {inputs}")
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                do_sample=self.do_sample,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
        # print("[DEBUG] OUTPUT PROCESSED")
        self.result_rext = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()

    def extract_answers(self):
        return self.result_rext.split("ASSISTANT:")[-1]

    def _extract_arguments(self, **kwargs):
        self.user_prompt = kwargs["user_prompt"]
        self.raw_image = kwargs["raw_image"]
        self.max_new_tokens = kwargs.get("max_new_tokens", 300)
        self.do_sample = kwargs.get("do_sample", False)
        self.temperature = kwargs.get("temperature", 1)

class Llava16Vicuna7BProcessor:
    def __init__(self, model_name, local_save_path=""):
        self.model_name = model_name
        self.local_save_path = local_save_path
        self.processor = None
        self.model = None
        self.raw_image = None
        self.user_prompt = None
        self.max_new_tokens = None
        self.do_sample = None
        self.temperature = None
        self.result_rext = None

    def load_model(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name, use_fast=True)  
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="cuda",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

    def inference(self, *args, **kwargs):
        self._extract_arguments(**kwargs)
        # print("[DEBUG] kwargs:", kwargs)
        # print("[DEBUG] START INFERENCE")
        # print(f"Type of raw_image: {type(self.raw_image)}")
        # print(f"User prompt: {self.user_prompt}")
        # self.raw_image = Image.open("/workspace/IG-VLM/example/imagegrid_sample/L01_V001_480p_sub1.jpg")
        inputs = self.processor(
            images=self.raw_image,
            text=self.user_prompt,
            return_tensors="pt",
            padding=False
        ).to(self.model.device, dtype=torch.float16)
        # print("[DEBUG] INPUT PROCESSED")
        # print(f"Processor inputs: {inputs}")
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                do_sample=self.do_sample,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
        # print("[DEBUG] OUTPUT PROCESSED")
        self.result_rext = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()

    def extract_answers(self):
        return self.result_rext.split("ASSISTANT:")[-1]

    def _extract_arguments(self, **kwargs):
        self.user_prompt = kwargs["user_prompt"]
        self.raw_image = kwargs["raw_image"]
        self.max_new_tokens = kwargs.get("max_new_tokens", 300)
        self.do_sample = kwargs.get("do_sample", False)
        self.temperature = kwargs.get("temperature", 1)

class LlavaNeXTVideo7BProcessor:
    def __init__(self, model_name, local_save_path=""):
        self.model_name = model_name
        self.local_save_path = local_save_path
        self.processor = None
        self.model = None
        self.question = None
        self.raw_video = None  
        self.max_new_tokens = None
        self.do_sample = None
        self.temperature = None
        self.result_text = None

    def load_model(self):
        self.processor = LlavaNextVideoProcessor.from_pretrained(self.model_name)
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map="cuda",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

    def inference(self, *args, **kwargs):
        self._extract_arguments(**kwargs)
        # self.question = question
        # self.raw_video = raw_video
        # self.max_new_tokens = max_new_tokens
        # self.do_sample = do_sample
        # self.temperature = temperature

        # Construct conversation
        # conversation = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "text", "text": self.question},
        #             {"type": "video"},
        #         ],
        #     }
        # ]
        # prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(
            text=self.user_prompt,
            videos=self.raw_video,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                use_cache=True,
            )
        self.result_text = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()

    def extract_answers(self):
        parts = self.result_text.split("ASSISTANT:")
        if len(parts) > 1:
            answer = parts[-1].strip()
            if "<|im_end|>" in answer:
                answer = answer.split("<|im_end|>")[0].strip()
            return answer
        else:
            return self.result_text

    def _extract_arguments(self, **kwargs):
        self.user_prompt = kwargs["user_prompt"]
        self.raw_video = kwargs["raw_video"]
        self.max_new_tokens = kwargs.get("max_new_tokens", 300)
        self.do_sample = kwargs.get("do_sample", False)
        self.temperature = kwargs.get("temperature", 1)


class Llava_Onevision_QWEN2_05BProcessor:
    def __init__(self, model_name, local_save_path=""):
        self.model_name = model_name
        self.local_save_path = local_save_path
        self.processor = None
        self.model = None
        self.raw_image = None
        self.user_prompt = None
        self.max_new_tokens = None
        self.do_sample = None
        self.temperature = None
        self.result_rext = None

    def load_model(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name, use_fast=False)  # Changed to use_fast=False to avoid do_pad
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map="cuda",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self.model.config.pad_token_id = self.model.config.eos_token_id

    def inference(self, *args, **kwargs):
        self._extract_arguments(**kwargs)
        # print("[DEBUG] kwargs:", kwargs)
        # print("[DEBUG] START INFERENCE")
        # print(f"Type of raw_image: {type(self.raw_image)}")
        # print(f"User prompt: {self.user_prompt}")
        # self.raw_image = Image.open("/workspace/IG-VLM/example/imagegrid_sample/L01_V001_480p_sub1.jpg")
        inputs = self.processor(
            images=self.raw_image,
            text=self.user_prompt,
            return_tensors="pt",
            padding=False
        ).to(self.model.device, dtype=torch.float16)
        # print("[DEBUG] INPUT PROCESSED")
        # print(f"Processor inputs: {inputs}")
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                do_sample=self.do_sample,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                pad_token_id=self.model.config.eos_token_id
            )
        # print("[DEBUG] OUTPUT PROCESSED")
        self.result_rext = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()

    def extract_answers(self):
        return self.result_rext.split("ASSISTANT:")[-1]

    def _extract_arguments(self, **kwargs):
        self.user_prompt = kwargs["user_prompt"]
        self.raw_image = kwargs["raw_image"]
        self.max_new_tokens = kwargs.get("max_new_tokens", 300)
        self.do_sample = kwargs.get("do_sample", False)
        self.temperature = kwargs.get("temperature", 1)
