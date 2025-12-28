import json
import os
from typing import Any, Dict, List
import math
import torch
from diffusers import DiffusionPipeline
from flagevalmm.common.logger import get_logger
from flagevalmm.models.base_model_adapter import BaseModelAdapter
from flagevalmm.server.utils import parse_args
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
logger = get_logger(__name__)

class ModelAdapter(BaseModelAdapter):
    """Minimal T2I adapter for the Qwen-Image diffusion pipeline."""

    def model_init(self, task_info: Dict) -> None:
        # Decide device/dtype
        # Load pipeline
        assert torch.cuda.is_available(), "CUDA is not available"
        device = "cuda"
        
        self.pipe = DiffusionPipeline.from_pretrained(
            task_info["model_path"], torch_dtype=torch.bfloat16, device_map=device
        )
        self.pipe.set_progress_bar_config(disable=True)
        # Runtime knobs (all optional; fall back to sane defaults)
        model_cfg = task_info.get("model_cfg", {}) or {}
        self.processor_path = model_cfg.get("processor_path", None)
        extra_args = model_cfg.get("extra_args", {}) or {}
        self.cfg_scale = float(extra_args.get("cfg_scale", 4.0))
        self.save_items = bool(extra_args.get("save_items", True))
        resolution = int(extra_args.get("resolution", 1024))
        aspect_ratios = {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1104),
            "3:4": (1104, 1472),
            "3:2": (1584, 1056),
            "2:3": (1056, 1584),
        }
        width, height = aspect_ratios["16:9"]
        self.width = width
        self.height = height
        self.num_timesteps = int(extra_args.get("num_timesteps", 50))
        self.seed = int(extra_args.get("seed", 42))
        # Deterministic seed on CPU by default; users can change if needed.
        self.generator = torch.Generator(device=device).manual_seed(self.seed)

    def _run_single_prompt(self, prompt: str, question_id: str, output_dir: str) -> List[str]:
        """Generate images for one prompt and return saved filenames."""
        results = self.pipe(
            prompt=prompt,
            width=self.width,
            height=self.height,
            num_inference_steps=self.num_timesteps,
            true_cfg_scale=self.cfg_scale,
            generator=self.generator,
        ).images

        return results

    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        task_type = meta_info["type"].lower()
        is_t2i = "t2i" in task_type
        is_i2i = "i2i" in task_type
        is_vqa = "vqa" in task_type
        logger.info(
            f"Running {task_name, meta_info} as "
            f"{'T2I' if is_t2i else ('I2I' if is_i2i else 'VQA')} task"
        )
        if is_i2i:
            self._run_i2i_task(task_name, meta_info)
        elif is_t2i:
            self._run_t2i_task(task_name, meta_info)
        elif is_vqa:
            self._run_vqa_task(task_name, meta_info)
        else:
            raise NotImplementedError(f"Qwen-Image adapter currently supports T2I, I2I and VQA, but got {task_type}")

    def _run_t2i_task(self, task_name: str, meta_info: Dict[str, Any]):
        data_len = meta_info["length"]
        sample_dir = meta_info["output_dir"]
        extra_args = getattr(self, "extra_args", {}) or {}
        print(f"extra_args: {extra_args}")
        save_items = bool(extra_args.get("save_items", True))
        os.makedirs(sample_dir, exist_ok=True)
        os.makedirs(self.get_items_dir(meta_info), exist_ok=True)
        output_info = []
        world_size = (
            self.accelerator.state.num_processes if self.accelerator is not None else 1
        )
        rank = (
            self.accelerator.state.local_process_index
            if self.accelerator is not None
            else 0
        )
        for idx in tqdm(range(rank, data_len, world_size), desc="Running T2I task"):
            data = self.task_manager.get_data(task_name, idx)
            print(f"data: {data}")
            prompt = data.get("prompt") or data.get("question")
            question_id = str(data.get("id") or data.get("question_id") or idx)
            image_list = self._run_single_prompt(prompt, question_id, sample_dir)
            
            sample_dir = os.path.join(sample_dir, "samples")
            os.makedirs(sample_dir, exist_ok=True)
            
            image_names: list[str] = []
            image_paths: list[str] = []
            for i, image in enumerate(image_list):
                image_name = f"{question_id}_{i:05}.png"
                sample_path = os.path.join(sample_dir, image_name)
                image.save(sample_path)
                image_names.append(image_name)
                image_paths.append(sample_path)
            
            out_item_result: dict[str, Any] = {
                "question_id": question_id,
                "id": question_id,
                "prompt": prompt,
                "images": image_names,
            }
            output_info.append(out_item_result)
            if save_items:
                out_item_save: dict[str, Any] = dict(out_item_result)
                # Save think content + absolute image paths in items JSON.
                out_item_save["image_paths"] = image_paths
                self.save_item(
                    out_item_save,
                    question_id=question_id,
                    meta_info=meta_info,
                )
                    # save results for each rank, then gather on main
        if world_size == 1:
            # single-process: keep legacy output name
            self.save_result(output_info, meta_info, rank=None)
            return

        self.save_result(output_info, meta_info, rank=rank)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self.collect_results_and_save(meta_info)
            
    def _run_i2i_task(self, task_name: str, meta_info: Dict[str, Any]):
        pass
    
    def _run_chat(self, model, processor, question, images):
        
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question
                },
            ]
        }]
        for image in images:
            messages[0]["content"].append({
                "type": "image",
                "image": image
            })
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text
        
    def _run_vqa_task(self, task_name: str, meta_info: Dict[str, Any]):
        assert self.processor_path is not None, "processor_path is not set"
        extra_args = getattr(self, "extra_args", {}) or {}
        save_items = bool(extra_args.get("save_items", True))
        processor = AutoProcessor.from_pretrained(self.processor_path)
        model = self.pipe.text_encoder
        data_len = meta_info["length"]
        output_dir = meta_info["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.get_items_dir(meta_info), exist_ok=True)
        output_info = []
        world_size = (
            self.accelerator.state.num_processes if self.accelerator is not None else 1
        )
        rank = (
            self.accelerator.state.local_process_index
            if self.accelerator is not None
            else 0
        )
        
        for idx in tqdm(range(rank, data_len, world_size), desc="Running VQA task"):
            data = self.task_manager.get_data(task_name, idx)
            question = data.get("question")
            images = data.get("images")
            question_id = str(data.get("id") or data.get("question_id") or idx)
            output_text = self._run_chat(model, processor, question, images)
            output_info.append(
                {
                    "question_id": question_id,
                    "prompt": question,
                    "answer": output_text
                }
            )
            if save_items:
                self.save_item(
                    output_info[-1],
                    question_id=question_id,
                    meta_info=meta_info,
                )
        rank = self.accelerator.state.local_process_index
        self.save_result(output_info, meta_info, rank=rank)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self.collect_results_and_save(meta_info)
            
if __name__ == "__main__":
    args = parse_args()
    model_adapter = ModelAdapter(
        extra_cfg=args.cfg,
        task_names=None,
    )
    model_adapter.run()

