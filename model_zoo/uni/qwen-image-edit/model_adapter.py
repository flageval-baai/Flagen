import json
import os
from typing import Any, Dict, List
import math
import torch
from diffusers import QwenImageEditPlusPipeline
from flagevalmm.common.logger import get_logger
from flagevalmm.models.base_model_adapter import BaseModelAdapter
from flagevalmm.server.utils import parse_args, load_pil_image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from PIL import Image
logger = get_logger(__name__)

class ModelAdapter(BaseModelAdapter):
    """Qwen-Image-Edit adapter for image editing (I2I) and text-to-image (T2I) tasks."""

    def model_init(self, task_info: Dict) -> None:
        # Decide device/dtype
        # Load pipeline
        assert torch.cuda.is_available(), "CUDA is not available"
        device = "cuda"
        
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            task_info["model_path"], torch_dtype=torch.bfloat16
        )
        self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)
        # Runtime knobs (all optional; fall back to sane defaults)
        model_cfg = task_info.get("model_cfg", {}) or {}
        self.processor_path = model_cfg.get("processor_path", None)
        extra_args = model_cfg.get("extra_args", {}) or {}
        self.extra_args = extra_args
        self.cfg_scale = float(extra_args.get("cfg_scale", 4.0))
        self.save_items = bool(extra_args.get("save_items", True))
        self.num_timesteps = int(extra_args.get("num_timesteps", 50))
        self.seed = int(extra_args.get("seed", 42))
        self.guidance_scale = float(extra_args.get("guidance_scale", 1.0))
        self.num_images_per_prompt = int(extra_args.get("num_images_per_prompt", 1))


    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        task_type = meta_info["type"].lower()
        is_i2i = 'i2i' in task_type
        logger.info(
            f"Running {task_name, meta_info} as "
            f"{'I2I'} task"
        )
        if is_i2i:
            self._run_i2i_task(task_name, meta_info)
        else:
            raise NotImplementedError(f"Qwen-Image-Edit adapter currently supports I2I, but got {task_type}")
        
    def _run_single_image_editing(self, prompt: str, images: List[Image.Image]) -> List[Image.Image]:
        """Run single image editing task."""
        inputs = {
            "image": images,
            "prompt": prompt,
            "generator": torch.manual_seed(self.seed),
            "true_cfg_scale": self.cfg_scale,
            "negative_prompt": " ",
            "num_inference_steps": self.num_timesteps,
            "guidance_scale": self.guidance_scale,
            "num_images_per_prompt": self.num_images_per_prompt,
        }
        with torch.inference_mode():
            output = self.pipe(**inputs)
            output_image = output.images[0]
        return output_image
            
    def _run_i2i_task(self, task_name: str, meta_info: Dict[str, Any]):
        """Run image-to-image editing task."""
        data_len = meta_info["length"]
        output_dir = meta_info["output_dir"]
        extra_args = getattr(self, "extra_args", {}) or {}
        save_items = bool(extra_args.get("save_items", True))
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.get_items_dir(meta_info), exist_ok=True)
        
        output_info: list[dict[str, Any]] = []
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
            prompt = data.get("prompt") or data.get("question")
            question_id = str(data.get("id") or data.get("question_id") or idx)
            
            source_path = (
                data.get("source_path")
                or data.get("source")
                or data.get("img_path")
                or data.get("image")
            )
            if not source_path:
                raise KeyError(f"Missing source image path for sample {question_id}")
            source_images, _ = load_pil_image(
                [source_path], img_idx=[0], reqiures_img=True, reduplicate=False
            )
            
            edited_image = self._run_single_image_editing(prompt, source_images)
            
            sample_dir = os.path.join(output_dir, "samples")
            os.makedirs(sample_dir, exist_ok=True)
            image_name = f"{question_id}_00000.png"
            sample_path = os.path.join(sample_dir, image_name)
            edited_image.save(sample_path)
            
            output_info.append(
                {
                    "question_id": question_id,
                    "prompt": prompt,
                    "images": [image_name],
                    "source_path": source_path,
                    "image_paths": [sample_path],
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



