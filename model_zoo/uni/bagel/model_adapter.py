from typing import Dict, Any
from contextlib import nullcontext
import copy

from flagevalmm.models.base_model_adapter import BaseModelAdapter
from modeling.utils import (
    load_model_and_tokenizer,
    load_gen_model_and_tokenizer,
    build_transform,
    process_conversation,
    ImageTransform,
)
import torch
from flagevalmm.common.logger import get_logger
from flagevalmm.server import ServerDataset
from flagevalmm.server.utils import (
    process_images_symbol,
    load_pil_image,
    default_collate_fn,
    parse_args,
    RunCfg,
)
from modeling.bagel.qwen2_navit import NaiveCache
from PIL import Image
import os
import time

logger = get_logger(__name__)

SYSTEM_PROMPT_WITH_THINK = (
    "You should first think about the planning process in the mind and then generate the image. \n"
    "The planning process is enclosed within <think> </think> tags, i.e. "
    "<think> planning process here </think> image here"
)


class BagelDataset(ServerDataset):
    """Load images as PIL and keep question text aligned with placeholders."""

    def __getitem__(self, index):
        data = self.get_data(index)
        question_id = data["question_id"]
        question = data["question"]
        img_paths = data["img_path"]
        if isinstance(img_paths, str):
            img_paths = [img_paths]

        # Extract referenced image indices (if any) to keep order consistent
        question, img_idx = process_images_symbol(question)
        images, _ = load_pil_image(
            img_paths, img_idx, reqiures_img=True, reduplicate=False
        )

        return question_id, question, images


class ModelAdapter(BaseModelAdapter):
    def preprocess_item_for_save(
        self, item: Dict[str, Any], question_id: str, meta_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Save absolute paths for traceability (keep `images` relative for evaluators).
        sample_dir = os.path.join(meta_info["output_dir"], "samples")
        images = item.get("images")
        if images and "image_paths" not in item:
            if isinstance(images, str):
                images = [images]
            image_paths: list[str] = []
            for name in images:
                if isinstance(name, str) and os.path.isabs(name):
                    image_paths.append(name)
                else:
                    image_paths.append(os.path.join(sample_dir, str(name)))
            item["image_paths"] = image_paths

        # Ensure we don't store legacy file-list fields.
        item.pop("think_files", None)
        return item

    def log_memory_usage(self, stage: str):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            logger.info(
                f"[{stage}] Memory: Alloc={allocated:.2f}GB, Rsrv={reserved:.2f}GB, MaxAlloc={max_allocated:.2f}GB"
            )

    def model_init(self, task_info: Dict) -> None:
        ckpt_path = task_info["model_path"]
        extra_args = task_info.get("extra_args", {}) or {}
        self.extra_args: Dict[str, Any] = extra_args
        use_gen_model = bool(extra_args.get("use_gen_model", True))
        max_latent_size = int(extra_args.get("max_latent_size", 64))

        torch.set_grad_enabled(False)
        device = (
            self.accelerator.device
            if getattr(self, "accelerator", None) is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"use_gen_model: {use_gen_model}")
        if use_gen_model:
            model, self.tokenizer, self.new_token_ids, vae_model = (
                load_gen_model_and_tokenizer(
                    ckpt_path, max_latent_size=max_latent_size
                )
            )
            print("Successfully loaded generation model")
            # Keep generation components available for T2I.
            self.vae_model = vae_model.to(
                device=device, dtype=torch.bfloat16
            ).eval()
            self.gen_model = model
            self._t2i_initialized = True
        else:
            model, self.tokenizer, self.new_token_ids = load_model_and_tokenizer(
                ckpt_path
            )
        model = model.to(device=device, dtype=torch.bfloat16).eval()
        if getattr(self, "accelerator", None) is not None:
            model = self.accelerator.prepare_model(model, evaluation_mode=True)
            if hasattr(model, "module"):
                model = model.module
        self.model = model
        self.image_transform = build_transform()
        self.vae_transform = ImageTransform(1024, 512, 16)
        self.vit_transform = ImageTransform(980, 378, 14)

    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        # Determine task type from meta info
        task_type = meta_info["type"].lower()
        is_t2i = "t2i" in task_type
        is_i2i = "i2i" in task_type
        logger.info(
            f"Running {task_name, meta_info} as "
            f"{'T2I' if is_t2i else ('I2I' if is_i2i else 'VQA')} task"
        )
        self.dataset = ServerDataset(
            task_name,
            task_type=meta_info["type"],
            task_manager=self.task_manager,
        )
        if is_t2i:
            self._run_t2i_task(task_name, meta_info)
        elif is_i2i:
            self._run_i2i_task(task_name, meta_info)
        else:
            self._run_vqa_task(task_name, meta_info)

    def _finalize_task_results(self, output_info, meta_info) -> int | None:
        """
        Save rank-specific results, sync processes, and collect all results on main process.
        Returns the count of collected results if on main process, otherwise None.
        """
        world_size = (
            self.accelerator.state.num_processes if self.accelerator is not None else 1
        )
        rank = (
            self.accelerator.state.local_process_index
            if self.accelerator is not None
            else 0
        )

        if world_size == 1:
            self.save_result(output_info, meta_info, rank=None)
            return len(output_info)

        self.save_result(output_info, meta_info, rank=rank)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                return self.collect_results_and_save(meta_info)
        return None

    def _run_second_round_gen(
        self,
        past_key_values,
        past_key_values_prompt,
        curr_kvlens: list[int],
        curr_rope: list[int],
        prompt: str,
        image: Image.Image,
        resolution: int,
        extra_args: Dict[str, Any],
        device,
        original_prompt: str,
        first_round_think: str | None = None,
    ):
        """
        Run a second round of generation with I2I logic.
        1. Update cache with prompt (reflection).
        2. Generate 'think/plan'.
        3. Generate image.
        """
        # Unpack args
        num_timesteps = int(extra_args.get("num_timesteps", 50))
        cfg_text_scale = float(extra_args.get("cfg_text_scale", 4.0))
        cfg_img_scale = float(extra_args.get("cfg_img_scale", 1.5))
        cfg_type = str(extra_args.get("cfg_type", "serial_text_img"))
        cfg_interval = extra_args.get("cfg_interval", [0.4, 1.0])
        cfg_renorm_min = float(extra_args.get("cfg_renorm_min", 0.0))
        cfg_renorm_type = str(extra_args.get("cfg_renorm_type", "text_channel"))
        timestep_shift = float(extra_args.get("timestep_shift", 3.0))
        think_max_length = int(extra_args.get("think_max_length", 2048))
        think_temperature = float(extra_args.get("think_temperature", 0.3))
        think_simple = bool(extra_args.get("think_simple", False))
        use_vit = bool(extra_args.get("use_vit", False))
        generate_reflection_image = bool(extra_args.get("generate_reflection_image", False))
        print(f'generate_reflection_image: {generate_reflection_image}')
        
        # 1. Update cache with reflection prompt
        past_key_values, curr_kvlens, curr_rope = self._update_text_cache_with_prompts(
            past_key_values, [prompt], curr_kvlens, curr_rope, device
        )
        past_key_values_prompt += prompt

        if use_vit:
            vit_gen_input, curr_kvlens, curr_rope = self.model.prepare_vit_images(
                curr_kvlens=curr_kvlens,
                curr_rope=curr_rope,
                images=[image],
                transforms=self.vit_transform,
                new_token_ids=self.new_token_ids,
            )
            vit_gen_input = self._move_generation_input_to_device(
                vit_gen_input, device
            )
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past_key_values = self.model.forward_cache_update_vit(
                    past_key_values, **vit_gen_input
                )
            past_key_values_prompt += '<image>'

        # 2. Generate Think/Plan
        (
            past_key_values,
            curr_kvlens,
            curr_rope,
            think_output,
        ) = self._generate_think_and_update_cache(
            past_key_values,
            curr_kvlens,
            curr_rope,
            device,
            think_max_length,
            think_temperature,
            think_simple,
        )
        past_key_values_prompt += think_output
        print(f'past_key_values_prompt: {past_key_values_prompt}')
        
        if not generate_reflection_image:
            return None, think_output
        
        # 3. Construct CFG Caches
        
        # A. cfg_text (Image Only - 1st Round Image)
        cfg_text_past_key_values = NaiveCache(
            self.gen_model.config.llm_config.num_hidden_layers
        )
        cfg_text_curr_kvlens = [0]
        cfg_text_curr_rope = [0]
        
        # Update with VAE
        generation_input_vae, cfg_text_curr_kvlens, cfg_text_curr_rope = self.model.prepare_vae_images(
            curr_kvlens=cfg_text_curr_kvlens,
            curr_rope=cfg_text_curr_rope,
            images=[image],
            transforms=self.vae_transform,
            new_token_ids=self.new_token_ids,
            timestep=0.0,
        )
        generation_input_vae = self._move_generation_input_to_device(
            generation_input_vae, device
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            cfg_text_past_key_values = self.model.forward_cache_update_vae(
                self.vae_model, cfg_text_past_key_values, **generation_input_vae
            )
            
        # Update with ViT (if use_vit)
        if use_vit:
            generation_input_vit, cfg_text_curr_kvlens, cfg_text_curr_rope = self.model.prepare_vit_images(
                curr_kvlens=cfg_text_curr_kvlens,
                curr_rope=cfg_text_curr_rope,
                images=[image],
                transforms=self.vit_transform,
                new_token_ids=self.new_token_ids,
            )
            generation_input_vit = self._move_generation_input_to_device(
                generation_input_vit, device
            )
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                cfg_text_past_key_values = self.model.forward_cache_update_vit(
                    cfg_text_past_key_values, **generation_input_vit
                )
        
        # B. cfg_img (Text Only)
        # Sequence: [SYSTEM_PROMPT, prompt, think_output, reflection, reflection_output]
        cfg_img_past_key_values = NaiveCache(
             self.gen_model.config.llm_config.num_hidden_layers
        )
        cfg_img_curr_kvlens = [0]
        cfg_img_curr_rope = [0]
        
        cfg_img_texts = [SYSTEM_PROMPT_WITH_THINK, original_prompt]
        if first_round_think:
            cfg_img_texts.append(first_round_think)
        cfg_img_texts.append(prompt) # Reflection prompt
        cfg_img_texts.append(think_output) # Reflection output
        
        for text in cfg_img_texts:
             cfg_img_past_key_values, cfg_img_curr_kvlens, cfg_img_curr_rope = self._update_text_cache_with_prompts(
                cfg_img_past_key_values, [text], cfg_img_curr_kvlens, cfg_img_curr_rope, device
             )

        # 4. Generate Image
        # Prepare Main Input (Latent)
        generation_input = self.gen_model.prepare_vae_latent(
            curr_kvlens=curr_kvlens,
            curr_rope=curr_rope,
            image_sizes=[(resolution, resolution)],
            new_token_ids=self.new_token_ids,
        )
        generation_input = self._move_generation_input_to_device(
            generation_input, device
        )
        
        # Prepare CFG Inputs
        # cfg_text (Image)
        generation_input_cfg_text = self.gen_model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_text_curr_kvlens,
            curr_rope=cfg_text_curr_rope,
            image_sizes=[(resolution, resolution)],
        )
        cfg_text_args = {
            "cfg_text_packed_position_ids": generation_input_cfg_text["cfg_packed_position_ids"],
            "cfg_text_packed_query_indexes": generation_input_cfg_text["cfg_packed_query_indexes"],
            "cfg_text_key_values_lens": generation_input_cfg_text["cfg_key_values_lens"],
            "cfg_text_packed_key_value_indexes": generation_input_cfg_text["cfg_packed_key_value_indexes"],
        }
        cfg_text_args = self._move_generation_input_to_device(cfg_text_args, device)
        
        # cfg_img (Text)
        generation_input_cfg_img = self.gen_model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_img_curr_kvlens,
            curr_rope=cfg_img_curr_rope,
            image_sizes=[(resolution, resolution)],
        )
        cfg_img_args = {
            "cfg_img_packed_position_ids": generation_input_cfg_img["cfg_packed_position_ids"],
            "cfg_img_packed_query_indexes": generation_input_cfg_img["cfg_packed_query_indexes"],
            "cfg_img_key_values_lens": generation_input_cfg_img["cfg_key_values_lens"],
            "cfg_img_packed_key_value_indexes": generation_input_cfg_img["cfg_packed_key_value_indexes"],
        }
        cfg_img_args = self._move_generation_input_to_device(cfg_img_args, device)
        
        cfg_args = {}
        cfg_args.update(cfg_text_args)
        cfg_args.update(cfg_img_args)
        
        kwargs = {
            "cfg_text_past_key_values": cfg_text_past_key_values,
            "cfg_img_past_key_values": cfg_img_past_key_values,
            "cfg_type": cfg_type,
            "cfg_renorm_type": cfg_renorm_type,
        }
        
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                unpacked_latent = self.model.generate_image(
                    past_key_values=past_key_values,
                    num_timesteps=num_timesteps,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=cfg_interval,
                    cfg_renorm_min=cfg_renorm_min,
                    timestep_shift=timestep_shift,
                    **generation_input,
                    **cfg_args,
                    **kwargs
                )

        decoded = self._decode_latents(unpacked_latent, resolution, resolution, device)
        return decoded[0], think_output

    def _run_gen_task(self, task_name: str, meta_info: Dict[str, Any], mode: str = "t2i"):
        if not hasattr(self, "gen_model") or not hasattr(self, "vae_model"):
            raise RuntimeError(
                f"{mode.upper()} requires generation components, but they are not initialized. "
                "Please set `model.use_gen_model=true` in runtime config."
            )
        text_num = meta_info["length"]
        output_dir = meta_info["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.get_items_dir(meta_info), exist_ok=True)

        # Sampling knobs
        extra_args = getattr(self, "extra_args", {}) or {}
        print(f"extra_args: {extra_args}")
        save_items = bool(extra_args.get("save_items", True))
        # If True, ignore existing per-item cache under output_dir/items and regenerate.
        # Useful when switching datasets (e.g. enabling `reflection`) or changing generation flags.
        force_regen = bool(extra_args.get("force_regen", False))
        
        # T2I specific args
        num_images = int(extra_args.get("num_images", 1))
        batch_size = int(extra_args.get("batch_size", 1))
        cfg_text_scale = float(extra_args.get("cfg_text_scale", 4.0))
        cfg_img_scale = float(extra_args.get("cfg_img_scale", 1.5))
        resolution = int(extra_args.get("resolution", 1024))
        num_timesteps = int(extra_args.get("num_timesteps", 50))
        cfg_interval = extra_args.get("cfg_interval", [0.4, 1.0])
        cfg_renorm_min = float(extra_args.get("cfg_renorm_min", 0.0))
        timestep_shift = float(extra_args.get("timestep_shift", 3.0))
        think = bool(extra_args.get("think", False))
        think_simple = bool(extra_args.get("think_simple", False))
        think_max_length = int(extra_args.get("think_max_length", 2048))
        think_temperature = float(extra_args.get("think_temperature", 0.3))
        think_system_prompt = str(
            extra_args.get("think_system_prompt", SYSTEM_PROMPT_WITH_THINK)
        )

        output_info: list[dict[str, Any]] = []
        world_size = (
            self.accelerator.state.num_processes if self.accelerator is not None else 1
        )
        rank = (
            self.accelerator.state.local_process_index
            if self.accelerator is not None
            else 0
        )

        for idx in range(rank, text_num, world_size):
            data = self.task_manager.get_data(task_name, idx)
            prompt = data.get("prompt") or data.get("question")
            question_id = str(data.get("id") or data.get("question_id") or idx)
            
            # Source path check for I2I
            source_path = None
            if mode == "i2i":
                source_path = (
                    data.get("source_path")
                    or data.get("source")
                    or data.get("img_path")
                    or data.get("image")
                )
                if not source_path:
                    raise KeyError(f"Missing source image path for sample {question_id}")

            # Cache check
            cached = None if force_regen else self.load_item_if_exists(question_id, meta_info)
            if cached is not None:
                cached_images = cached.get("images", [])
                if isinstance(cached_images, str):
                    cached_images = [cached_images]
                cached_think = cached.get("think")
                logger.info(
                    f"Skipping {question_id} because item already exists: "
                    f"{self.get_item_path(question_id, meta_info)}"
                )
                cached_res = {
                    "question_id": str(cached.get("question_id", question_id)),
                    "id": str(cached.get("id", question_id)),
                    "prompt": cached.get("prompt", prompt),
                    "images": cached_images,
                }
                if cached_think is not None:
                    # Keep think content in .json (and thus *_evaluated.json).
                    cached_res["think"] = cached_think
                if mode == "i2i" and source_path:
                    cached_res["source_path"] = source_path
                output_info.append(cached_res)
                continue

            # Generate / Edit
            reflection_prompt = data.get("reflection")
            do_second_round = bool(reflection_prompt) and batch_size == 1
            last_state = None
            device = next(self.gen_model.parameters()).device

            if mode == "t2i":
                res = self._generate_images(
                    prompt=prompt,
                    num_images=num_images,
                    batch_size=batch_size,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=cfg_interval,
                    cfg_renorm_min=cfg_renorm_min,
                    timestep_shift=timestep_shift,
                    num_timesteps=num_timesteps,
                    resolution=resolution,
                    think=think,
                    think_simple=think_simple,
                    think_max_length=think_max_length,
                    think_temperature=think_temperature,
                    think_system_prompt=think_system_prompt,
                    return_state=do_second_round,
                )
                if do_second_round:
                    image_list, think_list, last_state = res
                else:
                    image_list, think_list = res
            else:  # i2i
                source_images, _ = load_pil_image(
                    [source_path], img_idx=[0], reqiures_img=True, reduplicate=False
                )
                source_image = source_images[0]
                res = self._edit_image_with_text_img_cfg(
                    image=source_image,
                    prompt=prompt,
                    extra_args=extra_args,
                    return_state=do_second_round,
                )
                if do_second_round:
                    edited_image, think_list, last_state = res
                else:
                    edited_image, think_list = res
                image_list = [edited_image]
            
            print(f"do_second_round: {do_second_round}, last_state: {last_state}")
            if do_second_round and last_state:
                logger.info(f"Running second round with reflection: {reflection_prompt}")
                first_round_think = think_list[0] if think_list else None
                r2_image, r2_think = self._run_second_round_gen(
                    past_key_values=last_state["past_key_values"],
                    past_key_values_prompt=last_state["past_key_values_prompt"],
                    curr_kvlens=last_state["curr_kvlens"],
                    curr_rope=last_state["curr_rope"],
                    prompt=reflection_prompt,
                    image=image_list[0],
                    resolution=resolution,
                    extra_args=extra_args,
                    device=device,
                    original_prompt=prompt,
                    first_round_think=first_round_think,
                )
                if r2_image is not None:
                    image_list.append(r2_image)
                if think_list is None:
                    think_list = []
                think_list.append(r2_think)

            sample_dir = os.path.join(output_dir, "samples")
            os.makedirs(sample_dir, exist_ok=True)

            image_names: list[str] = []
            image_paths: list[str] = []
            for i, sample in enumerate(image_list):
                # I2I usually has 1 image, naming it _00000.png matches logic if loop runs once
                image_name = f"{question_id}_{i:05}.png"
                sample_path = os.path.join(sample_dir, image_name)
                sample.save(sample_path)
                image_names.append(image_name)
                image_paths.append(sample_path)

            out_item_result: dict[str, Any] = {
                "question_id": question_id,
                "id": question_id,
                "prompt": prompt,
                "images": image_names,
            }
            if think_list is not None:
                # Save think content into task-level result (.json -> *_evaluated.json).
                out_item_result["think"] = think_list
            if mode == "i2i":
                out_item_result["source_path"] = source_path

            output_info.append(out_item_result)
            if save_items:
                out_item_save: dict[str, Any] = dict(out_item_result)
                out_item_save["image_paths"] = image_paths
                self.save_item(
                    out_item_save,
                    question_id=question_id,
                    meta_info=meta_info,
                )

        self._finalize_task_results(output_info, meta_info)

    def _run_t2i_task(self, task_name: str, meta_info: Dict[str, Any]):
        self._run_gen_task(task_name, meta_info, mode="t2i")

    def _run_i2i_task(self, task_name: str, meta_info: Dict[str, Any]):
        self._run_gen_task(task_name, meta_info, mode="i2i")

    def _update_text_cache_with_prompts(
        self,
        past_key_values,
        prompts: list[str],
        curr_kvlens: list[int],
        curr_rope: list[int],
        device,
    ):
        generation_input, curr_kvlens, curr_rope = self.model.prepare_prompts(
            curr_kvlens=curr_kvlens,
            curr_rope=curr_rope,
            prompts=prompts,
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        generation_input = self._move_generation_input_to_device(
            generation_input, device
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.model.forward_cache_update_text(
                past_key_values, **generation_input
            )
        return past_key_values, curr_kvlens, curr_rope

    def _generate_think_and_update_cache(
        self,
        past_key_values,
        curr_kvlens: list[int],
        curr_rope: list[int],
        device,
        think_max_length: int,
        think_temperature: float,
        think_simple: bool,
    ) -> tuple[Any, list[int], list[int], str]:
        # Generate think text
        tmp_past_key_values = copy.deepcopy(past_key_values)
        tmp_generation_input = self.model.prepare_start_tokens(
            curr_kvlens, curr_rope, self.new_token_ids
        )
        tmp_generation_input = self._move_generation_input_to_device(
            tmp_generation_input, device
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            generated_token_ids = self.model.generate_text(
                past_key_values=tmp_past_key_values,
                max_length=think_max_length,
                do_sample=True,
                temperature=think_temperature,
                end_token_id=self.new_token_ids.get("eos_token_id"),
                **tmp_generation_input,
            )
        decoded = self._decode_generated_text(generated_token_ids)
        print(f"decoded think text: {decoded}")
        think_output = self._extract_think_text(decoded)
        if think_simple:
            parts = think_output.split("</think>")
            if len(parts) > 1 and parts[1] != "":
                think_output = parts[1].strip()

        # Feed back
        past_key_values, curr_kvlens, curr_rope = self._update_text_cache_with_prompts(
            past_key_values, [think_output], curr_kvlens, curr_rope, device
        )
        return past_key_values, curr_kvlens, curr_rope, think_output

    def _decode_latents(
        self, latents: torch.Tensor, h: int, w: int, device
    ) -> list[Image.Image]:
        """Decode a batch of latents to PIL images."""
        images = []
        vae_dtype = next(self.vae_model.parameters()).dtype
        
        for latent in latents:
            latent = latent.reshape(1, h // 16, w // 16, 2, 2, 16)
            latent = torch.einsum("nhwpqc->nchpwq", latent)
            latent = latent.reshape(1, 16, h // 8, w // 8)
            
            image_tensor = self.vae_model.decode(latent.to(device=device, dtype=vae_dtype))
            image_np = (
                (image_tensor * 0.5 + 0.5)
                .clamp(0, 1)[0]
                .permute(1, 2, 0)
                .mul(255)
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            images.append(Image.fromarray(image_np))
        return images

    def _edit_image_with_text_img_cfg(
        self,
        image: Image.Image,
        prompt: str,
        extra_args: Dict[str, Any],
        use_think_default: bool = False,
        stride: int = 16,
        return_state: bool = False,
    ) -> tuple[Image.Image, list[str] | None] | tuple[Image.Image, list[str] | None, dict]:
        """
        Mirror gen_edit.editing_with_text_img_cfg to edit an image with text + source conditioning.
        """
        def _make_divisible(value, stride):
            """Ensure the value is divisible by the stride."""
            return max(stride, int(round(value / stride) * stride))

        def _apply_scale(width, height, scale):
            new_width = round(width * scale)
            new_height = round(height * scale)
            new_width = _make_divisible(new_width, stride)
            new_height = _make_divisible(new_height, stride)
            return new_width, new_height

        device = next(self.gen_model.parameters()).device

        # Hyperparameters.
        num_timesteps = int(extra_args.get("num_timesteps", 50))
        cfg_text_scale = float(extra_args.get("cfg_text_scale", 4.0))
        cfg_img_scale = float(extra_args.get("cfg_img_scale", 1.5))
        cfg_type = str(extra_args.get("cfg_type", "serial_text_img"))
        cfg_interval = extra_args.get("cfg_interval", [0.4, 1.0])
        cfg_renorm_min = float(extra_args.get("cfg_renorm_min", 0.0))
        cfg_renorm_type = str(extra_args.get("cfg_renorm_type", "text_channel"))
        timestep_shift = float(extra_args.get("timestep_shift", 3.0))
        use_vit = bool(extra_args.get("use_vit", False))
        use_think = bool(extra_args.get("think", use_think_default))
        think_temperature = float(extra_args.get("think_temperature", 0.3))
        think_max_length = int(extra_args.get("think_max_length", 10240))
        seed = extra_args.get("seed", None)
        think_simple = bool(extra_args.get("think_simple", False))

        # Optional seeding for reproducibility.
        if seed is not None:
            import random
            import numpy as np
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        max_image_size = self.vae_transform.resize_transform.max_size
        min_image_size = self.vae_transform.resize_transform.min_size
        # Prepare image size
        w, h = image.size
        scale = min(max_image_size / max(w, h), 1.0)
        scale = max(scale, min_image_size / min(w, h))
        w, h = _apply_scale(w, h, scale)

        if max(w, h) > max_image_size:
            scale = max_image_size / max(w, h)
            w, h = _apply_scale(w, h, scale)

        past_key_values = NaiveCache(self.gen_model.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]
        think_list: list[str] | None = [] if use_think else None

        # Optional think priming (system prompt).
        if use_think:
            past_key_values, newlens, new_rope = self._update_text_cache_with_prompts(
                past_key_values, [SYSTEM_PROMPT_WITH_THINK], newlens, new_rope, device
            )

        # Encode source image via VAE path (conditioning).
        generation_input, newlens, new_rope = self.model.prepare_vae_images(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            images=[image],
            transforms=self.vae_transform,
            new_token_ids=self.new_token_ids,
            timestep=0.0,
        )
        generation_input = self._move_generation_input_to_device(
            generation_input, device
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.model.forward_cache_update_vae(
                self.vae_model, past_key_values, **generation_input
            )

        # Optionally add VIT path (visual understanding tokens).
        if use_vit:
            generation_input, newlens, new_rope = self.model.prepare_vit_images(
                curr_kvlens=newlens,
                curr_rope=new_rope,
                images=[image],
                transforms=self.vit_transform,
                new_token_ids=self.new_token_ids,
            )
            generation_input = self._move_generation_input_to_device(
                generation_input, device
            )
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past_key_values = self.model.forward_cache_update_vit(
                    past_key_values, **generation_input
                )

        # Text CFG cache (copy of current state).
        cfg_text_past_key_values = copy.deepcopy(past_key_values)
        cfg_text_newlens = copy.deepcopy(newlens)
        cfg_text_new_rope = copy.deepcopy(new_rope)
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            image_sizes=[(h, w)],
        )

        # Main prompt.
        past_key_values, newlens, new_rope = self._update_text_cache_with_prompts(
            past_key_values, [prompt], newlens, new_rope, device
        )

        # Optional think loop (generate + re-inject).
        if use_think:
            (
                past_key_values,
                newlens,
                new_rope,
                think_output,
            ) = self._generate_think_and_update_cache(
                past_key_values,
                newlens,
                new_rope,
                device,
                think_max_length,
                think_temperature,
                think_simple,
            )
            assert think_list is not None
            think_list.append(think_output)

        # Prepare VAE latents for generation (shape from resized H/W).
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            image_sizes=[(h, w)],
            new_token_ids=self.new_token_ids,
        )
        generation_input = self._move_generation_input_to_device(
            generation_input, device
        )

        # Image CFG branch (text side).
        cfg_img_past_key_values = NaiveCache(
            self.gen_model.config.llm_config.num_hidden_layers
        )
        cfg_img_newlens = [0]
        cfg_img_new_rope = [0]
        cfg_img_texts = (
            [SYSTEM_PROMPT_WITH_THINK, prompt, think_list[0]]
            if use_think and think_list
            else [prompt]
        )
        for text in cfg_img_texts:
            (
                cfg_img_past_key_values,
                cfg_img_newlens,
                cfg_img_new_rope,
            ) = self._update_text_cache_with_prompts(
                cfg_img_past_key_values,
                [text],
                cfg_img_newlens,
                cfg_img_new_rope,
                device,
            )

        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_img_newlens,
            curr_rope=cfg_img_new_rope,
            image_sizes=[(h, w)],
        )

        cfg_text_args = {
            "cfg_text_packed_position_ids": generation_input_cfg_text[
                "cfg_packed_position_ids"
            ],
            "cfg_text_packed_query_indexes": generation_input_cfg_text[
                "cfg_packed_query_indexes"
            ],
            "cfg_text_key_values_lens": generation_input_cfg_text[
                "cfg_key_values_lens"
            ],
            "cfg_text_packed_key_value_indexes": generation_input_cfg_text[
                "cfg_packed_key_value_indexes"
            ],
        }
        cfg_img_args = {
            "cfg_img_packed_position_ids": generation_input_cfg_img[
                "cfg_packed_position_ids"
            ],
            "cfg_img_packed_query_indexes": generation_input_cfg_img[
                "cfg_packed_query_indexes"
            ],
            "cfg_img_key_values_lens": generation_input_cfg_img[
                "cfg_key_values_lens"
            ],
            "cfg_img_packed_key_value_indexes": generation_input_cfg_img[
                "cfg_packed_key_value_indexes"
            ],
        }

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            generation_input = self._move_generation_input_to_device(
                generation_input, device
            )
            cfg_text_args = self._move_generation_input_to_device(
                cfg_text_args, device
            )
            cfg_img_args = self._move_generation_input_to_device(cfg_img_args, device)
            unpacked_latent = self.model.generate_image(
                past_key_values=past_key_values,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_img_past_key_values=cfg_img_past_key_values,
                num_timesteps=num_timesteps,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_type=cfg_type,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                timestep_shift=timestep_shift,
                **generation_input,
                **cfg_text_args,
                **cfg_img_args,
            )

        decoded_images = self._decode_latents(unpacked_latent, h, w, device)
        if return_state:
            state = {
                "past_key_values": past_key_values,
                "curr_kvlens": newlens,
                "curr_rope": new_rope,
                "cfg_text_past_key_values": cfg_text_past_key_values,
                "cfg_text_curr_kvlens": cfg_text_newlens,
                "cfg_text_curr_rope": cfg_text_new_rope,
                "cfg_img_past_key_values": cfg_img_past_key_values,
                "cfg_img_curr_kvlens": cfg_img_newlens,
                "cfg_img_curr_rope": cfg_img_new_rope,
            }
            return decoded_images[0], think_list, state
        return decoded_images[0], think_list

    def _run_vqa_task(self, task_name: str, meta_info: Dict[str, Any]):
        results = []
        cnt = 0
        start_time = None
        extra_args = getattr(self, "extra_args", {}) or {}
        num_workers = int(extra_args.get("num_workers", 2))
        max_new_tokens = int(extra_args.get("max_new_tokens", 1024))
        save_items = bool(extra_args.get("save_items", True))
        os.makedirs(self.get_items_dir(meta_info), exist_ok=True)
        data_loader = self.create_data_loader(
            BagelDataset,
            task_name,
            collate_fn=default_collate_fn,
            batch_size=1,
            num_workers=num_workers,
        )

        for question_id, question, images in data_loader:
            if cnt == 0:
                start_time = time.perf_counter()
            cnt += 1

            qid_str = str(question_id[0])
            cached = self.load_item_if_exists(qid_str, meta_info)
            if cached is not None:
                results.append(
                    {
                        "answer": cached.get("answer", ""),
                        "question_id": cached.get("question_id", qid_str),
                        "prompt": cached.get("prompt", ""),
                    }
                )
                continue

            # unwrap batch dimension (batch=1)
            images, prompt = process_conversation(images[0], question[0])

            pred = self.model.chat(
                self.tokenizer,
                self.new_token_ids,
                self.image_transform,
                images=images,
                prompt=prompt,
                max_length=max_new_tokens,
            )

            results.append(
                {
                    "answer": pred,
                    "question_id": question_id[0],
                    "prompt": prompt,
                }
            )
            if save_items:
                self.save_item(
                    {
                        "answer": pred,
                        "question_id": question_id[0],
                        "prompt": prompt,
                    },
                    question_id=qid_str,
                    meta_info=meta_info,
                )

        correct_num = self._finalize_task_results(results, meta_info)

        if correct_num is not None:
            if start_time is not None and cnt > 0:
                total_time = time.perf_counter() - start_time
                print(
                    f"Total time: {total_time}\nAverage time:{total_time / cnt}\nResults_collect number: {correct_num}"
                )

        rank = (
            self.accelerator.state.local_process_index
            if self.accelerator is not None
            else 0
        )
        print("rank", rank, "finished")

    def _move_generation_input_to_device(
        self, generation_input: Dict[str, Any], device
    ):
        for k, v in generation_input.items():
            if isinstance(v, torch.Tensor):
                generation_input[k] = v.to(device)
        return generation_input

    def _decode_generated_text(self, token_ids: torch.Tensor) -> str:
        # token_ids can be [seq] or [seq, batch]; we decode the first sample.
        if not isinstance(token_ids, torch.Tensor):
            return str(token_ids)
        if token_ids.dim() == 2:
            token_ids = token_ids[:, 0]
        elif token_ids.dim() > 2:
            token_ids = token_ids.reshape(-1)
        return self.tokenizer.decode(token_ids)

    def _extract_think_text(self, decoded: str) -> str:
        # infer_wise.py assumes Qwen-style chat markers.
        try:
            decoded = decoded.split("<|im_end|>")[0]
            parts = decoded.split("<|im_start|>")
            if len(parts) >= 2:
                return parts[1]
            return decoded
        except Exception:
            return decoded

    @torch.inference_mode()
    def _generate_images(
        self,
        prompt: str,
        num_images: int,
        batch_size: int,
        cfg_text_scale: float,
        cfg_img_scale: float,
        cfg_interval,
        cfg_renorm_min: float,
        timestep_shift: float,
        num_timesteps: int,
        resolution: int,
        think: bool = False,
        think_simple: bool = False,
        think_max_length: int = 2048,
        think_temperature: float = 0.3,
        think_system_prompt: str = SYSTEM_PROMPT_WITH_THINK,
        return_state: bool = False,
    ):
        # For T2I, we generally do not have an image condition for CFG, so force img_scale to 1.0
        # to avoid triggering the image branch in the model which requires missing arguments.
        if cfg_img_scale != 1.0:
            logger.warning(f"T2I mode detected, forcing cfg_img_scale from {cfg_img_scale} to 1.0")
            cfg_img_scale = 1.0

        device = next(self.gen_model.parameters()).device
        image_list = []
        think_list: list[str] | None = [] if think else None
        last_state = None

        assert batch_size == 1, "batch_size must be 1 for T2I"
        for _ in range(0, num_images, batch_size):
            curr_batch = min(batch_size, num_images - len(image_list))
            if think and curr_batch != 1:
                raise AssertionError("think mode currently requires batch_size=1")

            past_key_values = NaiveCache(
                self.gen_model.config.llm_config.num_hidden_layers
            )
            past_key_values_prompt = ""
            newlens = [0] * curr_batch
            new_rope = [0] * curr_batch

            # Optional: do a "think" step (generate a short plan), then feed it back.
            if think:
                # 1) system prompt
                past_key_values, newlens, new_rope = self._update_text_cache_with_prompts(
                    past_key_values, [think_system_prompt], newlens, new_rope, device
                )
                past_key_values_prompt = think_system_prompt

                # 2) user prompt
                past_key_values, newlens, new_rope = self._update_text_cache_with_prompts(
                    past_key_values, [prompt], newlens, new_rope, device
                )
                past_key_values_prompt += prompt
                # 3) generate think text + feed back
                (
                    past_key_values,
                    newlens,
                    new_rope,
                    think_output,
                ) = self._generate_think_and_update_cache(
                    past_key_values,
                    newlens,
                    new_rope,
                    device,
                    think_max_length,
                    think_temperature,
                    think_simple,
                )
                past_key_values_prompt += think_output
                assert think_list is not None
                think_list.append(think_output)
            else:
                past_key_values, newlens, new_rope = self._update_text_cache_with_prompts(
                    past_key_values, [prompt] * curr_batch, newlens, new_rope, device
                )
                past_key_values_prompt += prompt
            generation_input = self.gen_model.prepare_vae_latent(
                curr_kvlens=newlens,
                curr_rope=new_rope,
                image_sizes=[(resolution, resolution)] * curr_batch,
                new_token_ids=self.new_token_ids,
            )
            generation_input = self._move_generation_input_to_device(
                generation_input, device
            )

            cfg_past_key_values = NaiveCache(
                self.gen_model.config.llm_config.num_hidden_layers
            )
            cfg_newlens = [0] * curr_batch
            cfg_new_rope = [0] * curr_batch

            generation_input_cfg = self.gen_model.prepare_vae_latent_cfg(
                curr_kvlens=cfg_newlens,
                curr_rope=cfg_new_rope,
                image_sizes=[(resolution, resolution)] * curr_batch,
            )
            generation_input_cfg = self._move_generation_input_to_device(
                generation_input_cfg, device
            )

            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    unpacked_latent = self.model.generate_image(
                        past_key_values=past_key_values,
                        num_timesteps=num_timesteps,
                        cfg_text_scale=cfg_text_scale,
                        cfg_img_scale=cfg_img_scale,
                        cfg_interval=cfg_interval,
                        cfg_renorm_min=cfg_renorm_min,
                        timestep_shift=timestep_shift,
                        cfg_text_past_key_values=cfg_past_key_values,
                        cfg_text_packed_position_ids=generation_input_cfg[
                            "cfg_packed_position_ids"
                        ],
                        cfg_text_key_values_lens=generation_input_cfg[
                            "cfg_key_values_lens"
                        ],
                        cfg_text_packed_query_indexes=generation_input_cfg[
                            "cfg_packed_query_indexes"
                        ],
                        cfg_text_packed_key_value_indexes=generation_input_cfg[
                            "cfg_packed_key_value_indexes"
                        ],
                        **generation_input,
                    )

            decoded_batch = self._decode_latents(
                unpacked_latent, resolution, resolution, device
            )
            image_list.extend(decoded_batch)
            
            if return_state:
                last_state = {
                    "past_key_values": past_key_values,
                    "curr_kvlens": newlens,
                    "curr_rope": new_rope,
                    "past_key_values_prompt": past_key_values_prompt,
                }

        if return_state:
            return image_list, think_list, last_state
        return image_list, think_list


if __name__ == "__main__":
    args = parse_args()
    defaults_obj = RunCfg()
    model_adapter = ModelAdapter(
        extra_cfg=args.cfg,
        task_names=None,
    )
    model_adapter.run()
