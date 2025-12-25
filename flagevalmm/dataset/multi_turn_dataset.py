import copy
import json
import os.path as osp
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset

from flagevalmm.common.const import FLAGEVALMM_DATASETS_CACHE_DIR
from flagevalmm.dataset.utils import get_data_root
from flagevalmm.registry import DATASETS


@DATASETS.register_module()
class MultiTurnDataset(Dataset):
    """Multi-turn dataset that yields OpenAI-style conversation messages.

    Each record is expected to follow the schema used in
    `/share/project/mmdataset/t2i/GEdit-Bench/multi-turn/data.json`, i.e.:
      - id / question_id
      - messages: list of {"role": ..., "content": ...}
      - metadata: optional dict with extra information
    Image URLs inside message content are resolved to absolute paths so they can
    be consumed directly by downstream clients that will base64-encode them.
    """

    def __init__(
        self,
        name: str,
        data_root: Optional[str] = None,
        anno_file: Optional[str] = None,
        cache_dir: str = FLAGEVALMM_DATASETS_CACHE_DIR,
        config: Optional[dict] = None,
        base_dir: Optional[str] = None,
        debug: bool = False,
        image_dir: Optional[str] = None,
        id_key: str = "id",
        question_id_key: str = "question_id",
        messages_key: str = "messages",
        metadata_key: str = "metadata",
        **kwargs,
    ) -> None:
        self.data_root = get_data_root(
            data_root=data_root, config=config, cache_dir=cache_dir, base_dir=base_dir
        )
        self.image_root = (
            osp.join(self.data_root, image_dir) if image_dir else self.data_root
        )
        self.id_key = id_key
        self.question_id_key = question_id_key
        self.messages_key = messages_key
        self.metadata_key = metadata_key

        anno_file = "data.json" if anno_file is None else anno_file
        self.data = json.load(open(osp.join(self.data_root, anno_file)))
        self.name = name
        self.debug = debug
        if debug:
            self.data = self.data[:16]

    def __len__(self) -> int:
        return len(self.data)

    def text_number(self) -> int:
        return len(self.data)

    def _resolve_image_url(self, url: str) -> str:
        """Resolve relative image path to an absolute path (keeps http/https)."""
        if not isinstance(url, str):
            raise TypeError(f"image_url must be str, got {type(url)}")
        if url.startswith("http://") or url.startswith("https://") or url.startswith(
            "data:image"
        ):
            return url
        if osp.isabs(url):
            return url
        return osp.join(self.image_root, url)

    def _prepare_messages(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Deep-copy messages and resolve image_url paths."""
        if self.messages_key not in record:
            raise KeyError(f"Missing '{self.messages_key}' in record: {record}")
        messages = []
        for msg in record[self.messages_key]:
            role = msg.get("role")
            content = msg.get("content")

            if isinstance(content, list):
                prepared_content = []
                for part in content:
                    if (
                        isinstance(part, dict)
                        and part.get("type") == "image_url"
                        and isinstance(part.get("image_url"), dict)
                        and "url" in part["image_url"]
                    ):
                        resolved_url = self._resolve_image_url(
                            part["image_url"]["url"]
                        )
                        new_part = copy.deepcopy(part)
                        new_part["image_url"]["url"] = resolved_url
                        prepared_content.append(new_part)
                    else:
                        prepared_content.append(copy.deepcopy(part))
                messages.append({"role": role, "content": prepared_content})
            else:
                # Plain string content (e.g., assistant text)
                messages.append({"role": role, "content": copy.deepcopy(content)})
        return messages

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.data[index]
        if self.id_key not in record:
            raise KeyError(f"Missing '{self.id_key}' in data index {index}")
        question_id = str(
            record.get(self.question_id_key, record.get(self.id_key, index))
        )
        messages = self._prepare_messages(record)
        metadata = copy.deepcopy(record.get(self.metadata_key, {}))

        return {
            "id": str(record[self.id_key]),
            "question_id": question_id,
            "messages": messages,
            "metadata": metadata,
        }

    def get_data(self, index: int) -> Dict[str, Any]:
        assert index < self.text_number()
        return self.__getitem__(index)

    def meta_info(self) -> Dict[str, Any]:
        return {"name": self.name, "length": len(self.data), "type": "multi_turn"}

    def get_annotation(self) -> Dict[str, Any]:
        anno_dict = {}
        for i in range(self.text_number()):
            item = self[i]
            anno_dict[item["question_id"]] = item
        return anno_dict
