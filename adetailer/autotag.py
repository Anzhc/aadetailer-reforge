from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from PIL import Image

from adetailer.common import ensure_pil_image

VIT_LARGE_MODEL_REPO = "SmilingWolf/wd-vit-large-tagger-v3"
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"


class AutoTaggerError(RuntimeError):
    pass


@dataclass
class LabelData:
    names: list[str]
    rating: list[int]
    general: list[int]
    character: list[int]


def prepare_image(image: Image.Image, target_size: int) -> np.ndarray:
    import numpy as np

    image = ensure_pil_image(image, "RGB")

    max_dim = max(image.size)
    pad_left = (max_dim - image.size[0]) // 2
    pad_top = (max_dim - image.size[1]) // 2
    padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded.paste(image, (pad_left, pad_top))
    padded = padded.resize((target_size, target_size), Image.BICUBIC)

    arr = np.asarray(padded, dtype=np.float32)[..., [2, 1, 0]]
    return np.expand_dims(arr, axis=0)


def _download_model(model_repo: str):
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:  # noqa: BLE001
        raise AutoTaggerError(
            "huggingface_hub is required for ADetailer autotagging but is not installed"
        ) from e

    csv_path = hf_hub_download(model_repo, LABEL_FILENAME)
    model_path = hf_hub_download(model_repo, MODEL_FILENAME)
    return csv_path, model_path


def _load_model_and_tags(model_repo: str):
    import numpy as np

    try:
        import onnxruntime as rt
    except Exception as e:  # noqa: BLE001
        raise AutoTaggerError(
            "onnxruntime is required for ADetailer autotagging but is not installed"
        ) from e

    try:
        import pandas as pd
    except Exception as e:  # noqa: BLE001
        raise AutoTaggerError(
            "pandas is required for ADetailer autotagging but is not installed"
        ) from e

    csv_path, model_path = _download_model(model_repo)
    df = pd.read_csv(csv_path)
    tag_data = LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0]),
    )

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        session = rt.InferenceSession(model_path, providers=providers)
    except Exception:
        session = rt.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    target_size = session.get_inputs()[0].shape[2]
    return session, tag_data, target_size


@lru_cache(maxsize=1)
def get_autotagger(model_repo: str = VIT_LARGE_MODEL_REPO):
    return _load_model_and_tags(model_repo)


def autotag_image(
    image: Image.Image,
    *,
    general_thresh: float = 0.35,
    character_thresh: float = 0.85,
    hide_rating: bool = True,
    character_first: bool = True,
    remove_separator: bool = True,
    model_repo: str = VIT_LARGE_MODEL_REPO,
) -> list[str]:
    import numpy as np

    try:
        session, tag_data, target_size = get_autotagger(model_repo)
    except AutoTaggerError:
        raise
    except Exception as e:  # noqa: BLE001
        raise AutoTaggerError(f"ADetailer autotagging failed to load model: {e}") from e

    try:
        processed = prepare_image(image, target_size)
        preds = session.run(None, {session.get_inputs()[0].name: processed})[0]
    except Exception as e:  # noqa: BLE001
        raise AutoTaggerError(f"ADetailer autotagging inference failed: {e}") from e

    scores = preds.flatten()

    character = [
        tag_data.names[i] for i in tag_data.character if scores[i] >= character_thresh
    ]
    general = [
        tag_data.names[i] for i in tag_data.general if scores[i] >= general_thresh
    ]
    rating = [] if hide_rating else [tag_data.names[i] for i in tag_data.rating]

    final: list[str] = (
        character + general if character_first else general + character
    ) + rating

    if remove_separator:
        final = [tag.replace("_", " ") for tag in final]

    return final
