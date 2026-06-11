from __future__ import annotations

import os
from types import MethodType
from pathlib import Path


def configure_hf_cache_env(repo_root: str | Path | None = None) -> dict[str, str]:
    base_dir = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent
    cache_root = base_dir.expanduser().resolve() / ".hf-cache"

    env_map = {
        "HF_HOME": cache_root,
        "HF_HUB_CACHE": cache_root / "hub",
        "HUGGINGFACE_HUB_CACHE": cache_root / "hub",
        "HF_MODULES_CACHE": cache_root / "modules",
        "TRANSFORMERS_CACHE": cache_root / "transformers",
    }

    resolved: dict[str, str] = {}
    for env_name, default_path in env_map.items():
        effective_path = Path(os.environ.get(env_name, str(default_path))).expanduser().resolve()
        effective_path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(env_name, str(effective_path))
        resolved[env_name] = str(effective_path)
    return resolved


def patch_model_hf_tokenizer_cache(model, repo_root: str | Path | None = None) -> str:
    base_dir = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent
    cache_dir = base_dir.expanduser().resolve() / ".hf-cache" / "remote-tokenizer"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "_resolve_hf_cache_dir"):
        def _resolve_hf_cache_dir(self) -> str:
            return str(cache_dir)

        model._resolve_hf_cache_dir = MethodType(_resolve_hf_cache_dir, model)

    return str(cache_dir)
