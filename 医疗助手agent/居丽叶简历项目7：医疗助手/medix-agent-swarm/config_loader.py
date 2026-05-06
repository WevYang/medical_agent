"""
运行时配置加载器。

优先级：
1. 调用方显式传参
2. 仓库上层的 config.py
3. 环境变量
4. 内置默认值
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT.parent / "config.py"

DEFAULT_LLM_CONFIG: Dict[str, Any] = {
    "api_key": "",
    "model_name": "gpt-4o-mini",
    "base_url": "https://api.openai.com/v1",
    "temperature": 0.7,
    "max_tokens": 8192,
}

DEFAULT_MEM0_CONFIG: Dict[str, Any] = {
    "api_key": "",
}


def project_config_path() -> Path:
    """返回仓库配置文件路径。"""
    return CONFIG_PATH


def _load_repo_config_module() -> Optional[ModuleType]:
    """按文件路径加载仓库根目录的 config.py。"""
    if not CONFIG_PATH.exists():
        return None

    spec = importlib.util.spec_from_file_location("medix_repo_config", CONFIG_PATH)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _apply_env_override(
    config: Dict[str, Any],
    key: str,
    env_name: str,
    cast: type = str,
) -> None:
    """使用环境变量覆盖配置。"""
    raw_value = os.getenv(env_name)
    if raw_value in (None, ""):
        return

    if cast is str:
        config[key] = raw_value
        return

    try:
        config[key] = cast(raw_value)
    except ValueError:
        # 忽略格式错误的环境变量，保持已有配置
        pass


def load_llm_config() -> Dict[str, Any]:
    """加载 LLM 配置。"""
    config = DEFAULT_LLM_CONFIG.copy()
    module = _load_repo_config_module()

    if module is not None:
        llm_config = getattr(module, "LLM_CONFIG", None)
        if isinstance(llm_config, dict):
            config.update({k: v for k, v in llm_config.items() if v is not None})

    _apply_env_override(config, "api_key", "OPENAI_API_KEY")
    _apply_env_override(config, "api_key", "LLM_API_KEY")
    _apply_env_override(config, "model_name", "LLM_MODEL_NAME")
    _apply_env_override(config, "base_url", "LLM_BASE_URL")
    _apply_env_override(config, "temperature", "LLM_TEMPERATURE", float)
    _apply_env_override(config, "max_tokens", "LLM_MAX_TOKENS", int)

    return config


def load_mem0_config() -> Dict[str, Any]:
    """加载 Mem0 配置。"""
    config = DEFAULT_MEM0_CONFIG.copy()
    module = _load_repo_config_module()

    if module is not None:
        mem0_config = getattr(module, "MEM0_CONFIG", None)
        if isinstance(mem0_config, dict):
            config.update({k: v for k, v in mem0_config.items() if v is not None})

    _apply_env_override(config, "api_key", "MEM0_API_KEY")

    return config
