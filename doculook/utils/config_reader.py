# Copyright (c) Opendatalab. All rights reserved.
import json
import os
from loguru import logger

try:
    import torch
    import torch_npu
except ImportError:
    pass


# 定义配置文件名常量（DocuLook 优先，兼容 MinerU）
CONFIG_FILE_NAME = os.getenv('DOCULOOK_TOOLS_CONFIG_JSON', None) or os.getenv('MINERU_TOOLS_CONFIG_JSON', None) or 'doculook.json'


def read_config():
    if os.path.isabs(CONFIG_FILE_NAME):
        config_file = CONFIG_FILE_NAME
    else:
        home_dir = os.path.expanduser('~')
        config_file = os.path.join(home_dir, CONFIG_FILE_NAME)

    if not os.path.exists(config_file):
        # logger.warning(f'{config_file} not found, using default configuration')
        return None
    else:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config


def get_s3_config(bucket_name: str):
    """~/magic-pdf.json 读出来."""
    config = read_config()

    bucket_info = config.get('bucket_info')
    if bucket_name not in bucket_info:
        access_key, secret_key, storage_endpoint = bucket_info['[default]']
    else:
        access_key, secret_key, storage_endpoint = bucket_info[bucket_name]

    if access_key is None or secret_key is None or storage_endpoint is None:
        raise Exception(f'ak, sk or endpoint not found in {CONFIG_FILE_NAME}')

    # logger.info(f"get_s3_config: ak={access_key}, sk={secret_key}, endpoint={storage_endpoint}")

    return access_key, secret_key, storage_endpoint


def get_s3_config_dict(path: str):
    access_key, secret_key, storage_endpoint = get_s3_config(get_bucket_name(path))
    return {'ak': access_key, 'sk': secret_key, 'endpoint': storage_endpoint}


def get_bucket_name(path):
    bucket, key = parse_bucket_key(path)
    return bucket


def parse_bucket_key(s3_full_path: str):
    """
    输入 s3://bucket/path/to/my/file.txt
    输出 bucket, path/to/my/file.txt
    """
    s3_full_path = s3_full_path.strip()
    if s3_full_path.startswith("s3://"):
        s3_full_path = s3_full_path[5:]
    if s3_full_path.startswith("/"):
        s3_full_path = s3_full_path[1:]
    bucket, key = s3_full_path.split("/", 1)
    return bucket, key


def get_device():
    device_mode = os.getenv('DOCULOOK_DEVICE_MODE', None) or os.getenv('MINERU_DEVICE_MODE', None)
    if device_mode is not None:
        return device_mode
    else:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            try:
                if torch_npu.npu.is_available():
                    return "npu"
            except Exception as e:
                pass
        return "cpu"


def get_formula_enable(formula_enable):
    formula_enable_env = os.getenv('DOCULOOK_FORMULA_ENABLE') or os.getenv('MINERU_FORMULA_ENABLE')
    formula_enable = formula_enable if formula_enable_env is None else formula_enable_env.lower() == 'true'
    return formula_enable


def get_table_enable(table_enable):
    table_enable_env = os.getenv('DOCULOOK_TABLE_ENABLE') or os.getenv('MINERU_TABLE_ENABLE')
    table_enable = table_enable if table_enable_env is None else table_enable_env.lower() == 'true'
    return table_enable


def get_latex_delimiter_config():
    config = read_config()
    if config is None:
        return None
    latex_delimiter_config = config.get('latex-delimiter-config', None)
    if latex_delimiter_config is None:
        # logger.warning(f"'latex-delimiter-config' not found in {CONFIG_FILE_NAME}, use 'None' as default")
        return None
    else:
        return latex_delimiter_config


def get_llm_aided_config():
    config = read_config()
    if config is None:
        return None
    llm_aided_config = config.get('llm-aided-config', None)
    if llm_aided_config is None:
        # logger.warning(f"'llm-aided-config' not found in {CONFIG_FILE_NAME}, use 'None' as default")
        return None
    else:
        return llm_aided_config


def get_local_models_dir():
    config = read_config()
    if config is None:
        return None
    models_dir = config.get('models-dir')
    if models_dir is None:
        logger.warning(f"'models-dir' not found in {CONFIG_FILE_NAME}, use None as default")
    return models_dir


def get_qwen_vl_config():
    """获取Qwen-VL配置"""
    config = read_config()
    if config is None:
        return None
    qwen_vl_config = config.get('qwen-vl-config', None)
    if qwen_vl_config is None:
        logger.warning(f"'qwen-vl-config' not found in {CONFIG_FILE_NAME}, use None as default")
        return None
    elif not qwen_vl_config.get('enable', False):
        logger.info("Qwen-VL config is disabled")
        return None
    else:
        return qwen_vl_config


def get_vision_llm_config():
    """获取通用 Vision LLM 配置（OpenAI 兼容接口）。

    优先读取配置键 'vision-llm-config'，若不存在则回退到 'qwen-vl-config'（兼容旧配置）。
    期望字段：
      - enable: bool
      - api_key: str
      - base_url: str (可选，默认 https://api.openai.com/v1)
      - model: str
      - temperature/top_p/max_tokens: 可选
    """
    config = read_config()
    if config is None:
        return None
    llm_cfg = config.get('vision-llm-config', None)
    if llm_cfg is None:
        # 兼容旧配置
        qwen = config.get('qwen-vl-config', None)
        if qwen and qwen.get('enable', False):
            return {
                'enable': True,
                'api_key': qwen.get('api_key'),
                'base_url': qwen.get('base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
                'model': qwen.get('model', 'qwen-vl-max'),
                'temperature': qwen.get('temperature', 0.0),
                'top_p': qwen.get('top_p', 0.8),
                'max_tokens': qwen.get('max_tokens', 16384),
            }
        return None
    elif not llm_cfg.get('enable', False):
        logger.info("Vision LLM config is disabled")
        return None
    else:
        return llm_cfg

