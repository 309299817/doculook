import json
import os
import sys
import click
import requests
from loguru import logger

from doculook.utils.enum_class import ModelPath
from doculook.utils.models_download_utils import auto_download_and_get_model_root_path


def download_json(url):
    """下载JSON文件"""
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def download_and_modify_json(url, local_filename, modifications):
    """下载JSON并修改内容"""
    if os.path.exists(local_filename):
        data = json.load(open(local_filename))
        config_version = data.get('config_version', '0.0.0')
        if config_version < '1.3.0':
            data = download_json(url)
    else:
        data = download_json(url)

    # 修改内容
    for key, value in modifications.items():
        if key in data:
            if isinstance(data[key], dict):
                # 如果是字典，合并新值
                data[key].update(value)
            else:
                # 否则直接替换
                data[key] = value

    # 保存修改后的内容
    with open(local_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def configure_model(model_dir, model_type):
    """配置模型"""
    json_url = 'https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/mineru.template.json'
    config_file_name = os.getenv('DOCULOOK_TOOLS_CONFIG_JSON', 'doculook.json')
    home_dir = os.path.expanduser('~')
    config_file = os.path.join(home_dir, config_file_name)

    json_mods = {
        'models-dir': {
            f'{model_type}': model_dir
        }
    }

    download_and_modify_json(json_url, config_file, json_mods)
    logger.info(f'The configuration file has been successfully configured, the path is: {config_file}')


def download_pipeline_models():
    """下载Pipeline模型"""
    model_paths = [
        ModelPath.doclayout_yolo,
        ModelPath.yolo_v8_mfd,
        ModelPath.unimernet_small,
        ModelPath.pytorch_paddle,
        ModelPath.layout_reader,
        ModelPath.slanet_plus
    ]
    download_finish_path = ""
    for model_path in model_paths:
        logger.info(f"Downloading model: {model_path}")
        download_finish_path = auto_download_and_get_model_root_path(model_path, repo_mode='pipeline')
    logger.info(f"Pipeline models downloaded successfully to: {download_finish_path}")
    configure_model(download_finish_path, "pipeline")


@click.command()
@click.option(
    '-s',
    '--source',
    'model_source',
    type=click.Choice(['huggingface', 'modelscope']),
    help="""
        The source of the model repository. 
        """,
    default=None,
)
@click.option(
    '-m',
    '--model_type',
    'model_type',
    type=click.Choice(['pipeline']),
    help="""
        The type of the model to download.
        """,
    default='pipeline',
)
def download_models(model_source, model_type):
    """Download DocuLook pipeline model files."""
    if model_source is None:
        model_source = click.prompt(
            "Please select the model download source: ",
            type=click.Choice(['huggingface', 'modelscope']),
            default='huggingface'
        )

    if os.getenv('DOCULOOK_MODEL_SOURCE', None) is None:
        os.environ['DOCULOOK_MODEL_SOURCE'] = model_source

    logger.info(f"Downloading {model_type} model from {os.getenv('DOCULOOK_MODEL_SOURCE', None)}...")

    try:
        download_pipeline_models()
    except Exception as e:
        logger.exception(f"An error occurred while downloading models: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    download_models()
