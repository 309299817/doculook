# Copyright (c) Opendatalab. All rights reserved.
import os
import click
from pathlib import Path
from loguru import logger

from doculook.utils.cli_parser import arg_parse
from doculook.utils.config_reader import get_device
from doculook.utils.model_utils import get_vram
from ..version import __version__
from .common import do_parse, read_fn, pdf_suffixes, image_suffixes, office_suffixes

@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
@click.version_option(__version__,
                      '--version',
                      '-v',
                      help='display the version and exit')
@click.option(
    '-p',
    '--path',
    'input_path',
    type=click.Path(exists=True),
    required=True,
    help='local filepath or directory. support pdf, png, jpg, jpeg, docx, doc, xlsx, xls, pptx, ppt files',
)
@click.option(
    '-o',
    '--output',
    'output_dir',
    type=click.Path(),
    required=True,
    help='output local directory',
)
@click.option(
    '-m',
    '--method',
    'method',
    type=click.Choice(['auto', 'txt', 'ocr']),
    help="""the method for parsing pdf:
    auto: Automatically determine the method based on the file type.
    txt: Use text extraction method.
    ocr: Use OCR method for image-based PDFs.
    Without method specified, 'auto' will be used by default.
    Adapted only for the case where the backend is set to "pipeline".""",
    default='auto',
)
@click.option(
    '-b',
    '--backend',
    'backend',
    type=click.Choice(['pipeline', 'vlm-transformers', 'vlm-sglang-engine', 'vlm-sglang-client']),
    help="""the backend for parsing pdf:
    pipeline: More general.
    vlm-transformers: More general.
    vlm-sglang-engine: Faster(engine).
    vlm-sglang-client: Faster(client).
    without method specified, pipeline will be used by default.""",
    default='pipeline',
)
@click.option(
    '-l',
    '--lang',
    'lang',
    type=click.Choice(['ch', 'ch_server', 'ch_lite', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka',
                       'latin', 'arabic', 'east_slavic', 'cyrillic', 'devanagari']),
    help="""
    Input the languages in the pdf (if known) to improve OCR accuracy.  Optional.
    Without languages specified, 'ch' will be used by default.
    Adapted only for the case where the backend is set to "pipeline".
    """,
    default='ch',
)
@click.option(
    '-u',
    '--url',
    'server_url',
    type=str,
    help="""
    When the backend is `sglang-client`, you need to specify the server_url, for example:`http://127.0.0.1:30000`
    """,
    default=None,
)
@click.option(
    '-s',
    '--start',
    'start_page_id',
    type=int,
    help='The starting page for PDF parsing, beginning from 0.',
    default=0,
)
@click.option(
    '-e',
    '--end',
    'end_page_id',
    type=int,
    help='The ending page for PDF parsing, beginning from 0.',
    default=None,
)
@click.option(
    '-f',
    '--formula',
    'formula_enable',
    type=bool,
    help='Enable formula parsing. Default is True. Adapted only for the case where the backend is set to "pipeline".',
    default=True,
)
@click.option(
    '-t',
    '--table',
    'table_enable',
    type=bool,
    help='Enable table parsing. Default is True. Adapted only for the case where the backend is set to "pipeline".',
    default=True,
)
@click.option(
    '-d',
    '--device',
    'device_mode',
    type=str,
    help='Device mode for model inference, e.g., "cpu", "cuda", "cuda:0", "npu", "npu:0", "mps". Adapted only for the case where the backend is set to "pipeline". ',
    default=None,
)
@click.option(
    '--vram',
    'virtual_vram',
    type=int,
    help='Upper limit of GPU memory occupied by a single process. Adapted only for the case where the backend is set to "pipeline". ',
    default=None,
)
@click.option(
    '--source',
    'model_source',
    type=click.Choice(['huggingface', 'modelscope', 'local']),
    help="""
    The source of the model repository. Default is 'huggingface'.
    """,
    default='huggingface',
)


def main(
        ctx,
        input_path, output_dir, method, backend, lang, server_url,
        start_page_id, end_page_id, formula_enable, table_enable,
        device_mode, virtual_vram, model_source, **kwargs
):

    kwargs.update(arg_parse(ctx))

    if not backend.endswith('-client'):
        def get_device_mode() -> str:
            if device_mode is not None:
                return device_mode
            else:
                return get_device()
        if os.getenv('MINERU_DEVICE_MODE', None) is None:
            os.environ['MINERU_DEVICE_MODE'] = get_device_mode()

        def get_virtual_vram_size() -> int:
            if virtual_vram is not None:
                return virtual_vram
            if get_device_mode().startswith("cuda") or get_device_mode().startswith("npu"):
                return round(get_vram(get_device_mode()))
            return 1
        if os.getenv('MINERU_VIRTUAL_VRAM_SIZE', None) is None:
            os.environ['MINERU_VIRTUAL_VRAM_SIZE']= str(get_virtual_vram_size())

        if os.getenv('MINERU_MODEL_SOURCE', None) is None:
            os.environ['MINERU_MODEL_SOURCE'] = model_source

    os.makedirs(output_dir, exist_ok=True)

    def parse_doc(path_list: list[Path]):
        try:
            file_name_list = []
            pdf_bytes_list = []
            lang_list = []
            original_paths = []
            has_word_docs = False
            
            for path in path_list:
                file_name = str(Path(path).stem)
                file_name_list.append(file_name)
                lang_list.append(lang)
                original_paths.append(str(path))
                
                # 检查是否为Word文档
                if path.suffix.lower() in ['.docx', '.doc']:
                    has_word_docs = True
                    # 对于Word文档，我们仍然需要准备PDF字节以防直接处理失败
                    pdf_bytes = read_fn(path)
                    pdf_bytes_list.append(pdf_bytes)
                else:
                    pdf_bytes = read_fn(path)
                    pdf_bytes_list.append(pdf_bytes)
            
            do_parse(
                output_dir=output_dir,
                pdf_file_names=file_name_list,
                pdf_bytes_list=pdf_bytes_list,
                p_lang_list=lang_list,
                backend=backend,
                parse_method=method,
                formula_enable=formula_enable,
                table_enable=table_enable,
                server_url=server_url,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
                word_direct_processing=has_word_docs,
                original_file_paths=original_paths if has_word_docs else None,
                **kwargs,
            )
        except Exception as e:
            logger.exception(e)

    if os.path.isdir(input_path):
        doc_path_list = []
        for doc_path in Path(input_path).glob('*'):
            if doc_path.suffix in pdf_suffixes + image_suffixes + office_suffixes:
                doc_path_list.append(doc_path)
        parse_doc(doc_path_list)
    else:
        parse_doc([Path(input_path)])

if __name__ == '__main__':
    main()
