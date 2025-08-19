# Copyright (c) Opendatalab. All rights reserved.
import io
import json
import os
import copy
from pathlib import Path

import pypdfium2 as pdfium
from loguru import logger

from doculook.data.data_reader_writer import FileBasedDataWriter
from doculook.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from doculook.utils.enum_class import MakeMode
from doculook.utils.pdf_image_tools import images_bytes_to_pdf_bytes
"""
已去除 VLM 后端依赖，仅保留 pipeline。保留的 LLM 能力用于图片裁剪文字识别（在 pipeline 的后置 OCR 中）。
"""

pdf_suffixes = [".pdf"]
image_suffixes = [".png", ".jpeg", ".jpg", ".webp", ".gif"]
office_suffixes = [".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt"]


def read_fn(path):
    if not isinstance(path, Path):
        path = Path(path)
    with open(str(path), "rb") as input_file:
        file_bytes = input_file.read()
        if path.suffix in image_suffixes:
            return images_bytes_to_pdf_bytes(file_bytes)
        elif path.suffix in pdf_suffixes:
            return file_bytes
        elif path.suffix in office_suffixes:
            # 导入Office转换器
            from doculook.utils.office_converter import office_to_pdf_bytes
            return office_to_pdf_bytes(file_bytes, path.suffix)
        else:
            raise Exception(f"Unknown file suffix: {path.suffix}")


def prepare_env(output_dir, pdf_file_name, parse_method):
    local_md_dir = str(os.path.join(output_dir, pdf_file_name, parse_method))
    local_image_dir = os.path.join(str(local_md_dir), "images")
    os.makedirs(local_image_dir, exist_ok=True)
    os.makedirs(local_md_dir, exist_ok=True)
    return local_image_dir, local_md_dir


def convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id=0, end_page_id=None):

    # 从字节数据加载PDF
    pdf = pdfium.PdfDocument(pdf_bytes)

    # 确定结束页
    end_page_id = end_page_id if end_page_id is not None and end_page_id >= 0 else len(pdf) - 1
    if end_page_id > len(pdf) - 1:
        logger.warning("end_page_id is out of range, use pdf_docs length")
        end_page_id = len(pdf) - 1

    # 创建一个新的PDF文档
    output_pdf = pdfium.PdfDocument.new()

    # 选择要导入的页面索引
    page_indices = list(range(start_page_id, end_page_id + 1))

    # 从原PDF导入页面到新PDF
    output_pdf.import_pages(pdf, page_indices)

    # 将新PDF保存到内存缓冲区
    output_buffer = io.BytesIO()
    output_pdf.save(output_buffer)

    # 获取字节数据
    output_bytes = output_buffer.getvalue()

    pdf.close()  # 关闭原PDF文档以释放资源
    output_pdf.close()  # 关闭新PDF文档以释放资源

    return output_bytes


def _prepare_pdf_bytes(pdf_bytes_list, start_page_id, end_page_id):
    """准备处理PDF字节数据"""
    result = []
    for pdf_bytes in pdf_bytes_list:
        new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
        result.append(new_pdf_bytes)
    return result


def _process_output(
        pdf_info,
        pdf_bytes,
        pdf_file_name,
        local_md_dir,
        local_image_dir,
        md_writer,
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_orig_pdf,
        f_dump_md,
        f_dump_content_list,
        f_dump_middle_json,
        f_dump_model_output,
        f_make_md_mode,
        middle_json,
        model_output=None,
):
    from doculook.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
    """处理输出文件"""
    if f_draw_layout_bbox:
        draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")

    if f_draw_span_bbox:
        draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")

    if f_dump_orig_pdf:
        md_writer.write(
            f"{pdf_file_name}_origin.pdf",
            pdf_bytes,
        )

    image_dir = str(os.path.basename(local_image_dir))

    if f_dump_md:
        md_content_str = pipeline_union_make(pdf_info, f_make_md_mode, image_dir)
        md_writer.write(
            f"{pdf_file_name}.md",
            md_content_str.encode('utf-8'),
        )

    if f_dump_content_list:
        content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
        md_writer.write(
            f"{pdf_file_name}_content_list.json",
            json.dumps(content_list, ensure_ascii=False, indent=4).encode('utf-8'),
        )

    if f_dump_middle_json:
        md_writer.write(
            f"{pdf_file_name}_middle.json",
            json.dumps(middle_json, ensure_ascii=False, indent=4).encode('utf-8'),
        )

    if f_dump_model_output:
        md_writer.write(
            f"{pdf_file_name}_model.json",
            json.dumps(model_output, ensure_ascii=False, indent=4).encode('utf-8'),
        )

    logger.info(f"local output dir is {local_md_dir}")


def _process_pipeline(
        output_dir,
        pdf_file_names,
        pdf_bytes_list,
        p_lang_list,
        parse_method,
        p_formula_enable,
        p_table_enable,
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_md,
        f_dump_middle_json,
        f_dump_model_output,
        f_dump_orig_pdf,
        f_dump_content_list,
        f_make_md_mode,
        image_text_backend: str = "ppocr",
):
    """处理pipeline后端逻辑"""
    from doculook.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
    from doculook.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze

    infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = (
        pipeline_doc_analyze(
            pdf_bytes_list, p_lang_list, parse_method=parse_method,
            formula_enable=p_formula_enable, table_enable=p_table_enable
        )
    )

    for idx, model_list in enumerate(infer_results):
        model_json = copy.deepcopy(model_list)
        pdf_file_name = pdf_file_names[idx]
        local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
        image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

        images_list = all_image_lists[idx]
        pdf_doc = all_pdf_docs[idx]
        _lang = lang_list[idx]
        _ocr_enable = ocr_enabled_list[idx]

        middle_json = pipeline_result_to_middle_json(
            model_list, images_list, pdf_doc, image_writer,
            _lang, _ocr_enable, p_formula_enable, image_text_backend=image_text_backend
        )

        pdf_info = middle_json["pdf_info"]
        pdf_bytes = pdf_bytes_list[idx]

        _process_output(
            pdf_info, pdf_bytes, pdf_file_name, local_md_dir, local_image_dir,
            md_writer, f_draw_layout_bbox, f_draw_span_bbox, f_dump_orig_pdf,
            f_dump_md, f_dump_content_list, f_dump_middle_json, f_dump_model_output,
            f_make_md_mode, middle_json, model_json
        )


async def _async_process_vlm(*args, **kwargs):
    raise RuntimeError("VLM backend has been removed; pipeline is the only supported backend.")


def _process_vlm(*args, **kwargs):
    raise RuntimeError("VLM backend has been removed; pipeline is the only supported backend.")


def do_parse(
        output_dir,
        pdf_file_names: list[str],
        pdf_bytes_list: list[bytes],
        p_lang_list: list[str],
        backend="pipeline",
        parse_method="auto",
        formula_enable=True,
        table_enable=True,
        server_url=None,
        f_draw_layout_bbox=True,
        f_draw_span_bbox=True,
        f_dump_md=True,
        f_dump_middle_json=True,
        f_dump_model_output=True,
        f_dump_orig_pdf=True,
        f_dump_content_list=True,
        f_make_md_mode=MakeMode.MM_MD,
        start_page_id=0,
        end_page_id=None,
        word_direct_processing=False,
        original_file_paths=None,
        image_ocr_backend: str = "ppocr",
        **kwargs,
):
    # 检查是否有Office文档需要直接处理
    if word_direct_processing and original_file_paths:
        from doculook.utils.word_handler import handle_word_document_directly, is_word_document
        from doculook.utils.excel_handler import handle_excel_document_directly, is_excel_document
        from doculook.utils.ppt_handler import handle_ppt_document_directly, is_ppt_document
        
        processed_files = []
        for i, (file_name, file_path, lang) in enumerate(zip(pdf_file_names, original_file_paths, p_lang_list)):
            if file_path:
                file_path_obj = Path(file_path)
                
                # Word文档直接处理
                if is_word_document(file_path_obj):
                    logger.info(f"检测到Word文档，使用直接处理模式: {file_path}")
                    try:
                        result = handle_word_document_directly(
                            word_path=file_path_obj,
                            output_dir=output_dir,
                            lang=lang,
                            parse_method=parse_method,
                            word_image_backend=image_ocr_backend,
                            **kwargs
                        )
                        processed_files.append(file_name)
                        logger.info(f"Word文档 {file_name} 直接处理完成")
                        continue
                    except Exception as e:
                        logger.error(f"Word文档直接处理失败，回退到PDF转换模式: {e}")
                
                # Excel文档直接处理
                elif is_excel_document(file_path_obj):
                    logger.info(f"检测到Excel文档，使用直接处理模式: {file_path}")
                    try:
                        result = handle_excel_document_directly(
                            excel_path=file_path_obj,
                            output_dir=output_dir,
                            lang=lang,
                            parse_method=parse_method,
                            excel_image_backend=image_ocr_backend,
                            **kwargs
                        )
                        processed_files.append(file_name)
                        logger.info(f"Excel文档 {file_name} 直接处理完成")
                        continue
                    except Exception as e:
                        logger.error(f"Excel文档直接处理失败，回退到PDF转换模式: {e}")
                
                # PowerPoint文档直接处理
                elif is_ppt_document(file_path_obj):
                    logger.info(f"检测到PowerPoint文档，使用直接处理模式: {file_path}")
                    try:
                        result = handle_ppt_document_directly(
                            file_path=file_path_obj,
                            output_dir=output_dir,
                            lang=lang
                        )
                        
                        if result:
                            processed_files.append(file_name)
                            logger.info(f"PowerPoint文档 {file_name} 直接处理完成")
                            continue
                    except Exception as e:
                        logger.error(f"PowerPoint文档直接处理失败，回退到PDF转换模式: {e}")
        
        # 如果所有文件都是Office文档且都处理完成，直接返回
        if len(processed_files) == len(pdf_file_names):
            logger.info("所有Office文档已完成直接处理")
            return
    
    # 预处理PDF字节数据
    pdf_bytes_list = _prepare_pdf_bytes(pdf_bytes_list, start_page_id, end_page_id)

    if backend == "pipeline":
        _process_pipeline(
            output_dir, pdf_file_names, pdf_bytes_list, p_lang_list,
            parse_method, formula_enable, table_enable,
            f_draw_layout_bbox, f_draw_span_bbox, f_dump_md, f_dump_middle_json,
            f_dump_model_output, f_dump_orig_pdf, f_dump_content_list, f_make_md_mode,
            image_text_backend=image_ocr_backend
        )
    else:
        if backend.startswith("vlm-"):
            backend = backend[4:]
        elif backend == "qwen-vl":
            # 保留兼容但引导到通用 llm
            from doculook.utils.config_reader import get_vision_llm_config
            vcfg = get_vision_llm_config()
            if vcfg:
                kwargs.update({
                    'api_key': vcfg.get('api_key'),
                    'base_url': vcfg.get('base_url', 'https://api.openai.com/v1'),
                    'model': vcfg.get('model', 'gpt-4o-mini'),
                    'temperature': vcfg.get('temperature', 0.0),
                    'top_p': vcfg.get('top_p', 0.8),
                    'max_new_tokens': vcfg.get('max_tokens', 16384),
                })

        os.environ['DOCULOOK_VLM_FORMULA_ENABLE'] = str(formula_enable)
        os.environ['DOCULOOK_VLM_TABLE_ENABLE'] = str(table_enable)

        _process_vlm(
            output_dir, pdf_file_names, pdf_bytes_list, backend,
            f_draw_layout_bbox, f_draw_span_bbox, f_dump_md, f_dump_middle_json,
            f_dump_model_output, f_dump_orig_pdf, f_dump_content_list, f_make_md_mode,
            server_url, **kwargs,
        )


async def aio_do_parse(
        output_dir,
        pdf_file_names: list[str],
        pdf_bytes_list: list[bytes],
        p_lang_list: list[str],
        backend="pipeline",
        parse_method="auto",
        formula_enable=True,
        table_enable=True,
        server_url=None,
        f_draw_layout_bbox=True,
        f_draw_span_bbox=True,
        f_dump_md=True,
        f_dump_middle_json=True,
        f_dump_model_output=True,
        f_dump_orig_pdf=True,
        f_dump_content_list=True,
        f_make_md_mode=MakeMode.MM_MD,
        start_page_id=0,
        end_page_id=None,
        word_direct_processing=False,
        original_file_paths=None,
        image_ocr_backend: str = "ppocr",
        **kwargs,
):
    # 检查是否有Office文档需要直接处理
    if word_direct_processing and original_file_paths:
        from doculook.utils.word_handler import handle_word_document_directly, is_word_document
        from doculook.utils.excel_handler import handle_excel_document_directly, is_excel_document
        from doculook.utils.ppt_handler import handle_ppt_document_directly, is_ppt_document
        
        processed_files = []
        for i, (file_name, file_path, lang) in enumerate(zip(pdf_file_names, original_file_paths, p_lang_list)):
            if file_path:
                file_path_obj = Path(file_path)
                
                # Word文档直接处理
                if is_word_document(file_path_obj):
                    logger.info(f"检测到Word文档，使用直接处理模式: {file_path}")
                    try:
                        result = handle_word_document_directly(
                            word_path=file_path_obj,
                            output_dir=output_dir,
                            lang=lang,
                            parse_method=parse_method,
                            word_image_backend=image_ocr_backend,
                            **kwargs
                        )
                        processed_files.append(file_name)
                        logger.info(f"Word文档 {file_name} 直接处理完成")
                        
                        # 清理临时文件
                        try:
                            os.unlink(file_path)
                        except:
                            pass
                        continue
                            
                    except Exception as e:
                        logger.error(f"Word文档直接处理失败，回退到PDF转换模式: {e}")
                
                # Excel文档直接处理
                elif is_excel_document(file_path_obj):
                    logger.info(f"检测到Excel文档，使用直接处理模式: {file_path}")
                    try:
                        result = handle_excel_document_directly(
                            excel_path=file_path_obj,
                            output_dir=output_dir,
                            lang=lang,
                            parse_method=parse_method,
                            excel_image_backend=image_ocr_backend,
                            **kwargs
                        )
                        processed_files.append(file_name)
                        logger.info(f"Excel文档 {file_name} 直接处理完成")
                        
                        # 清理临时文件
                        try:
                            os.unlink(file_path)
                        except:
                            pass
                        continue
                            
                    except Exception as e:
                        logger.error(f"Excel文档直接处理失败，回退到PDF转换模式: {e}")
        
        # 如果所有文件都是Office文档且都处理完成，直接返回
        if len(processed_files) == len(pdf_file_names):
            logger.info("所有Office文档已完成直接处理")
            return
    
    # 预处理PDF字节数据
    pdf_bytes_list = _prepare_pdf_bytes(pdf_bytes_list, start_page_id, end_page_id)

    if backend == "pipeline":
        # pipeline模式暂不支持异步，使用同步处理方式
        _process_pipeline(
            output_dir, pdf_file_names, pdf_bytes_list, p_lang_list,
            parse_method, formula_enable, table_enable,
            f_draw_layout_bbox, f_draw_span_bbox, f_dump_md, f_dump_middle_json,
            f_dump_model_output, f_dump_orig_pdf, f_dump_content_list, f_make_md_mode,
            image_text_backend=image_ocr_backend
        )
    else:
        if backend.startswith("vlm-"):
            backend = backend[4:]
        elif backend == "qwen-vl":
            # qwen-vl后端不需要前缀处理，但需要加载配置
            from doculook.utils.config_reader import get_qwen_vl_config
            qwen_config = get_qwen_vl_config()
            if qwen_config:
                # 自动添加qwen-vl配置参数
                kwargs.update({
                    'api_key': qwen_config.get('api_key'),
                    'base_url': qwen_config.get('base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
                    'model': qwen_config.get('model', 'qwen-vl-max'),
                    'temperature': qwen_config.get('temperature', 0.0),
                    'top_p': qwen_config.get('top_p', 0.8),
                    'max_new_tokens': qwen_config.get('max_tokens', 16384),
                })

        os.environ['DOCULOOK_VLM_FORMULA_ENABLE'] = str(formula_enable)
        os.environ['DOCULOOK_VLM_TABLE_ENABLE'] = str(table_enable)

        await _async_process_vlm(
            output_dir, pdf_file_names, pdf_bytes_list, backend,
            f_draw_layout_bbox, f_draw_span_bbox, f_dump_md, f_dump_middle_json,
            f_dump_model_output, f_dump_orig_pdf, f_dump_content_list, f_make_md_mode,
            server_url, **kwargs,
        )



if __name__ == "__main__":
    # pdf_path = "../../demo/pdfs/demo3.pdf"
    pdf_path = "C:/Users/zhaoxiaomeng/Downloads/4546d0e2-ba60-40a5-a17e-b68555cec741.pdf"

    try:
       do_parse("./output", [Path(pdf_path).stem], [read_fn(Path(pdf_path))],["ch"],
                end_page_id=10,
                backend='vlm-huggingface'
                # backend = 'pipeline'
                )
    except Exception as e:
        logger.exception(e)

