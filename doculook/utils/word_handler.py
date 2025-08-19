# Copyright (c) Opendatalab. All rights reserved.
"""
Word文档专用处理器
实现Word文档的直接处理，不经过PDF转换
"""

import os
import json
import copy
from pathlib import Path
from loguru import logger
from typing import Dict, List

def handle_word_document_directly(
    word_path: Path,
    output_dir: str,
    lang: str = "ch",
    parse_method: str = "auto",
    word_image_backend: str = "ppocr",
    **kwargs
) -> Dict:
    """
    直接处理Word文档的完整流程
    
    Args:
        word_path: Word文档路径
        output_dir: 输出目录
        lang: OCR语言设置
        parse_method: 解析方法
        **kwargs: 其他参数
        
    Returns:
        Dict: 处理结果
    """
    try:
        from doculook.utils.word_direct_processor import process_word_document_directly, save_word_images_for_ocr
        from doculook.data.data_reader_writer import FileBasedDataWriter
        
        # 读取Word文档
        with open(word_path, "rb") as f:
            word_bytes = f.read()
        
        # 直接提取Word内容
        logger.info(f"开始直接处理Word文档: {word_path.name}")
        word_result = process_word_document_directly(word_bytes)
        
        # 设置输出目录
        file_name = word_path.stem
        local_md_dir = os.path.join(output_dir, file_name, parse_method)
        local_image_dir = os.path.join(local_md_dir, "images")
        os.makedirs(local_image_dir, exist_ok=True)
        os.makedirs(local_md_dir, exist_ok=True)
        
        md_writer = FileBasedDataWriter(local_md_dir)
        image_writer = FileBasedDataWriter(local_image_dir)
        
        # 处理文本内容
        markdown_content = word_result["markdown_content"]
        
        # 处理图片内容（如果有）
        if word_result["has_images"]:
            logger.info(f"处理Word文档中的 {word_result['image_count']} 张图片")
            
            # 保存图片并进行OCR，然后替换Markdown中的图片引用
            ocr_results = {}
            for i, image_data in enumerate(word_result["images"]):
                try:
                    # 保存图片文件
                    image_filename = f"word_image_{i+1}.png"
                    image_writer.write(image_filename, image_data)
                    
                    # 对图片进行OCR处理（可选后端）
                    use_llm = False
                    if word_image_backend == "llm":
                        use_llm = True
                    elif word_image_backend == "auto":
                        try:
                            from doculook.utils.config_reader import get_vision_llm_config
                            vcfg_probe = get_vision_llm_config()
                            if vcfg_probe and vcfg_probe.get('api_key'):
                                use_llm = True
                        except Exception:
                            use_llm = False

                    if use_llm:
                        from doculook.utils.config_reader import get_vision_llm_config
                        from doculook.backend.vlm.vision_predictor import GenericVisionPredictor
                        vcfg = get_vision_llm_config()
                        if not vcfg or not vcfg.get('api_key'):
                            raise RuntimeError("使用 llm 识别需要在配置中提供 api_key")
                        predictor = GenericVisionPredictor(
                            api_key=vcfg['api_key'],
                            base_url=vcfg.get('base_url', 'https://api.openai.com/v1'),
                            model=vcfg.get('model', 'gpt-4o-mini'),
                            temperature=vcfg.get('temperature', 0.0),
                            top_p=vcfg.get('top_p', 0.8),
                            max_new_tokens=vcfg.get('max_tokens', 16384),
                        )
                        # 确保图片是可被Qwen-VL接受的JPEG/PNG字节
                        try:
                            from PIL import Image
                            import io
                            pil = Image.open(io.BytesIO(image_data))
                            bio = io.BytesIO()
                            # 转 jpeg，避免不支持的格式
                            pil.convert('RGB').save(bio, format='JPEG', quality=90, optimize=True)
                            qwen_image = bio.getvalue()
                        except Exception:
                            qwen_image = image_data
                        try:
                            ocr_result = predictor.predict(image=qwen_image, prompt="请识别图片中的可读文本，按从上到下、从左到右顺序输出")
                        except Exception as qe:
                            logger.warning(f"LLM 识别失败，回退 ppocr: {qe}")
                            ocr_result = process_image_with_ocr(
                                image_data,
                                lang=lang,
                                image_name=f"图片{i+1}"
                            )
                    else:
                        ocr_result = process_image_with_ocr(
                            image_data,
                            lang=lang,
                            image_name=f"图片{i+1}",
                            enhance_for_text=True
                        )
                    
                    # 存储OCR结果，用于后续替换
                    ocr_results[i+1] = ocr_result
                    
                    logger.info(f"图片 {i+1} OCR完成，识别文字长度: {len(ocr_result)}")
                    
                except Exception as e:
                    logger.warning(f"处理图片 {i+1} 失败: {e}")
                    ocr_results[i+1] = f"图片 {i+1} 处理失败"
            
            # 在Markdown内容中的相应位置替换图片引用为OCR结果
            for image_num, ocr_text in ocr_results.items():
                image_placeholder = f"![图片{image_num}](images/word_image_{image_num}.png)"
                if image_placeholder in markdown_content and ocr_text.strip():
                    # 替换为图片+OCR文本的组合
                    replacement = f"{image_placeholder}\n\n**图片{image_num}识别内容:**\n{ocr_text}\n"
                    markdown_content = markdown_content.replace(image_placeholder, replacement)
        
        # 保存最终的Markdown文件
        md_writer.write(f"{file_name}.md", markdown_content.encode('utf-8'))
        
        # 保存处理结果的JSON
        result_json = {
            "pdf_info": {
                "pages": [
                    {
                        "page_no": 0,
                        "height": 1000,
                        "width": 800,
                        "elements": []
                    }
                ]
            },
            "word_direct_processing": True,
            "text_content": word_result["text_content"],
            "image_count": word_result["image_count"],
            "processing_method": "direct_word_processing"
        }
        
        md_writer.write(f"{file_name}_middle.json", 
                             json.dumps(result_json, ensure_ascii=False, indent=4).encode('utf-8'))
        
        logger.info(f"Word文档处理完成，输出目录: {local_md_dir}")
        
        return {
            "markdown_content": markdown_content,
            "text_length": len(word_result["text_content"]),
            "image_count": word_result["image_count"],
            "output_dir": local_md_dir
        }
        
    except Exception as e:
        logger.error(f"Word文档直接处理失败: {e}")
        raise


def process_image_with_ocr(image_data: bytes, lang: str = "ch", image_name: str = "图片", enhance_for_text: bool = False) -> str:
    """
    对单张图片进行OCR处理（直接使用本地 ppocr，避免中转PDF带来的降质）。
    """
    try:
        enhance_info = "增强模式" if enhance_for_text else "标准模式"
        logger.info(f"对{image_name}进行OCR处理（语言: {lang}，后端: ppocr，{enhance_info}）")
        
        from doculook.backend.pipeline.model_init import AtomModelSingleton
        from doculook.utils.ocr_utils import check_img
        import numpy as np

        # 如果启用增强模式，先进行图像增强
        processed_image_data = image_data
        if enhance_for_text:
            try:
                from doculook.utils.enhanced_text_detector import enhance_image_for_ocr
                processed_image_data = enhance_image_for_ocr(image_data, method="auto")
                logger.debug(f"{image_name} 应用了OCR增强处理")
            except Exception as e:
                logger.debug(f"{image_name} 增强处理失败，使用原图: {e}")

        # 解码为OpenCV图像
        img = check_img(processed_image_data)
        if img is None:
            return f"[{image_name}: 无法读取图片数据]"

        # 获取OCR模型并执行检测+识别
        atom_model_manager = AtomModelSingleton()
        ocr_model = atom_model_manager.get_atom_model(
            atom_model_name='ocr',
            det_db_box_thresh=0.3,
            lang=lang
        )
        ocr_res_list = ocr_model.ocr(img, det=True, rec=True, tqdm_enable=False)
        
        # 如果增强模式下仍无结果，尝试更低的阈值
        if enhance_for_text and (not ocr_res_list or not ocr_res_list[0]):
            logger.debug(f"{image_name} 尝试更低的检测阈值")
            ocr_model_low_thresh = atom_model_manager.get_atom_model(
                atom_model_name='ocr',
                det_db_box_thresh=0.1,  # 更低的阈值
                lang=lang
            )
            ocr_res_list = ocr_model_low_thresh.ocr(img, det=True, rec=True, tqdm_enable=False)
        
        if not ocr_res_list:
            return f"[{image_name}: OCR无结果]"

        # 仅一张图，取第1项
        ocr_res = ocr_res_list[0]
        if not ocr_res:
            return f"[{image_name}: OCR无结果]"

        lines = []
        for box, rec in ocr_res:
            if isinstance(rec, (list, tuple)) and len(rec) >= 1:
                text = rec[0]
                if isinstance(text, str) and text.strip():
                    lines.append(text.strip())

        text_out = "\n".join(lines).strip()
        if not text_out:
            return ""
        
        logger.debug(f"{image_name} OCR识别成功，共 {len(lines)} 行文字")
        return text_out

    except Exception as e:
        logger.error(f"图片OCR处理失败: {e}")
        return f"[{image_name} OCR处理失败: {str(e)}]"


def is_word_document(file_path: Path) -> bool:
    """
    检查文件是否为Word文档
    
    Args:
        file_path: 文件路径
        
    Returns:
        bool: 是否为Word文档
    """
    return file_path.suffix.lower() in ['.docx', '.doc']


if __name__ == "__main__":
    # 测试代码
    logger.info("Word文档处理器加载完成")
