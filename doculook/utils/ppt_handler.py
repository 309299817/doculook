#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PowerPoint文档处理器
负责调用PPT直接处理器，管理OCR和文件输出
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

from doculook.utils.ppt_direct_processor import process_ppt_document_directly, is_ppt_document


def process_image_with_ocr(image_data: bytes, lang: str = "ch", image_name: str = "图片", enhance_for_text: bool = False) -> str:
    """
    对图片进行OCR识别
    
    Args:
        image_data: 图片字节数据
        lang: OCR语言
        image_name: 图片名称（用于日志）
        enhance_for_text: 是否增强文本检测
        
    Returns:
        str: OCR识别结果
    """
    try:
        from doculook.backend.pipeline.model_init import AtomModelSingleton
        from doculook.utils.ocr_utils import check_img
        import cv2
        import numpy as np
        from PIL import Image
        
        # 加载OCR模型
        atom_model_manager = AtomModelSingleton()
        ocr_model = atom_model_manager.get_atom_model(
            atom_model_name='ocr',
            det_db_box_thresh=0.3,
            lang=lang
        )
        
        # 导入必要的模块
        import io
        
        # 转换图片格式
        img = check_img(image_data)
        if img is None:
            logger.warning(f"{image_name} 图片解码失败")
            return f"[{image_name}: 无法读取图片数据]"
        
        # 预处理图片数据
        processed_image_data = image_data
        if enhance_for_text:
            try:
                # 尝试图片增强处理
                pil_img = Image.open(io.BytesIO(image_data))
                
                # 简单的对比度和亮度调整
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Contrast(pil_img)
                pil_img = enhancer.enhance(1.2)  # 增强对比度
                enhancer = ImageEnhance.Brightness(pil_img)
                pil_img = enhancer.enhance(1.1)  # 稍微增加亮度
                
                # 转换回字节
                buffer = io.BytesIO()
                pil_img.save(buffer, format='PNG')
                processed_image_data = buffer.getvalue()
                
                # 重新解码
                img = check_img(processed_image_data)
                
                logger.debug(f"{image_name} 应用了OCR增强处理")
            except Exception as e:
                logger.debug(f"{image_name} 增强处理失败，使用原图: {e}")
        
        # 执行OCR
        ocr_res_list = ocr_model.ocr(img, det=True, rec=True, tqdm_enable=False)
        
        # 如果启用了增强且初次OCR无结果，尝试更低的阈值
        if enhance_for_text and (not ocr_res_list or not ocr_res_list[0]):
            logger.debug(f"{image_name} 尝试更低的检测阈值")
            ocr_model_low_thresh = atom_model_manager.get_atom_model(
                atom_model_name='ocr',
                det_db_box_thresh=0.1,  # 更低的阈值
                lang=lang
            )
            ocr_res_list = ocr_model_low_thresh.ocr(img, det=True, rec=True, tqdm_enable=False)
        
        # 提取文本
        if ocr_res_list and ocr_res_list[0]:
            texts = []
            for line in ocr_res_list[0]:
                if line and len(line) >= 2:
                    text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                    confidence = line[1][1] if isinstance(line[1], (list, tuple)) and len(line[1]) > 1 else 1.0
                    if confidence > 0.5:  # 只保留置信度较高的文本
                        texts.append(text)
            
            ocr_result = '\n'.join(texts) if texts else ""
            logger.debug(f"{image_name} OCR识别完成，共识别 {len(texts)} 行文字")
            return ocr_result
        else:
            logger.debug(f"{image_name} OCR未识别到任何内容")
            return ""
            
    except Exception as e:
        logger.error(f"{image_name} OCR处理失败: {e}")
        return f"{image_name} OCR处理失败: {str(e)}"


def handle_ppt_document_directly(file_path: Path, output_dir: Path, lang: str = "ch") -> bool:
    """
    直接处理PowerPoint文档
    
    Args:
        file_path: PowerPoint文档路径
        output_dir: 输出目录
        lang: OCR语言
        
    Returns:
        bool: 是否处理成功
    """
    try:
        logger.info(f"开始直接处理PowerPoint文档: {file_path.name}")
        
        # 检查文件是否为PowerPoint文档
        if not is_ppt_document(file_path):
            logger.error(f"文件不是PowerPoint文档: {file_path}")
            return False
        
        # 读取文档内容
        with open(file_path, 'rb') as f:
            ppt_bytes = f.read()
        
        # 处理PowerPoint文档
        ppt_result = process_ppt_document_directly(ppt_bytes)
        
        # 设置输出文件名
        file_name = file_path.stem
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存图片文件并进行OCR
        markdown_content = ppt_result["markdown_content"]
        
        if ppt_result["has_images"]:
            logger.info(f"开始处理 {ppt_result['image_count']} 张图片的OCR识别")
            
            # 创建图片目录
            images_dir = output_dir / "images"
            images_dir.mkdir(exist_ok=True)
            
            # 设置数据写入器
            from doculook.data.data_reader_writer.filebase import FileBasedDataWriter
            img_writer = FileBasedDataWriter(str(images_dir))
            
            # 保存图片并进行OCR
            ocr_results = {}
            image_metadata = ppt_result.get("image_metadata", [])
            
            for i, image_data in enumerate(ppt_result["images"]):
                try:
                    # 获取图片格式信息
                    image_format = "png"  # 默认格式
                    if i < len(image_metadata):
                        image_format = image_metadata[i].get('format', 'png')
                    
                    # 保存图片文件（使用正确的格式）
                    image_filename = f"ppt_image_{i+1}.{image_format}"
                    img_writer.write(image_filename, image_data)
                    logger.info(f"保存图片: {image_filename}, 大小: {len(image_data)} bytes")
                    
                    # 进行OCR识别
                    ocr_result = process_image_with_ocr(
                        image_data,
                        lang=lang,
                        image_name=f"图片{i+1}",
                        enhance_for_text=True
                    )
                    
                    ocr_results[i+1] = ocr_result
                    logger.info(f"图片 {i+1} OCR完成，识别文字长度: {len(ocr_result)}")
                    
                except Exception as e:
                    logger.warning(f"处理图片 {i+1} 失败: {e}")
                    ocr_results[i+1] = f"图片 {i+1} 处理失败: {str(e)}"
            
            # 在Markdown内容中替换图片引用为OCR结果
            for image_num, ocr_text in ocr_results.items():
                # 需要匹配实际的图片格式
                image_format = "png"  # 默认格式
                if image_num - 1 < len(image_metadata):
                    image_format = image_metadata[image_num - 1].get('format', 'png')
                
                image_placeholder = f"![图片{image_num}](images/ppt_image_{image_num}.{image_format})"
                logger.debug(f"查找图片占位符: {image_placeholder}")
                logger.debug(f"OCR识别结果长度: {len(ocr_text) if ocr_text else 0}")
                
                if image_placeholder in markdown_content and ocr_text.strip():
                    # 替换为图片+OCR文本的组合
                    replacement = f"{image_placeholder}\n\n**图片{image_num}识别内容:**\n{ocr_text}\n"
                    markdown_content = markdown_content.replace(image_placeholder, replacement)
                    logger.info(f"成功替换图片{image_num}的OCR结果")
                else:
                    if image_placeholder not in markdown_content:
                        logger.warning(f"未找到图片占位符: {image_placeholder}")
                        # 尝试查找其他可能的格式
                        for alt_format in ['png', 'jpg', 'jpeg', 'gif', 'bmp']:
                            alt_placeholder = f"![图片{image_num}](images/ppt_image_{image_num}.{alt_format})"
                            if alt_placeholder in markdown_content:
                                logger.info(f"找到替代格式的占位符: {alt_placeholder}")
                                replacement = f"{alt_placeholder}\n\n**图片{image_num}识别内容:**\n{ocr_text}\n"
                                markdown_content = markdown_content.replace(alt_placeholder, replacement)
                                break
                    if not ocr_text.strip():
                        logger.warning(f"图片{image_num}的OCR结果为空")
        
        # 保存最终的Markdown文件
        from doculook.data.data_reader_writer.filebase import FileBasedDataWriter
        md_writer = FileBasedDataWriter(str(output_dir))
        md_writer.write(f"{file_name}.md", markdown_content.encode('utf-8'))
        
        # 保存处理结果的JSON（移除不能序列化的字节数据）
        result_json = {
            "ppt_info": {
                "slide_count": len(ppt_result["slides"]),
                "image_count": ppt_result["image_count"],
                "has_images": ppt_result["has_images"],
                "has_tables": ppt_result["has_tables"],
                "slides": ppt_result["slides"]
            },
            "processing_info": {
                "file_name": file_name,
                "processing_method": "direct_ppt_processing",
                "ocr_language": lang,
                "total_images_processed": ppt_result["image_count"]
            }
        }
        
        md_writer.write(f"{file_name}_middle.json", json.dumps(result_json, ensure_ascii=False, indent=2).encode('utf-8'))
        
        logger.info(f"PowerPoint文档 {file_path.name} 处理完成")
        logger.info(f"输出文件: {output_dir / f'{file_name}.md'}")
        logger.info(f"处理结果: {len(ppt_result['slides'])}张幻灯片, {ppt_result['image_count']}张图片")
        
        return True
        
    except Exception as e:
        logger.error(f"PowerPoint文档处理失败: {e}")
        return False
