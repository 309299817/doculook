import uuid
import os
import uvicorn
import click
from pathlib import Path
from glob import glob
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
from loguru import logger
from base64 import b64encode

from doculook.cli.common import aio_do_parse, read_fn, pdf_suffixes, image_suffixes, office_suffixes
from doculook.utils.cli_parser import arg_parse
from doculook.version import __version__

app = FastAPI(
    title="DocuLook 文档解析 API",
    description="DocuLook 支持解析 PDF、图片与 Office 文档（Word、Excel、PowerPoint）",
    version=__version__
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

def encode_image(image_path: str) -> str:
    """Encode image using base64"""
    with open(image_path, "rb") as f:
        return b64encode(f.read()).decode()


def get_infer_result(file_suffix_identifier: str, pdf_name: str, parse_dir: str) -> Optional[str]:
    """从结果文件中读取推理结果"""
    result_file_path = os.path.join(parse_dir, f"{pdf_name}{file_suffix_identifier}")
    if os.path.exists(result_file_path):
        with open(result_file_path, "r", encoding="utf-8") as fp:
            return fp.read()
    return None


@app.get("/")
async def root():
    """API根路径，返回基本信息"""
    return JSONResponse(
        status_code=200,
        content={
            "service": "DocuLook Document Parser API",
            "version": __version__,
            "description": "支持解析PDF、图片和Office文档的智能文档处理服务",
            "supported_formats": {
                "pdf": pdf_suffixes,
                "images": image_suffixes,
                "office": office_suffixes
            },
            "endpoints": {
                "parse": "/file_parse",
                "formats": "/supported_formats",
                "health": "/health",
                "docs": "/docs",
                "redoc": "/redoc"
            }
        }
    )


@app.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        # 检查Office转换器模块是否可用
        from doculook.utils.office_converter import OFFICE_SUFFIXES
        office_available = True
    except ImportError:
        office_available = False
    
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "version": __version__,
            "office_support": office_available,
            "supported_formats_count": len(pdf_suffixes) + len(image_suffixes) + len(office_suffixes)
        }
    )


@app.get("/supported_formats")
async def get_supported_formats():
    """获取支持的文件格式信息"""
    return JSONResponse(
        status_code=200,
        content={
            "version": __version__,
            "supported_formats": {
                "pdf": {
                    "extensions": pdf_suffixes,
                    "description": "PDF文档格式"
                },
                "images": {
                    "extensions": image_suffixes,
                    "description": "图片文件格式"
                },
                "office": {
                    "extensions": office_suffixes,
                    "description": "Microsoft Office文档格式",
                    "details": {
                        "word": [".docx", ".doc"],
                        "excel": [".xlsx", ".xls"],
                        "powerpoint": [".pptx", ".ppt"]
                    }
                }
            },
            "total_formats": len(pdf_suffixes) + len(image_suffixes) + len(office_suffixes)
        }
    )


@app.post(path="/file_parse",
          summary="解析文档",
          description="上传并解析PDF、图片或Office文档（Word/Excel/PowerPoint），返回Markdown格式结果")
async def parse_document(
        files: List[UploadFile] = File(..., description="要解析的文档文件，支持PDF、图片（png/jpg/jpeg/webp/gif）和Office文档（docx/doc/xlsx/xls/pptx/ppt）"),
        output_dir: str = Form("./output", description="输出目录路径"),
        lang_list: List[str] = Form(["ch"], description="OCR语言列表，支持中文(ch)、英文(en)、韩文(korean)、日文(japan)等"),
        backend: str = Form("pipeline", description="处理后端，仅支持: pipeline"),
        parse_method: str = Form("auto", description="解析方法，可选: auto, txt, ocr"),
        formula_enable: bool = Form(True, description="是否启用公式解析"),
        table_enable: bool = Form(True, description="是否启用表格解析"),
        image_ocr_backend: str = Form("ppocr", description="图片文字识别后端（统一控制），可选: ppocr, llm, auto"),
        word_image_backend: Optional[str] = Form(None, description="[已废弃] Word内图片识别后端，可选: ppocr, llm, auto"),
        image_text_backend: Optional[str] = Form(None, description="[已废弃] 文档图片文字识别后端（pipeline专用），可选: ppocr, llm"),
        server_url: Optional[str] = Form(None, description="VLM服务器URL（当backend为sglang-client时需要）"),
        return_md: bool = Form(True, description="是否返回Markdown内容"),
        return_middle_json: bool = Form(False, description="是否返回中间JSON结果"),
        return_model_output: bool = Form(False, description="是否返回模型原始输出"),
        return_content_list: bool = Form(False, description="是否返回内容列表"),
        return_images: bool = Form(False, description="是否返回提取的图片（base64编码）"),
        start_page_id: int = Form(0, description="起始页码（从0开始）"),
        end_page_id: int = Form(99999, description="结束页码"),
):

    # 获取命令行配置参数
    config = getattr(app.state, "config", {})

    try:
        # 创建唯一的输出目录
        unique_dir = os.path.join(output_dir, str(uuid.uuid4()))
        os.makedirs(unique_dir, exist_ok=True)

        # 处理上传的文档文件（PDF、图片、Office文档）
        pdf_file_names = []
        pdf_bytes_list = []
        original_file_paths = []
        has_office_docs = False
        office_files = []  # 单独存储Office文件信息

        all_images = True
        for file in files:
            content = await file.read()
            file_path = Path(file.filename)

            # 如果是支持的文件格式（PDF、图片、Office文档），使用read_fn处理
            if file_path.suffix.lower() in pdf_suffixes + image_suffixes + office_suffixes:
                # 创建临时文件
                temp_path = Path(unique_dir) / file_path.name
                with open(temp_path, "wb") as f:
                    f.write(content)

                try:
                    # 检查是否为Office文档
                    if file_path.suffix.lower() in ['.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt']:
                        has_office_docs = True
                        if file_path.suffix.lower() in ['.docx', '.doc']:
                            logger.info(f"检测到Word文档: {file_path.name}")
                        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                            logger.info(f"检测到Excel文档: {file_path.name}")
                        elif file_path.suffix.lower() in ['.pptx', '.ppt']:
                            logger.info(f"检测到PowerPoint文档: {file_path.name}")
                        
                        # Office文档：单独存储，不加入PDF处理列表
                        office_files.append({
                            'name': file_path.stem,
                            'path': str(temp_path),
                            'type': file_path.suffix.lower()
                        })
                        logger.info(f"Office文档 {file_path.name} 将使用直接处理模式")
                    
                    else:
                        # 非Office文档：进行PDF读取
                        pdf_bytes = read_fn(temp_path)
                        pdf_bytes_list.append(pdf_bytes)
                        pdf_file_names.append(file_path.stem)
                        original_file_paths.append(str(temp_path))
                    
                    # 注意：不立即删除临时文件，Office直接处理需要访问原始文件
                    if not has_office_docs:
                        os.remove(temp_path)  # 非Office文档可以立即删除临时文件
                        
                except Exception as e:
                    return JSONResponse(
                        status_code=400,
                        content={"error": f"Failed to load file: {str(e)}"}
                    )
                # 标记是否全为图片
                if file_path.suffix.lower() not in image_suffixes:
                    all_images = False
            else:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Unsupported file type: {file_path.suffix}"}
                )


        # 设置语言列表，确保与文件数量一致
        actual_lang_list = lang_list
        if len(actual_lang_list) != len(pdf_file_names):
            # 如果语言列表长度不匹配，使用第一个语言或默认"ch"
            actual_lang_list = [actual_lang_list[0] if actual_lang_list else "ch"] * len(pdf_file_names)

        # 若全部为图片，强制走 OCR 流程
        if all_images:
            parse_method = "ocr"

        # 统一决定图片文字识别后端：优先使用 image_ocr_backend；若老参数有设置为 llm/auto，则覆盖
        effective_image_ocr_backend = image_ocr_backend or "ppocr"
        if word_image_backend in ("llm", "auto") or image_text_backend in ("llm", "auto"):
            # 只要任一老参数要求使用 llm/auto，就统一采用该设置
            effective_image_ocr_backend = "llm" if (word_image_backend == "llm" or image_text_backend == "llm") else "auto"

        # 统一 backend 为 pipeline
        backend = "pipeline"

        # 先处理Office文档
        if office_files:
            logger.info(f"开始处理 {len(office_files)} 个Office文档")
            
            for office_file in office_files:
                try:
                    office_path = Path(office_file['path'])
                    office_name = office_file['name']
                    office_type = office_file['type']
                    
                    if office_type in ['.docx', '.doc']:
                        from doculook.utils.word_handler import handle_word_document_directly
                        logger.info(f"开始处理Word文档: {office_name}")
                        result = handle_word_document_directly(
                            word_path=office_path,
                            output_dir=Path(unique_dir),
                            lang=actual_lang_list[0] if actual_lang_list else "ch",
                            parse_method=parse_method,
                            word_image_backend=image_ocr_backend
                        )
                        
                    elif office_type in ['.xlsx', '.xls']:
                        from doculook.utils.excel_handler import handle_excel_document_directly
                        logger.info(f"开始处理Excel文档: {office_name}")
                        result = handle_excel_document_directly(
                            excel_path=office_path,
                            output_dir=Path(unique_dir),
                            lang=actual_lang_list[0] if actual_lang_list else "ch",
                            parse_method=parse_method,
                            excel_image_backend=image_ocr_backend
                        )
                        
                    elif office_type in ['.pptx', '.ppt']:
                        from doculook.utils.ppt_handler import handle_ppt_document_directly
                        logger.info(f"开始处理PowerPoint文档: {office_name}")
                        result = handle_ppt_document_directly(
                            file_path=office_path,
                            output_dir=Path(unique_dir),
                            lang=actual_lang_list[0] if actual_lang_list else "ch"
                        )
                    
                    if result:
                        logger.info(f"Office文档 {office_name} 处理成功")
                    else:
                        logger.warning(f"Office文档 {office_name} 处理失败")
                        
                except Exception as e:
                    logger.error(f"Office文档 {office_name} 处理异常: {e}")
        
        # 如果只有Office文档，直接返回结果
        if office_files and not pdf_bytes_list:
            logger.info("所有文件都是Office文档，直接处理完成")
            
            # 收集输出结果
            results = []
            for office_file in office_files:
                office_name = office_file['name']
                office_type = office_file['type']
                
                # 根据不同Office类型查找正确的文件路径
                potential_paths = [
                    Path(unique_dir) / f"{office_name}.md",  # 直接路径
                    Path(unique_dir) / office_name / "auto" / f"{office_name}.md",  # Word/Excel路径
                    Path(unique_dir) / office_name / f"{office_name}.md",  # 其他可能路径
                ]
                
                content = ""
                md_file_found = None
                
                for md_file in potential_paths:
                    if md_file.exists():
                        try:
                            with open(md_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                            md_file_found = md_file
                            logger.info(f"找到Office文档 {office_name} 的Markdown文件: {md_file}")
                            break
                        except Exception as e:
                            logger.warning(f"读取文件 {md_file} 失败: {e}")
                
                if not md_file_found:
                    logger.warning(f"未找到Office文档 {office_name} 的Markdown文件，尝试查找: {[str(p) for p in potential_paths]}")
                    content = f"Office文档 {office_name} 处理完成，但输出文件未找到。"
                
                results.append({
                    "name": office_name, 
                    "md_content": content
                })
            
            return JSONResponse(content={"results": results})

        # 如果还有PDF文档需要处理，调用异步处理函数
        await aio_do_parse(
            output_dir=unique_dir,
            pdf_file_names=pdf_file_names,
            pdf_bytes_list=pdf_bytes_list,
            p_lang_list=actual_lang_list,
            backend=backend,
            parse_method=parse_method,
            formula_enable=formula_enable,
            table_enable=table_enable,
            server_url=server_url,
            image_ocr_backend=effective_image_ocr_backend,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=return_md,
            f_dump_middle_json=return_middle_json,
            f_dump_model_output=return_model_output,
            f_dump_orig_pdf=False,
            f_dump_content_list=return_content_list,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            word_direct_processing=False,  # Office文档已经单独处理了
            original_file_paths=None,
            **config
        )

        # 构建结果路径
        result_dict = {}
        for pdf_name in pdf_file_names:
            result_dict[pdf_name] = {}
            data = result_dict[pdf_name]

            if backend.startswith("pipeline"):
                parse_dir = os.path.join(unique_dir, pdf_name, parse_method)
            else:
                parse_dir = os.path.join(unique_dir, pdf_name, "vlm")

            if os.path.exists(parse_dir):
                if return_md:
                    data["md_content"] = get_infer_result(".md", pdf_name, parse_dir)
                if return_middle_json:
                    data["middle_json"] = get_infer_result("_middle.json", pdf_name, parse_dir)
                if return_model_output:
                    data["model_output"] = get_infer_result("_model.json", pdf_name, parse_dir)
                if return_content_list:
                    data["content_list"] = get_infer_result("_content_list.json", pdf_name, parse_dir)
                if return_images:
                    image_paths = glob(f"{parse_dir}/images/*.jpg")
                    data["images"] = {
                        os.path.basename(
                            image_path
                        ): f"data:image/jpeg;base64,{encode_image(image_path)}"
                        for image_path in image_paths
                    }
        # 仅支持 pipeline 后端
        backend_resp = "pipeline"
        return JSONResponse(
            status_code=200,
            content={
                "backend": backend_resp,
                "version": __version__,
                "results": result_dict
            }
        )
    except Exception as e:
        logger.exception(e)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process file: {str(e)}"}
        )


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
@click.option('--host', default='127.0.0.1', help='Server host (default: 127.0.0.1)')
@click.option('--port', default=8000, type=int, help='Server port (default: 8000)')
@click.option('--reload', is_flag=True, help='Enable auto-reload (development mode)')
def main(ctx, host, port, reload, **kwargs):

    kwargs.update(arg_parse(ctx))

    # 将配置参数存储到应用状态中
    app.state.config = kwargs

    """启动DocuLook FastAPI服务器的命令行入口"""
    print(f"Start DocuLook FastAPI Service: http://{host}:{port}")
    print("The API documentation can be accessed at the following address:")
    print(f"- Swagger UI: http://{host}:{port}/docs")
    print(f"- ReDoc: http://{host}:{port}/redoc")

    uvicorn.run(
        "doculook.cli.fast_api:app",
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    main()