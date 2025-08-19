# Copyright (c) Opendatalab. All rights reserved.
import os
import time

from loguru import logger
from tqdm import tqdm

from doculook.utils.config_reader import get_device, get_llm_aided_config, get_formula_enable
from doculook.backend.pipeline.model_init import AtomModelSingleton
from doculook.backend.pipeline.para_split import para_split
from doculook.utils.block_pre_proc import prepare_block_bboxes, process_groups
from doculook.utils.block_sort import sort_blocks_by_bbox
from doculook.utils.boxbase import calculate_overlap_area_in_bbox1_area_ratio
from doculook.utils.cut_image import cut_image_and_table
from doculook.utils.enum_class import ContentType
from doculook.utils.llm_aided import llm_aided_title
from doculook.utils.model_utils import clean_memory
from doculook.backend.pipeline.pipeline_magic_model import MagicModel
from doculook.utils.ocr_utils import OcrConfidence
from doculook.utils.span_block_fix import fill_spans_in_blocks, fix_discarded_block, fix_block_spans
from doculook.utils.span_pre_proc import remove_outside_spans, remove_overlaps_low_confidence_spans, \
    remove_overlaps_min_spans, txt_spans_extract
from doculook.version import __version__
from doculook.utils.hash_utils import str_md5


def page_model_info_to_page_info(page_model_info, image_dict, page, image_writer, page_index, ocr_enable=False, formula_enabled=True):
    scale = image_dict["scale"]
    page_pil_img = image_dict["img_pil"]
    page_img_md5 = str_md5(image_dict["img_base64"])
    page_w, page_h = map(int, page.get_size())
    magic_model = MagicModel(page_model_info, scale)

    """从magic_model对象中获取后面会用到的区块信息"""
    discarded_blocks = magic_model.get_discarded()
    text_blocks = magic_model.get_text_blocks()
    title_blocks = magic_model.get_title_blocks()
    inline_equations, interline_equations, interline_equation_blocks = magic_model.get_equations()

    img_groups = magic_model.get_imgs()
    table_groups = magic_model.get_tables()

    """对image和table的区块分组"""
    img_body_blocks, img_caption_blocks, img_footnote_blocks, maybe_text_image_blocks = process_groups(
        img_groups, 'image_body', 'image_caption_list', 'image_footnote_list'
    )

    table_body_blocks, table_caption_blocks, table_footnote_blocks, _ = process_groups(
        table_groups, 'table_body', 'table_caption_list', 'table_footnote_list'
    )

    """获取所有的spans信息"""
    spans = magic_model.get_all_spans()

    """某些图可能是文本块，通过简单的规则判断一下"""
    if len(maybe_text_image_blocks) > 0:
        for block in maybe_text_image_blocks:
            should_add_to_text_blocks = False

            if ocr_enable:
                # 找到与当前block重叠的text spans
                span_in_block_list = [
                    span for span in spans
                    if span['type'] == 'text' and
                       calculate_overlap_area_in_bbox1_area_ratio(span['bbox'], block['bbox']) > 0.7
                ]

                if len(span_in_block_list) > 0:
                    # 计算spans总面积
                    spans_area = sum(
                        (span['bbox'][2] - span['bbox'][0]) * (span['bbox'][3] - span['bbox'][1])
                        for span in span_in_block_list
                    )

                    # 计算block面积
                    block_area = (block['bbox'][2] - block['bbox'][0]) * (block['bbox'][3] - block['bbox'][1])

                    # 判断是否符合文本图条件
                    if block_area > 0 and spans_area / block_area > 0.25:
                        should_add_to_text_blocks = True

            # 根据条件决定添加到哪个列表
            if should_add_to_text_blocks:
                block.pop('group_id', None)  # 移除group_id
                text_blocks.append(block)
            else:
                img_body_blocks.append(block)


    """将所有区块的bbox整理到一起"""
    if formula_enabled:
        interline_equation_blocks = []

    if len(interline_equation_blocks) > 0:

        for block in interline_equation_blocks:
            spans.append({
                "type": ContentType.INTERLINE_EQUATION,
                'score': block['score'],
                "bbox": block['bbox'],
                "content": "",
            })

        all_bboxes, all_discarded_blocks, footnote_blocks = prepare_block_bboxes(
            img_body_blocks, img_caption_blocks, img_footnote_blocks,
            table_body_blocks, table_caption_blocks, table_footnote_blocks,
            discarded_blocks,
            text_blocks,
            title_blocks,
            interline_equation_blocks,
            page_w,
            page_h,
        )
    else:
        all_bboxes, all_discarded_blocks, footnote_blocks = prepare_block_bboxes(
            img_body_blocks, img_caption_blocks, img_footnote_blocks,
            table_body_blocks, table_caption_blocks, table_footnote_blocks,
            discarded_blocks,
            text_blocks,
            title_blocks,
            interline_equations,
            page_w,
            page_h,
        )

    """在删除重复span之前，应该通过image_body和table_body的block过滤一下image和table的span"""
    """顺便删除大水印并保留abandon的span"""
    spans = remove_outside_spans(spans, all_bboxes, all_discarded_blocks)

    """删除重叠spans中置信度较低的那些"""
    spans, dropped_spans_by_confidence = remove_overlaps_low_confidence_spans(spans)
    """删除重叠spans中较小的那些"""
    spans, dropped_spans_by_span_overlap = remove_overlaps_min_spans(spans)

    """根据parse_mode，构造spans，主要是文本类的字符填充"""
    if ocr_enable:
        pass
    else:
        """使用新版本的混合ocr方案."""
        spans = txt_spans_extract(page, spans, page_pil_img, scale, all_bboxes, all_discarded_blocks)

    """先处理不需要排版的discarded_blocks"""
    discarded_block_with_spans, spans = fill_spans_in_blocks(
        all_discarded_blocks, spans, 0.4
    )
    fix_discarded_blocks = fix_discarded_block(discarded_block_with_spans)

    """如果当前页面没有有效的bbox则跳过"""
    if len(all_bboxes) == 0:
        return None

    """对image/table/interline_equation截图"""
    for span in spans:
        if span['type'] in [ContentType.IMAGE, ContentType.TABLE, ContentType.INTERLINE_EQUATION]:
            span = cut_image_and_table(
                span, page_pil_img, page_img_md5, page_index, image_writer, scale=scale
            )

    """span填充进block"""
    block_with_spans, spans = fill_spans_in_blocks(all_bboxes, spans, 0.5)

    """对block进行fix操作"""
    fix_blocks = fix_block_spans(block_with_spans)

    """对block进行排序"""
    sorted_blocks = sort_blocks_by_bbox(fix_blocks, page_w, page_h, footnote_blocks)

    """构造page_info"""
    page_info = make_page_info_dict(sorted_blocks, page_index, page_w, page_h, fix_discarded_blocks)

    return page_info


def result_to_middle_json(model_list, images_list, pdf_doc, image_writer, lang=None, ocr_enable=False, formula_enabled=True, image_text_backend: str = "ppocr"):
    middle_json = {"pdf_info": [], "_backend":"pipeline", "_version_name": __version__}
    formula_enabled = get_formula_enable(formula_enabled)
    for page_index, page_model_info in tqdm(enumerate(model_list), total=len(model_list), desc="Processing pages"):
        page = pdf_doc[page_index]
        image_dict = images_list[page_index]
        page_info = page_model_info_to_page_info(
            page_model_info, image_dict, page, image_writer, page_index, ocr_enable=ocr_enable, formula_enabled=formula_enabled
        )
        if page_info is None:
            page_w, page_h = map(int, page.get_size())
            page_info = make_page_info_dict([], page_index, page_w, page_h, [])
        middle_json["pdf_info"].append(page_info)

    """后置ocr处理"""
    need_ocr_list = []
    img_crop_list = []
    text_block_list = []
    for page_info in middle_json["pdf_info"]:
        for block in page_info['preproc_blocks']:
            if block['type'] in ['table', 'image']:
                for sub_block in block['blocks']:
                    if sub_block['type'] in ['image_caption', 'image_footnote', 'table_caption', 'table_footnote']:
                        text_block_list.append(sub_block)
            elif block['type'] in ['text', 'title']:
                text_block_list.append(block)
        for block in page_info['discarded_blocks']:
            text_block_list.append(block)
    for block in text_block_list:
        for line in block['lines']:
            for span in line['spans']:
                if 'np_img' in span:
                    need_ocr_list.append(span)
                    img_crop_list.append(span['np_img'])
                    span.pop('np_img')
    if len(img_crop_list) > 0:
        if image_text_backend == "llm":
            from doculook.utils.config_reader import get_vision_llm_config
            from doculook.backend.vlm.vision_predictor import GenericVisionPredictor
            vcfg = get_vision_llm_config()
            if not vcfg or not vcfg.get('api_key'):
                logger.warning("image_text_backend 为 llm，但未配置 api_key，回退到 ppocr")
                image_text_backend = "ppocr"

        if image_text_backend == "llm":
            # 使用通用 Vision LLM 对每个裁剪的 span 图片做 OCR
            predictor = GenericVisionPredictor(
                api_key=vcfg['api_key'],
                base_url=vcfg.get('base_url', 'https://api.openai.com/v1'),
                model=vcfg.get('model', 'gpt-4o-mini'),
                temperature=vcfg.get('temperature', 0.0),
                top_p=vcfg.get('top_p', 0.8),
                max_new_tokens=vcfg.get('max_tokens', 16384),
            )
            safe_imgs = []
            for np_img in img_crop_list:
                try:
                    # 将numpy图像转为高质量JPEG字节
                    import cv2
                    import numpy as np
                    import io
                    from PIL import Image
                    if isinstance(np_img, bytes):
                        # 已经是字节，直接用
                        safe_imgs.append(np_img)
                    else:
                        # numpy ndarray (H,W,C)
                        if isinstance(np_img, np.ndarray):
                            rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB) if np_img.ndim == 3 and np_img.shape[2] == 3 else np_img
                            pil = Image.fromarray(rgb)
                            bio = io.BytesIO()
                            pil.save(bio, format='JPEG', quality=90, optimize=True)
                            safe_imgs.append(bio.getvalue())
                        else:
                            # 兜底：转 str/bytes 失败则跳过
                            bio = io.BytesIO()
                            Image.fromarray(np.array(np_img)).save(bio, format='JPEG', quality=85)
                            safe_imgs.append(bio.getvalue())
                except Exception:
                    # 失败时加入空白图片占位，避免整个流程崩溃
                    import numpy as np
                    blank = (255 * np.ones((16, 16, 3), dtype=np.uint8))
                    from PIL import Image
                    import io
                    bio = io.BytesIO()
                    Image.fromarray(blank).save(bio, format='JPEG', quality=80)
                    safe_imgs.append(bio.getvalue())
            ocr_res_list = [[(None, (predictor.predict(image=img, prompt="请识别图片中的文本"), 1.0))] for img in safe_imgs]
        else:
            atom_model_manager = AtomModelSingleton()
            ocr_model = atom_model_manager.get_atom_model(
                atom_model_name='ocr',
                ocr_show_log=False,
                det_db_box_thresh=0.3,
                lang=lang
            )
            ocr_res_list = ocr_model.ocr(img_crop_list, det=False, tqdm_enable=True)[0]
        assert len(ocr_res_list) == len(
            need_ocr_list), f'ocr_res_list: {len(ocr_res_list)}, need_ocr_list: {len(need_ocr_list)}'
        for index, span in enumerate(need_ocr_list):
            ocr_text, ocr_score = ocr_res_list[index]
            if ocr_score > OcrConfidence.min_confidence:
                span['content'] = ocr_text
                span['score'] = float(f"{ocr_score:.3f}")
            else:
                span['content'] = ''
                span['score'] = 0.0

    """分段"""
    para_split(middle_json["pdf_info"])

    """llm优化"""
    llm_aided_config = get_llm_aided_config()

    if llm_aided_config is not None:
        """标题优化"""
        title_aided_config = llm_aided_config.get('title_aided', None)
        if title_aided_config is not None:
            if title_aided_config.get('enable', False):
                llm_aided_title_start_time = time.time()
                llm_aided_title(middle_json["pdf_info"], title_aided_config)
                logger.info(f'llm aided title time: {round(time.time() - llm_aided_title_start_time, 2)}')

    """清理内存"""
    pdf_doc.close()
    if os.getenv('DOCULOOK_DONOT_CLEAN_MEM') is None and len(model_list) >= 10:
        clean_memory(get_device())

    return middle_json


def make_page_info_dict(blocks, page_id, page_w, page_h, discarded_blocks):
    return_dict = {
        'preproc_blocks': blocks,
        'page_idx': page_id,
        'page_size': [page_w, page_h],
        'discarded_blocks': discarded_blocks,
    }
    return return_dict