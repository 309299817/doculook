from __future__ import annotations

import json
from base64 import b64encode
from typing import Iterable, List, Optional, Union

import httpx
from loguru import logger


class GenericVisionPredictor:
    """通用 Vision LLM 图片识别（OpenAI 兼容接口）

    仅实现最小化的单图识别调用，用于从图片中提取文字（类似 OCR）。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        top_p: float = 0.8,
        max_new_tokens: int = 2048,
        http_timeout: int = 600,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.http_timeout = http_timeout

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_messages(self, image: bytes, prompt: str) -> list[dict]:
        image_base64 = b64encode(image).decode("utf-8")
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": prompt or "请识别图片中的文本内容，按阅读顺序输出。"},
                ],
            }
        ]

    def predict(self, image: bytes, prompt: str = "") -> str:
        messages = self._build_messages(image, prompt)
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_new_tokens,
        }

        try:
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=body,
                timeout=self.http_timeout,
            )
            if resp.status_code != 200:
                logger.error(f"Vision LLM 错误: {resp.status_code} {resp.text}")
                raise RuntimeError(f"Vision LLM 请求失败: {resp.status_code}")
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"调用 Vision LLM 失败: {e}")
            raise


