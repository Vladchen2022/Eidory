from __future__ import annotations

import json
import re
from dataclasses import dataclass

import requests

from eidory.core.inspiration import InspirationTerm, normalize_inspiration_terms


@dataclass(frozen=True)
class InspirationProposal:
    questions: list[str]
    terms: list[InspirationTerm]
    model_name: str


class LLMProviderError(RuntimeError):
    pass


class LMStudioProvider:
    def __init__(
        self,
        *,
        base_url: str = "http://localhost:1234/v1",
        model_name: str | None = None,
        api_key: str = "",
        service_name: str = "LM Studio",
        temperature: float | None = None,
        timeout_seconds: int = 90,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key.strip()
        self.service_name = service_name.strip() or "LLM"
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    def generate_inspiration_terms(
        self,
        *,
        brief: str,
        answers: str = "",
        language: str = "zh",
    ) -> InspirationProposal:
        clean_brief = brief.strip()
        if not clean_brief:
            raise LLMProviderError("创作主题不能为空")
        model_name = self.model_name or self._first_available_model()
        messages = [
            {
                "role": "system",
                "content": _system_prompt(language),
            },
            {
                "role": "user",
                "content": _build_inspiration_prompt(clean_brief, answers.strip(), language=language),
            },
        ]
        content = self._chat_completion(model_name=model_name, messages=messages, prefer_json=True)
        try:
            return parse_inspiration_proposal(content, model_name=model_name)
        except LLMProviderError as first_error:
            fallback_terms = _terms_from_plain_text(content)
            if len(fallback_terms) >= 5:
                return InspirationProposal(
                    questions=[],
                    terms=fallback_terms[:30],
                    model_name=model_name,
                )
            repaired_content = self._repair_json_response(
                model_name=model_name,
                original_content=content,
            )
            try:
                return parse_inspiration_proposal(repaired_content, model_name=model_name)
            except LLMProviderError as second_error:
                raise LLMProviderError(
                    f"{first_error}；自动修复也失败：{second_error}"
                ) from second_error

    def _chat_completion(
        self,
        *,
        model_name: str,
        messages: list[dict[str, str]],
        prefer_json: bool,
        temperature: float = 0.75,
        max_tokens: int = 2200,
    ) -> str:
        payload: dict[str, object] = {
            "model": model_name,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
        }
        if prefer_json:
            payload["response_format"] = _inspiration_response_format()
        headers = self._headers()
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            if response.status_code in {400, 422} and prefer_json:
                payload.pop("response_format", None)
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LLMProviderError(f"{self.service_name} 请求失败：{exc}") from exc

        try:
            message = response.json()["choices"][0]["message"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMProviderError("LM Studio 返回格式无效") from exc
        content = str(message.get("content") or "").strip()
        if not content:
            content = str(message.get("reasoning_content") or "").strip()
        if not content:
            raise LLMProviderError("LM Studio 返回了空内容")
        return content

    def _repair_json_response(self, *, model_name: str, original_content: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "你只做格式修复。把用户提供的内容转换成严格 JSON。"
                    "不要输出 Markdown，不要解释。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "把下面内容整理成 JSON："
                    '{"questions":[],"terms":[{"title":"","query":"","axis":"visual","reason":""}]}\n\n'
                    f"{original_content}"
                ),
            },
        ]
        return self._chat_completion(
            model_name=model_name,
            messages=messages,
            prefer_json=True,
            temperature=0.2,
            max_tokens=2200,
        )

    def _first_available_model(self) -> str:
        try:
            response = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=5)
            response.raise_for_status()
            data = response.json()
            models = data.get("data", [])
            if models:
                model_id = models[0].get("id")
                if isinstance(model_id, str) and model_id.strip():
                    return model_id.strip()
        except (KeyError, TypeError, ValueError, requests.RequestException):
            pass
        return "local-model"

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}


def parse_inspiration_proposal(content: str, *, model_name: str) -> InspirationProposal:
    payload = _load_json_object(content)
    questions = [
        " ".join(question.strip().split())[:120]
        for question in payload.get("questions", [])
        if isinstance(question, str) and question.strip()
    ][:3]
    terms = normalize_inspiration_terms(payload.get("terms", []))
    if len(terms) < 5:
        raise LLMProviderError("AI 生成的语义探针太少，请换个主题或重试")
    return InspirationProposal(
        questions=questions,
        terms=terms[:30],
        model_name=model_name,
    )


def _load_json_object(content: str) -> dict[str, object]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = _load_embedded_json_object(content)
    if not isinstance(data, dict):
        raise LLMProviderError("AI 返回的 JSON 顶层必须是对象")
    return data


def _inspiration_response_format() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "eidory_inspiration_terms",
            "schema": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "terms": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "query": {"type": "string"},
                                "axis": {"type": "string"},
                                "reason": {"type": "string"},
                            },
                            "required": ["title", "query"],
                        },
                    },
                },
                "required": ["questions", "terms"],
            },
        },
    }


def _load_embedded_json_object(content: str) -> object:
    decoder = json.JSONDecoder()
    for index, char in enumerate(content):
        if char != "{":
            continue
        try:
            data, _end = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        return data
    raise LLMProviderError("AI 没有返回 JSON")


def _terms_from_plain_text(content: str) -> list[InspirationTerm]:
    terms: list[InspirationTerm] = []
    paragraph_candidates: list[str] = []
    seen_titles: set[str] = set()
    for line in content.splitlines():
        clean = re.sub(r"^[\s\-*•\d.、）)\[\]【】#]+", "", line).strip()
        if not clean or len(clean) < 4:
            continue
        if any(marker in clean.lower() for marker in ["json", "```", "terms", "questions"]):
            continue
        key_value_term = _term_from_key_value_line(clean)
        if key_value_term is not None and key_value_term.title not in seen_titles:
            seen_titles.add(key_value_term.title)
            terms.append(key_value_term)
            continue
        clean = re.sub(r"^(标题|探针|关键词|短语|query|reason)\s*[:：]\s*", "", clean, flags=re.I)
        title, separator, query = clean.partition("：")
        if not separator:
            title, separator, query = clean.partition(":")
        if not separator:
            paragraph_candidates.append(clean)
            title = clean[:16]
            query = clean
        term = {
            "title": title.strip()[:20],
            "query": query.strip()[:120] or clean[:120],
            "axis": "visual",
            "reason": "从 AI 非 JSON 输出中提取",
        }
        parsed = normalize_inspiration_terms([term])
        if parsed and parsed[0].title not in seen_titles:
            seen_titles.add(parsed[0].title)
            terms.append(parsed[0])
    if len(terms) < 5 and paragraph_candidates:
        for sentence in re.split(r"[。；;]\s*", "；".join(paragraph_candidates)):
            clean = sentence.strip()
            if len(clean) < 6:
                continue
            term = {
                "title": clean[:12],
                "query": clean[:120],
                "axis": "visual",
                "reason": "从 AI 非 JSON 输出中提取",
            }
            parsed = normalize_inspiration_terms([term])
            if parsed and parsed[0].title not in seen_titles:
                seen_titles.add(parsed[0].title)
                terms.append(parsed[0])
    return terms


def _term_from_key_value_line(line: str) -> InspirationTerm | None:
    values: dict[str, str] = {}
    for key, value in re.findall(
        r"\b(title|query|axis|reason)\s*[:：]\s*(.*?)(?=\s+\|\s+\b(?:title|query|axis|reason)\s*[:：]|$)",
        line,
        flags=re.I,
    ):
        values[key.lower()] = value.strip().strip('"').strip("'").strip()
    if "title" not in values or "query" not in values:
        return None
    parsed = normalize_inspiration_terms([
        {
            "title": values.get("title", ""),
            "query": values.get("query", ""),
            "axis": values.get("axis", "visual"),
            "reason": values.get("reason", "从 AI 非 JSON 输出中提取"),
        }
    ])
    return parsed[0] if parsed else None


def _system_prompt(language: str) -> str:
    if language == "en":
        return (
            "You are a visual reference search planner for a local creative image library. "
            "Your job is not to write a story or a drawing prompt. Split a vague creative idea "
            "into concrete visual search phrases for semantic image retrieval. "
            "Return strict JSON only, with no Markdown."
        )
    return (
        "你是一个面向视觉创作参考图库的检索策划助手。"
        "你的任务不是写故事，也不是生成绘图提示词，而是把模糊创作想法拆成可用于图片语义检索的视觉短语。"
        "输出必须是严格 JSON，不要使用 Markdown。"
    )


def _build_inspiration_prompt(brief: str, answers: str, *, language: str = "zh") -> str:
    if language == "en":
        answers_block = answers if answers else "The user has not provided extra constraints."
        return f"""
Creative brief:
{brief}

Additional context:
{answers_block}

Return JSON with this exact structure:
{{
  "questions": ["up to 3 questions that help narrow the visual direction"],
  "terms": [
    {{
      "title": "short display title, 2-5 words",
      "query": "the actual English semantic image-search phrase, concrete and visual",
      "axis": "object_detail | environment | character | lighting | mood | composition | material | era",
      "reason": "why this probe is useful, under 25 words"
    }}
  ]
}}

Requirements:
1. Output 20 to 30 terms.
2. Do not merely paraphrase the original brief; include visual associations and substitute-object associations.
3. Abstract ideas must become visible objects, environments, materials, lighting, composition, or mood.
4. Avoid standalone abstract words such as "loneliness", "ruin", or "fate".
5. Every query must be suitable for direct English text-image embedding search.
6. Do not output Markdown or any text outside the JSON.
""".strip()

    answers_block = answers if answers else "用户尚未补充。"
    return f"""
创作主题：
{brief}

用户补充：
{answers_block}

请输出 JSON，结构如下：
{{
  "questions": ["最多 3 个有助于缩小视觉方向的问题"],
  "terms": [
    {{
      "title": "显示给用户的短标题，2-10 个汉字为主",
      "query": "真正用于语义搜图的中文视觉检索短语，必须具体、可视觉化",
      "axis": "object_detail | environment | character | lighting | mood | composition | material | era",
      "reason": "为什么这个探针对找参考有价值，40 字以内"
    }}
  ]
}}

要求：
1. terms 输出 20 到 30 条。
2. 不要只改写主题原句，要做视觉联想和替代物联想。
3. 抽象概念必须落到可见实体、环境、材料、光线、构图或氛围。
4. 避免单独输出“孤独”“落魄”“命运”这类不可直接搜图的词。
5. 每条 query 应适合直接交给中文图文 embedding 搜图。
6. 不要输出 Markdown，不要解释 JSON 之外的内容。
""".strip()
