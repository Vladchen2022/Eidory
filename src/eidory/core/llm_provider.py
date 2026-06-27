from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass

import requests

from eidory.core.ai_vision import (
    AI_VISION_FIELD_VALUES,
    AI_VISION_LIGHTING_VALUES,
    normalize_ai_vision_value,
)
from eidory.core.inspiration import InspirationTerm, normalize_inspiration_terms


@dataclass(frozen=True)
class InspirationProposal:
    questions: list[str]
    terms: list[InspirationTerm]
    model_name: str


@dataclass(frozen=True)
class SearchPlanFilter:
    field: str
    value: str
    optional: bool = False
    reason: str = ""


@dataclass(frozen=True)
class SearchPlanProposal:
    questions: list[str]
    terms: list[InspirationTerm]
    filters: list[SearchPlanFilter]
    model_name: str


@dataclass(frozen=True)
class ProjectSuggestion:
    name: str
    summary: str
    model_name: str


@dataclass(frozen=True)
class GroupNameSuggestion:
    name: str
    summary: str


@dataclass(frozen=True)
class CreativeNodeSuggestion:
    title: str
    note: str
    search_query: str


@dataclass(frozen=True)
class CreativeNodeNoteSuggestion:
    note: str
    search_query: str


@dataclass(frozen=True)
class CreativeProjectSeedSuggestion:
    title: str
    brief: str
    extra: str


@dataclass(frozen=True)
class CreativeProjectCopySuggestion:
    copy_text: str
    nodes: list[CreativeNodeSuggestion]


class LLMProviderError(RuntimeError):
    pass


_CREATIVE_PROJECT_RANDOM = random.SystemRandom()


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

    def generate_search_plan(
        self,
        *,
        brief: str,
        answers: str = "",
        language: str = "zh",
    ) -> SearchPlanProposal:
        clean_brief = brief.strip()
        if not clean_brief:
            raise LLMProviderError("创作主题不能为空")
        model_name = self.model_name or self._first_available_model()
        messages = [
            {
                "role": "system",
                "content": _search_plan_system_prompt(language),
            },
            {
                "role": "user",
                "content": _build_search_plan_prompt(clean_brief, answers.strip(), language=language),
            },
        ]
        content = self._chat_completion(
            model_name=model_name,
            messages=messages,
            prefer_json=True,
            response_format=_search_plan_response_format(),
            temperature=0.55,
            max_tokens=2600,
        )
        try:
            return parse_search_plan_proposal(content, model_name=model_name)
        except LLMProviderError as first_error:
            repaired_content = self._repair_search_plan_json_response(
                model_name=model_name,
                original_content=content,
            )
            try:
                return parse_search_plan_proposal(repaired_content, model_name=model_name)
            except LLMProviderError as second_error:
                raise LLMProviderError(
                    f"{first_error}；自动修复也失败：{second_error}"
                ) from second_error

    def suggest_project_details(
        self,
        *,
        brief: str,
        selected_terms: list[str],
        file_names: list[str],
        language: str = "zh",
    ) -> ProjectSuggestion:
        model_name = self.model_name or self._first_available_model()
        messages = [
            {
                "role": "system",
                "content": (
                    "You name temporary visual reference projects. Return strict JSON only."
                    if language == "en"
                    else "你负责给视觉参考临时项目命名并写摘要。只输出严格 JSON。"
                ),
            },
            {
                "role": "user",
                "content": _build_project_suggestion_prompt(
                    brief=brief,
                    selected_terms=selected_terms,
                    file_names=file_names,
                    language=language,
                ),
            },
        ]
        content = self._chat_completion(
            model_name=model_name,
            messages=messages,
            prefer_json=True,
            response_format=_project_response_format(),
            temperature=0.35,
            max_tokens=700,
        )
        name, summary = parse_project_suggestion(content, language=language)
        return ProjectSuggestion(name=name, summary=summary, model_name=model_name)

    def suggest_reference_group_names(
        self,
        *,
        groups: list[dict[str, object]],
        language: str = "zh",
    ) -> list[GroupNameSuggestion]:
        model_name = self.model_name or self._first_available_model()
        messages = [
            {
                "role": "system",
                "content": (
                    "You name clustered visual reference groups. Return strict JSON only."
                    if language == "en"
                    else "你负责给已经聚类的视觉参考图组命名。只输出严格 JSON。"
                ),
            },
            {
                "role": "user",
                "content": _build_group_naming_prompt(groups=groups, language=language),
            },
        ]
        content = self._chat_completion(
            model_name=model_name,
            messages=messages,
            prefer_json=True,
            response_format=_group_names_response_format(),
            temperature=0.4,
            max_tokens=1400,
        )
        return parse_group_name_suggestions(content, expected_count=len(groups), language=language)

    def generate_creative_nodes(
        self,
        *,
        project_brief: str,
        parent_title: str,
        parent_note: str = "",
        language: str = "zh",
    ) -> tuple[list[CreativeNodeSuggestion], str]:
        clean_brief = project_brief.strip()
        clean_title = parent_title.strip()
        if not clean_brief and not clean_title:
            raise LLMProviderError("创作主题不能为空")
        model_name = self.model_name or self._first_available_model()
        messages = [
            {
                "role": "system",
                "content": (
                    "You split an illustration reference project into useful visual reference nodes. "
                    "Return strict JSON only."
                    if language == "en"
                    else "你负责把插画参考项目拆成可执行的视觉参考节点。只输出严格 JSON。"
                ),
            },
            {
                "role": "user",
                "content": _build_creative_nodes_prompt(
                    project_brief=clean_brief,
                    parent_title=clean_title,
                    parent_note=parent_note.strip(),
                    language=language,
                ),
            },
        ]
        content = self._chat_completion(
            model_name=model_name,
            messages=messages,
            prefer_json=True,
            response_format=_creative_nodes_response_format(),
            temperature=0.65,
            max_tokens=1800,
        )
        nodes = parse_creative_node_suggestions(content, language=language)
        return nodes, model_name

    def generate_creative_node_note(
        self,
        *,
        project_brief: str,
        project_extra: str = "",
        project_outline: str = "",
        node_title: str,
        current_note: str = "",
        node_path: str = "",
        language: str = "zh",
    ) -> tuple[CreativeNodeNoteSuggestion, str]:
        clean_title = node_title.strip()
        clean_brief = project_brief.strip()
        if not clean_title and not clean_brief:
            raise LLMProviderError("创作主题不能为空")
        model_name = self.model_name or self._first_available_model()
        messages = [
            {
                "role": "system",
                "content": (
                    "You fill one fixed illustration planning node. Do not create child nodes. "
                    "Return one strict JSON object only. Do not output analysis, reasoning, "
                    "chain-of-thought, or a thinking process."
                    if language == "en"
                    else "你只负责补全一个固定插画规划节点。不要生成子节点。只输出严格 JSON 对象，不要输出分析、推理、思考过程或步骤。"
                ),
            },
            {
                "role": "user",
                "content": _build_creative_node_note_prompt(
                    project_brief=clean_brief,
                    project_extra=project_extra.strip(),
                    project_outline=project_outline.strip(),
                    node_title=clean_title,
                    current_note=current_note.strip(),
                    node_path=node_path.strip(),
                    language=language,
                ),
            },
        ]
        content = self._chat_completion(
            model_name=model_name,
            messages=messages,
            prefer_json=False,
            reasoning_effort="none",
            temperature=0.35,
            max_tokens=900,
        )
        suggestion = parse_creative_node_note_suggestion(content, language=language)
        return _refine_creative_node_note_suggestion(
            suggestion,
            project_brief=clean_brief,
            project_extra=project_extra.strip(),
            node_title=clean_title,
            current_note=current_note.strip(),
            node_path=node_path.strip(),
            language=language,
        ), model_name

    def generate_creative_project_seed(
        self,
        *,
        template_label: str,
        template_outline: str,
        language: str = "zh",
    ) -> tuple[CreativeProjectSeedSuggestion, str]:
        model_name = self.model_name or self._first_available_model()
        messages = [
            {
                "role": "system",
                "content": (
                    "You create one complete illustration planning test prompt. "
                    "Return one strict JSON object only. Do not output reasoning or Markdown."
                    if language == "en"
                    else "你负责生成一个完整的插画创作测试题。只输出严格 JSON 对象，不要输出推理、解释或 Markdown。"
                ),
            },
            {
                "role": "user",
                "content": _build_creative_project_seed_prompt(
                    template_label=template_label,
                    template_outline=template_outline,
                    language=language,
                ),
            },
        ]
        content = self._chat_completion(
            model_name=model_name,
            messages=messages,
            prefer_json=False,
            reasoning_effort="none",
            temperature=1.05,
            max_tokens=900,
        )
        return parse_creative_project_seed_suggestion(content, language=language), model_name

    def generate_creative_project_copy(
        self,
        *,
        project_brief: str,
        nodes: list[dict[str, str]],
        language: str = "zh",
    ) -> tuple[CreativeProjectCopySuggestion, str]:
        clean_brief = project_brief.strip()
        if not clean_brief and not nodes:
            raise LLMProviderError("创作项目不能为空")
        model_name = self.model_name or self._first_available_model()
        messages = [
            {
                "role": "system",
                "content": (
                    "You write concise, visual illustration concept copy from fixed planning nodes. "
                    "Respect any existing node information. Output only the final copy text. "
                    "Do not output JSON, Markdown, explanation, or reasoning."
                    if language == "en"
                    else "你负责根据固定创作节点写一段画面感强的插画概念文案。必须尊重已有节点信息。只输出最终文案正文，不要输出 JSON、Markdown、解释或思考过程。"
                ),
            },
            {
                "role": "user",
                "content": _build_creative_project_copy_text_prompt(
                    project_brief=clean_brief,
                    nodes=nodes,
                    language=language,
                ),
            },
        ]
        content = self._chat_completion(
            model_name=model_name,
            messages=messages,
            prefer_json=False,
            reasoning_effort="none",
            temperature=0.75,
            max_tokens=1400,
            allow_reasoning_content=True,
        )
        return parse_creative_project_copy_suggestion(content, language=language), model_name

    def _chat_completion(
        self,
        *,
        model_name: str,
        messages: list[dict[str, str]],
        prefer_json: bool,
        response_format: dict[str, object] | None = None,
        reasoning_effort: str | None = None,
        temperature: float = 0.75,
        max_tokens: int = 2200,
        allow_reasoning_content: bool = False,
    ) -> str:
        payload: dict[str, object] = {
            "model": model_name,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
        }
        if prefer_json:
            payload["response_format"] = response_format or _inspiration_response_format()
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        headers = self._headers()
        response = None
        for attempt in range(2):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
                if response.status_code in {400, 422} and (
                    (prefer_json and "response_format" in payload) or "reasoning_effort" in payload
                ):
                    if prefer_json:
                        payload.pop("response_format", None)
                    payload.pop("reasoning_effort", None)
                    response = requests.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                        timeout=self.timeout_seconds,
                    )
                response.raise_for_status()
                break
            except requests.Timeout as exc:
                raise LLMProviderError(
                    f"{self.service_name} 请求超时：模型在 {self.timeout_seconds} 秒内没有返回。"
                    "请确认本地模型未在排队/卡住，或换用更快的文本模型后重试。"
                ) from exc
            except (requests.ConnectionError, requests.ChunkedEncodingError) as exc:
                if attempt == 0:
                    time.sleep(0.35)
                    continue
                raise LLMProviderError(
                    f"{self.service_name} 连接被中断：本地模型服务关闭了连接。"
                    "请确认 LM Studio 仍在运行且模型已加载，然后重试。"
                ) from exc
            except requests.RequestException as exc:
                raise LLMProviderError(f"{self.service_name} 请求失败：{exc}") from exc
        if response is None:
            raise LLMProviderError(f"{self.service_name} 请求失败：没有收到响应")

        try:
            message = response.json()["choices"][0]["message"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMProviderError("LM Studio 返回格式无效") from exc
        content = str(message.get("content") or "").strip()
        reasoning_content = str(message.get("reasoning_content") or "").strip()
        if not content and reasoning_content:
            if allow_reasoning_content:
                return reasoning_content
            raise LLMProviderError(
                f"{self.service_name} 只返回了思考过程，没有返回最终内容。"
                "请在模型设置中关闭 Thinking/Reasoning，或换用非推理文本模型后重试。"
            )
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

    def _repair_search_plan_json_response(self, *, model_name: str, original_content: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "你只做格式修复。把用户提供的内容转换成严格 JSON，不要解释。",
            },
            {
                "role": "user",
                "content": (
                    "把下面内容整理成 JSON："
                    '{"questions":[],"terms":[{"title":"","query":"","axis":"visual","reason":""}],'
                    '"filters":[{"field":"scene_location","value":"indoor","optional":false,"reason":""}]}\n\n'
                    f"{original_content}"
                ),
            },
        ]
        return self._chat_completion(
            model_name=model_name,
            messages=messages,
            prefer_json=True,
            response_format=_search_plan_response_format(),
            temperature=0.2,
            max_tokens=2600,
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


def parse_search_plan_proposal(content: str, *, model_name: str) -> SearchPlanProposal:
    payload = _load_json_object(content)
    questions = [
        " ".join(question.strip().split())[:120]
        for question in payload.get("questions", [])
        if isinstance(question, str) and question.strip()
    ][:3]
    terms = normalize_inspiration_terms(payload.get("terms", []))
    if len(terms) < 5:
        raise LLMProviderError("AI 生成的语义探针太少，请换个主题或重试")
    raw_filters = payload.get("filters", [])
    if not isinstance(raw_filters, list):
        raw_filters = []
    filters: list[SearchPlanFilter] = []
    seen: set[tuple[str, str]] = set()
    for raw_filter in raw_filters:
        parsed = _search_plan_filter_from_payload(raw_filter)
        if parsed is None:
            continue
        key = (parsed.field, parsed.value)
        if key in seen:
            continue
        seen.add(key)
        filters.append(parsed)
    return SearchPlanProposal(
        questions=questions,
        terms=terms[:30],
        filters=filters[:14],
        model_name=model_name,
    )


def _search_plan_filter_from_payload(payload: object) -> SearchPlanFilter | None:
    if not isinstance(payload, dict):
        return None
    field = str(payload.get("field") or "").strip()
    if field not in AI_VISION_FIELD_VALUES and field != "lighting":
        return None
    value = normalize_ai_vision_value(field, payload.get("value"))
    allowed = AI_VISION_LIGHTING_VALUES if field == "lighting" else AI_VISION_FIELD_VALUES[field]
    if value not in allowed:
        return None
    return SearchPlanFilter(
        field=field,
        value=value,
        optional=bool(payload.get("optional")),
        reason=_clean_project_text(payload.get("reason"), max_length=120),
    )


def parse_project_suggestion(content: str, *, language: str = "zh") -> tuple[str, str]:
    payload = _load_json_object(content)
    fallback_name = "Reference Set" if language == "en" else "灵感参考组"
    name = _clean_project_text(payload.get("name"), max_length=60) or fallback_name
    summary = _clean_project_text(payload.get("summary"), max_length=300)
    return name, summary


def parse_group_name_suggestions(
    content: str,
    *,
    expected_count: int,
    language: str = "zh",
) -> list[GroupNameSuggestion]:
    payload = _load_json_object(content)
    raw_groups = payload.get("groups", [])
    if not isinstance(raw_groups, list):
        raw_groups = []
    fallback_prefix = "Reference Group" if language == "en" else "参考组"
    suggestions: list[GroupNameSuggestion] = []
    for index in range(expected_count):
        raw = raw_groups[index] if index < len(raw_groups) and isinstance(raw_groups[index], dict) else {}
        name = _clean_project_text(raw.get("name"), max_length=60) or f"{fallback_prefix} {index + 1}"
        summary = _clean_project_text(raw.get("summary"), max_length=240)
        suggestions.append(GroupNameSuggestion(name=name, summary=summary))
    return suggestions


def parse_creative_node_suggestions(
    content: str,
    *,
    language: str = "zh",
) -> list[CreativeNodeSuggestion]:
    payload = _load_json_object(content)
    raw_nodes = payload.get("nodes", [])
    if not isinstance(raw_nodes, list):
        raw_nodes = []
    fallback_prefix = "Reference Node" if language == "en" else "参考节点"
    suggestions: list[CreativeNodeSuggestion] = []
    seen_titles: set[str] = set()
    for index, raw in enumerate(raw_nodes):
        if not isinstance(raw, dict):
            continue
        title = _clean_project_text(raw.get("title"), max_length=48) or f"{fallback_prefix} {index + 1}"
        if title in seen_titles:
            continue
        note = _clean_project_text(raw.get("note"), max_length=260)
        search_query = _clean_project_text(raw.get("search_query"), max_length=180)
        if not search_query:
            search_query = " ".join(part for part in [title, note] if part).strip()[:180]
        suggestions.append(
            CreativeNodeSuggestion(
                title=title,
                note=note,
                search_query=search_query,
            )
        )
        seen_titles.add(title)
        if len(suggestions) >= 8:
            break
    if len(suggestions) < 2:
        raise LLMProviderError("AI 生成的创作节点太少，请补充主题后重试")
    return suggestions


def parse_creative_node_note_suggestion(
    content: str,
    *,
    language: str = "zh",
) -> CreativeNodeNoteSuggestion:
    if _looks_like_model_reasoning(content):
        raise LLMProviderError("AI 返回了思考过程，而不是可用的节点说明")
    try:
        payload = _load_json_object(content)
    except LLMProviderError:
        note = _clean_creative_node_note_plain_text(content, max_length=900)
        search_query = _creative_node_search_query_from_plain_note(note, max_length=240)
        if not _is_useful_creative_node_note(note):
            raise LLMProviderError("AI 没有返回可用的节点说明")
        return CreativeNodeNoteSuggestion(note=note, search_query=search_query)
    note = _clean_project_text(payload.get("note"), max_length=900)
    search_query = _clean_project_text(payload.get("search_query"), max_length=240)
    if not _is_useful_creative_node_note(note) and not _is_useful_creative_node_note(search_query):
        raise LLMProviderError("AI 没有返回可用的节点说明")
    if not search_query:
        search_query = note[:240]
    if not note:
        note = search_query
    return CreativeNodeNoteSuggestion(note=note, search_query=search_query)


def parse_creative_project_seed_suggestion(
    content: str,
    *,
    language: str = "zh",
) -> CreativeProjectSeedSuggestion:
    payload = _load_json_object(content)
    fallback_title = "AI Creative Test" if language == "en" else "AI创作测试题"
    title = _first_clean_project_text(
        payload,
        ("title", "name", "project_title", "projectTitle", "项目名称", "标题"),
        max_length=80,
    )
    brief = _first_clean_project_text(
        payload,
        ("brief", "theme", "topic", "creative_brief", "creativeBrief", "创作主题", "主题"),
        max_length=500,
    )
    extra = _first_clean_project_text(
        payload,
        ("extra", "supplement", "supplemental", "context", "details", "补充信息", "补充"),
        max_length=900,
    )
    if not brief:
        raise LLMProviderError("AI 没有返回可用的创作主题")
    if not title:
        title = brief[:32].strip() or fallback_title
    return CreativeProjectSeedSuggestion(title=title, brief=brief, extra=extra)


def _refine_creative_node_note_suggestion(
    suggestion: CreativeNodeNoteSuggestion,
    *,
    project_brief: str,
    project_extra: str,
    node_title: str,
    current_note: str,
    node_path: str,
    language: str,
) -> CreativeNodeNoteSuggestion:
    if language == "en":
        return suggestion
    refined_query = _refine_creative_node_search_query(
        suggestion.search_query,
        project_brief=project_brief,
        project_extra=project_extra,
        node_title=node_title,
        current_note=current_note,
        node_path=node_path,
    )
    if not refined_query:
        refined_query = suggestion.search_query
    return CreativeNodeNoteSuggestion(note=suggestion.note, search_query=refined_query[:240])


def _refine_creative_node_search_query(
    query: str,
    *,
    project_brief: str,
    project_extra: str,
    node_title: str,
    current_note: str,
    node_path: str,
) -> str:
    kind = _creative_node_scope_kind(node_title=node_title, node_path=node_path)
    if kind in {"root", "world", "unknown"}:
        if kind == "world":
            return _remove_query_terms(query, _CREATIVE_EVENT_TERMS)
        return query

    user_context = f"{project_brief} {project_extra} {current_note}"
    natural_location = any(term in user_context for term in _CREATIVE_NATURAL_LOCATION_TERMS)
    built_location = any(term in user_context for term in _CREATIVE_BUILT_LOCATION_TERMS)

    if kind == "time":
        return _remove_query_terms(
            query,
            _CREATIVE_WORLD_TERMS
            | _CREATIVE_EVENT_TERMS
            | _CREATIVE_IDENTITY_TERMS
            | _CREATIVE_LOCATION_TERMS
            | _CREATIVE_OBJECT_CONTEXT_TERMS,
        )
    if kind == "place":
        remove_terms = _CREATIVE_EVENT_TERMS | _CREATIVE_IDENTITY_TERMS | _CREATIVE_TIME_TERMS
        if natural_location and not built_location:
            remove_terms |= _CREATIVE_WORLD_TERMS | _CREATIVE_BUILT_LOCATION_TERMS
        return _remove_query_terms(query, remove_terms)
    if kind == "object":
        return _remove_query_terms(query, _CREATIVE_EVENT_TERMS | _CREATIVE_IDENTITY_TERMS)
    if kind == "person":
        return _remove_query_terms(query, _CREATIVE_LOCATION_TERMS | _CREATIVE_TIME_TERMS)
    if kind == "event":
        return _remove_query_terms(
            query,
            _CREATIVE_WORLD_TERMS
            | _CREATIVE_IDENTITY_TERMS
            | _CREATIVE_LOCATION_TERMS
            | _CREATIVE_TIME_TERMS
            | _CREATIVE_OBJECT_CONTEXT_TERMS,
        )
    if kind == "mood":
        return _remove_query_terms(
            query,
            _CREATIVE_WORLD_TERMS
            | _CREATIVE_IDENTITY_TERMS
            | _CREATIVE_LOCATION_TERMS
            | _CREATIVE_EVENT_TERMS,
        )
    if kind == "composition":
        refined = _remove_query_terms(
            query,
            _CREATIVE_WORLD_TERMS
            | _CREATIVE_IDENTITY_TERMS
            | _CREATIVE_LOCATION_TERMS
            | _CREATIVE_TIME_TERMS
            | (_CREATIVE_EVENT_TERMS - {"冲突"})
            | _CREATIVE_SINGLE_CHARACTER_COMPOSITION_TERMS,
        )
        return _ensure_composition_scene_query(refined)
    return query


def _creative_node_scope_kind(*, node_title: str, node_path: str) -> str:
    clean_title = node_title.strip()
    clean_path = node_path.strip()
    if clean_title and clean_path in {"", clean_title}:
        return "root"
    target = f"{clean_path} / {clean_title}".lower()
    checks: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("world", ("世界观", "设定", "时代", "技术", "社会", "秩序", "world", "worldview", "setting")),
        ("time", ("时间", "天气", "季节", "昼夜", "time", "weather", "season")),
        ("place", ("地点", "场景", "环境", "空间", "location", "place", "environment", "scene")),
        ("object", ("物件", "道具", "武器", "工具", "载具", "家具", "器皿", "装备", "object", "prop", "weapon", "tool", "vehicle")),
        ("person", ("人物", "角色", "主角", "次要角色", "服装", "姿态", "character", "person", "people", "costume", "pose")),
        ("event", ("事件", "动作", "行为", "互动", "event", "action", "behavior", "gesture")),
        ("mood", ("氛围", "气氛", "情绪", "色调", "光影", "光线", "mood", "atmosphere", "lighting")),
        ("composition", ("构图", "镜头", "视角", "画面", "布局", "景别", "composition", "camera", "framing")),
    )
    for kind, keywords in checks:
        if any(keyword.lower() in target for keyword in keywords):
            return kind
    return "unknown"


def _remove_query_terms(query: str, remove_terms: set[str]) -> str:
    terms = _split_creative_search_query_terms(query)
    if not terms:
        return ""
    refined: list[str] = []
    seen: set[str] = set()
    ordered_removals = sorted(remove_terms, key=len, reverse=True)
    for term in terms:
        clean = term
        for remove in ordered_removals:
            clean = clean.replace(remove, "")
        clean = re.sub(r"[\s,，、;；]+", " ", clean).strip(" -_/，,、;；")
        clean = " ".join(clean.split())
        if not clean or clean in _CREATIVE_QUERY_DROP_TERMS:
            continue
        if len(re.findall(r"[\w\u4e00-\u9fff]", clean)) < 2:
            continue
        if clean not in seen:
            refined.append(clean)
            seen.add(clean)
    return " ".join(refined)[:240]


def _split_creative_search_query_terms(query: str) -> list[str]:
    return [
        term.strip()
        for term in re.split(r"[,，、;；|/\n]+|\s{1,}", str(query or ""))
        if term.strip()
    ]


def _ensure_composition_scene_query(query: str) -> str:
    clean = " ".join(str(query or "").split())
    if not clean:
        return "场景空间构图 镜头位置 广角环境 前中后景 空间纵深"
    scene_anchors = (
        "场景",
        "空间",
        "环境",
        "镜头",
        "构图",
        "广角",
        "群像",
        "前景",
        "中景",
        "后景",
        "纵深",
        "透视",
        "视角",
        "机位",
    )
    if any(anchor in clean for anchor in scene_anchors):
        return clean[:240]
    return f"场景空间构图 镜头位置 {clean}"[:240]


_CREATIVE_WORLD_TERMS = {
    "中世纪",
    "古代",
    "近未来",
    "未来",
    "现代",
    "科幻",
    "赛博朋克",
    "幻想",
    "魔法",
    "太空",
    "空间站",
    "太空舱",
    "飞船",
}

_CREATIVE_EVENT_TERMS = {
    "战斗",
    "战争",
    "战场",
    "搏斗",
    "打斗",
    "斗殴",
    "厮杀",
    "冲突",
    "追逐",
    "吃饭",
    "用餐",
    "进食",
    "交火",
}

_CREATIVE_IDENTITY_TERMS = {
    "士兵",
    "骑士",
    "步兵",
    "弓箭手",
    "军队",
    "阵营",
    "宇航员",
    "航天员",
    "角色",
    "人物",
}

_CREATIVE_LOCATION_TERMS = {
    "田野",
    "农田",
    "战场",
    "草地",
    "草坡",
    "泥地",
    "泥泞地面",
    "城堡",
    "村庄",
    "空间站",
    "太空舱",
    "室内",
    "餐桌",
    "餐厅",
    "房间",
    "地面",
}

_CREATIVE_BUILT_LOCATION_TERMS = {
    "城堡",
    "村庄",
    "街道",
    "市场",
    "教堂",
    "桥",
    "宫殿",
    "酒馆",
    "餐厅",
    "室内",
    "房间",
    "空间站",
    "太空舱",
    "飞船",
    "城市",
}

_CREATIVE_NATURAL_LOCATION_TERMS = {
    "田野",
    "农田",
    "草地",
    "草坡",
    "泥泞",
    "泥地",
    "平原",
    "森林",
    "山地",
    "山坡",
    "河流",
    "湖泊",
    "海岸",
    "荒野",
    "沼泽",
    "丘陵",
}

_CREATIVE_TIME_TERMS = {
    "傍晚",
    "黄昏",
    "暮色",
    "清晨",
    "黎明",
    "正午",
    "白天",
    "夜晚",
    "夜间",
    "雨天",
    "雪天",
    "雾天",
}

_CREATIVE_OBJECT_CONTEXT_TERMS = {
    "武器",
    "盾牌",
    "长矛",
    "铁剑",
    "盔甲",
    "马车",
    "水车",
    "旗帜",
    "头盔",
    "道具",
}

_CREATIVE_SINGLE_CHARACTER_COMPOSITION_TERMS = {
    "单人",
    "单个角色",
    "单角色",
    "角色特写",
    "人物特写",
    "肖像",
    "半身",
    "头像",
    "大头照",
    "站姿",
    "坐姿",
    "表情",
    "人物姿态",
    "角色姿态",
    "角色设定",
    "人物设定",
}

_CREATIVE_QUERY_DROP_TERMS = {
    "参考",
    "参考图",
    "写实",
    "风格",
    "写实风格",
    "环境参考",
    "感",
}


def _clean_creative_node_note_plain_text(content: str, *, max_length: int) -> str:
    text = str(content or "").strip()
    text = re.sub(r"^```(?:json|markdown|text)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    text = re.sub(r"(?im)^\s*(节点说明|当前节点说明|note)\s*[:：]\s*", "", text).strip()
    text = re.sub(r"(?im)^\s*(搜索语句|搜索短语|search_query|search query)\s*[:：].*$", "", text).strip()
    return " ".join(text.split())[:max_length]


def _creative_node_search_query_from_plain_note(note: str, *, max_length: int) -> str:
    clean = " ".join(str(note or "").split())
    if not clean:
        return ""
    stop_index = len(clean)
    for separator in ("。", "；", ";", ".", "\n"):
        index = clean.find(separator)
        if index > 0:
            stop_index = min(stop_index, index)
    return clean[:stop_index].strip()[:max_length] or clean[:max_length]


def _is_useful_creative_node_note(text: str) -> bool:
    clean = str(text or "").strip()
    if not clean:
        return False
    if _looks_like_model_reasoning(clean):
        return False
    if not re.search(r"[\w\u4e00-\u9fff]", clean):
        return False
    meaningful = re.findall(r"[\w\u4e00-\u9fff]", clean)
    return len(meaningful) >= 6


def _looks_like_model_reasoning(text: str) -> bool:
    clean = str(text or "")
    if not clean.strip():
        return False
    patterns = (
        r"here'?s\s+a\s+thinking\s+process",
        r"\bthinking\s+process\b",
        r"\banaly[sz]e\s+user\s+input\b",
        r"\bdeconstruct\s+constraints\b",
        r"\bchain[-\s]?of[-\s]?thought\b",
        r"\breasoning\s+process\b",
        r"思考过程",
        r"推理过程",
        r"分析用户输入",
    )
    return any(re.search(pattern, clean, flags=re.IGNORECASE) for pattern in patterns)


def parse_creative_project_copy_suggestion(
    content: str,
    *,
    language: str = "zh",
) -> CreativeProjectCopySuggestion:
    try:
        payload = _unwrap_creative_project_copy_payload(_load_json_object(content))
    except LLMProviderError:
        copy_text = _clean_plain_project_copy_text(content)
        if copy_text:
            return CreativeProjectCopySuggestion(copy_text=copy_text, nodes=[])
        raise
    fallback_copy = "A focused visual reference concept." if language == "en" else "一段围绕当前创作节点展开的视觉参考文案。"
    copy_text = _first_clean_project_text(
        payload,
        (
            "copy_text",
            "copyText",
            "copy text",
            "copy",
            "text",
            "final_text",
            "finalCopy",
            "final_copy",
            "caption",
            "description",
            "summary",
            "正文",
            "最终文案",
            "文案",
        ),
        max_length=1600,
    ) or _fallback_project_copy_from_payload(payload)
    if not copy_text and _payload_has_reasoning_text(payload):
        raise LLMProviderError("AI 只返回了思考过程，没有返回可用项目文案")
    copy_text = copy_text or fallback_copy
    raw_nodes = payload.get("nodes", payload.get("node_updates", []))
    if not isinstance(raw_nodes, list):
        raw_nodes = []
    suggestions: list[CreativeNodeSuggestion] = []
    seen_titles: set[str] = set()
    for raw in raw_nodes:
        if not isinstance(raw, dict):
            continue
        title = _clean_project_text(raw.get("title"), max_length=80)
        if not title or title in seen_titles:
            continue
        note = _clean_project_text(raw.get("note"), max_length=900)
        search_query = _clean_project_text(raw.get("search_query"), max_length=240)
        if not note and not search_query:
            continue
        suggestions.append(
            CreativeNodeSuggestion(
                title=title,
                note=note or search_query,
                search_query=search_query or note[:240],
            )
        )
        seen_titles.add(title)
    return CreativeProjectCopySuggestion(copy_text=copy_text, nodes=suggestions)


def _unwrap_creative_project_copy_payload(payload: dict[str, object]) -> dict[str, object]:
    if "copy_text" in payload or "nodes" in payload:
        return payload
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict):
                for key in ("content", "text", "reasoning_content", "reasoning", "output"):
                    unwrapped = _payload_from_nested_content(message.get(key))
                    if unwrapped is not None:
                        return _unwrap_creative_project_copy_payload(unwrapped)
            for key in ("text", "content", "output_text"):
                unwrapped = _payload_from_nested_content(first_choice.get(key))
                if unwrapped is not None:
                    return _unwrap_creative_project_copy_payload(unwrapped)
    message = payload.get("message")
    if isinstance(message, dict):
        for key in ("content", "text", "reasoning_content", "reasoning", "output"):
            unwrapped = _payload_from_nested_content(message.get(key))
            if unwrapped is not None:
                return _unwrap_creative_project_copy_payload(unwrapped)
    for key in ("content", "text", "output_text", "response", "result"):
        content_payload = _payload_from_nested_content(payload.get(key))
        if content_payload is not None:
            return _unwrap_creative_project_copy_payload(content_payload)
        content_text = _clean_project_text(payload.get(key), max_length=1600)
        if content_text:
            return {"copy_text": content_text, "nodes": payload.get("nodes", [])}
    return payload


def _payload_from_nested_content(value: object) -> dict[str, object] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    clean_value = value.strip()
    if not clean_value:
        return None
    try:
        nested = _load_json_object(clean_value)
    except LLMProviderError:
        copy_text = _clean_plain_project_copy_text(clean_value)
        if copy_text:
            return {"copy_text": copy_text, "nodes": []}
        if _looks_like_model_reasoning(clean_value) or _looks_like_reasoning_line(clean_value):
            return None
        return {"copy_text": clean_value, "nodes": []}
    return nested if isinstance(nested, dict) else None


def _payload_has_reasoning_text(value: object) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in {"reasoning", "reasoning_content"} and isinstance(nested, str) and nested.strip():
                return True
            if _payload_has_reasoning_text(nested):
                return True
    if isinstance(value, list):
        return any(_payload_has_reasoning_text(nested) for nested in value)
    return False


def _clean_plain_project_copy_text(content: str) -> str:
    partial_copy = _extract_partial_project_copy_text(content)
    if partial_copy:
        return partial_copy
    draft = _extract_best_project_copy_paragraph(content)
    if draft:
        return draft
    if _looks_like_model_reasoning(content) or _looks_like_reasoning_line(str(content).strip()):
        return ""
    return _strip_project_copy_prefix(_clean_project_text(content, max_length=1600))


def _fallback_project_copy_from_payload(payload: dict[str, object]) -> str:
    candidates: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, str):
            copy_text = _clean_plain_project_copy_text(value)
            if _cjk_count(copy_text) >= 24 or len(copy_text) >= 80:
                candidates.append(copy_text)
            return
        if isinstance(value, dict):
            for key, nested in value.items():
                if key in {"nodes", "node_updates", "title", "search_query", "note"}:
                    continue
                visit(nested)
            return
        if isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(payload)
    if not candidates:
        return ""
    return max(candidates, key=lambda text: (_cjk_count(text), len(text)))


def _extract_partial_project_copy_text(content: str) -> str:
    candidates: list[str] = []
    for key in ("copy_text", "copyText", "copy text", "copy", "text", "文案"):
        pattern = rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)'
        for match in re.finditer(pattern, content, flags=re.DOTALL):
            value = _decode_json_string_fragment(match.group(1))
            clean_value = _strip_project_copy_prefix(_clean_project_text(value, max_length=1600))
            if _cjk_count(clean_value) >= 24 or len(clean_value) >= 80:
                candidates.append(clean_value)
    if not candidates:
        return ""
    return max(candidates, key=lambda text: (_cjk_count(text), len(text)))


def _decode_json_string_fragment(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        return (
            value.replace(r"\\", "\\")
            .replace(r"\"", '"')
            .replace(r"\n", "\n")
            .replace(r"\t", "\t")
        )


def _extract_best_project_copy_paragraph(content: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>", "", content).strip()
    lines = [_strip_project_copy_prefix(line.strip()) for line in text.splitlines()]
    blocks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            blocks.append(" ".join(current))
            current.clear()

    for line in lines:
        if not line:
            flush()
            continue
        if _looks_like_reasoning_line(line):
            flush()
            continue
        if _cjk_count(line) < 20 and len(line) < 80:
            flush()
            continue
        current.append(line)
    flush()

    candidates = [
        _clean_project_text(block, max_length=1600)
        for block in blocks
        if _cjk_count(block) >= 40 or len(block) >= 120
    ]
    if not candidates:
        return ""
    return max(candidates, key=lambda text: (_cjk_count(text), len(text)))


def _looks_like_reasoning_line(line: str) -> bool:
    clean = line.strip()
    lower = clean.lower()
    if re.match(r"^\d+[\.)]\s", clean):
        return True
    if clean.startswith(("-", "*", "•")):
        return True
    if "**" in clean or "```" in clean:
        return True
    reasoning_markers = (
        "thinking process",
        "analyze user input",
        "deconstruct",
        "synthesize",
        "draft",
        "character count",
        "cross-check",
        "requirements",
        "project theme",
        "json nodes",
        "total:",
        "perfectly",
        "用户需求",
        "分析",
        "思考",
        "字数",
        "草稿",
    )
    return any(marker in lower for marker in reasoning_markers)


def _strip_project_copy_prefix(text: str) -> str:
    clean = re.sub(
        r"^\s*(?:最终文案|文案正文|项目文案|文案|copy_text|copy|final)\s*[:：]\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    english_prefix = re.match(r"^[A-Za-z][^。！？\n]*[:：]\s*([\u3400-\u9fff].*)$", clean)
    if english_prefix:
        return english_prefix.group(1).strip()
    return clean


def _cjk_count(text: str) -> int:
    return len(re.findall(r"[\u3400-\u9fff]", text))


def _first_clean_project_text(
    payload: dict[str, object],
    keys: tuple[str, ...],
    *,
    max_length: int,
) -> str:
    for key in keys:
        text = _clean_project_text(payload.get(key), max_length=max_length)
        if text:
            return text
    return ""


def _clean_project_text(value: object, *, max_length: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())[:max_length]


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


def _project_response_format() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "eidory_project_suggestion",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["name", "summary"],
            },
        },
    }


def _group_names_response_format() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "eidory_reference_group_names",
            "schema": {
                "type": "object",
                "properties": {
                    "groups": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "summary": {"type": "string"},
                            },
                            "required": ["name", "summary"],
                        },
                    },
                },
                "required": ["groups"],
            },
        },
    }


def _creative_nodes_response_format() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "eidory_creative_nodes",
            "schema": {
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "note": {"type": "string"},
                                "search_query": {"type": "string"},
                            },
                            "required": ["title", "note", "search_query"],
                        },
                    },
                },
                "required": ["nodes"],
            },
        },
    }


def _creative_node_note_response_format() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "eidory_creative_node_note",
            "schema": {
                "type": "object",
                "properties": {
                    "note": {"type": "string"},
                    "search_query": {"type": "string"},
                },
                "required": ["note", "search_query"],
            },
        },
    }


def _search_plan_response_format() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "eidory_search_plan",
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
                    "filters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "field": {"type": "string"},
                                "value": {"type": "string"},
                                "optional": {"type": "boolean"},
                                "reason": {"type": "string"},
                            },
                            "required": ["field", "value", "optional"],
                        },
                    },
                },
                "required": ["questions", "terms", "filters"],
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


def _search_plan_system_prompt(language: str) -> str:
    if language == "en":
        return (
            "You are a search planner for a local visual reference library. "
            "Do not inspect individual images. Convert the user's concept into semantic probes "
            "and reliable structured visual filters. Return strict JSON only."
        )
    return (
        "你是一个本地视觉参考图库的搜索规划器。"
        "你不逐张看图，只把用户创作意图拆成语义探针和可靠的结构化视觉筛选条件。"
        "只输出严格 JSON。"
    )


def _build_search_plan_prompt(brief: str, answers: str, *, language: str = "zh") -> str:
    answers_block = answers if answers else (
        "The user has not provided extra constraints."
        if language == "en"
        else "用户尚未补充。"
    )
    allowed_fields = {
        **AI_VISION_FIELD_VALUES,
        "lighting": AI_VISION_LIGHTING_VALUES,
    }
    field_notes = "\n".join(
        f"- {field}: {values}"
        for field, values in allowed_fields.items()
    )
    if language == "en":
        return f"""
Creative brief:
{brief}

Additional context:
{answers_block}

Return JSON:
{{
  "questions": ["up to 3 useful follow-up questions"],
  "terms": [
    {{
      "title": "short label",
      "query": "English semantic image-search phrase, concrete and visual",
      "axis": "object_detail | environment | character | lighting | mood | composition | material | era",
      "reason": "why this probe helps"
    }}
  ],
  "filters": [
    {{
      "field": "scene_location",
      "value": "indoor",
      "optional": false,
      "reason": "why this structured filter fits"
    }}
  ]
}}

Allowed structured fields and values:
{field_notes}

Rules:
1. Output 12 to 20 semantic terms.
2. Output 2 to 8 structured filters only when the visual condition is likely useful.
3. Mark filters optional=true when they are plausible but not necessary.
4. Use only the allowed field/value strings above.
5. Use axis precisely: object_detail/material/character probes are standalone references and should not depend on the structured filters; environment/lighting/mood/composition/era probes are scene-context references and may depend on the structured filters.
6. Avoid fragile high-risk claims such as occupation, identity, artist, IP, exact era, or story events.
7. Do not output Markdown or any text outside JSON.
""".strip()
    return f"""
创作主题：
{brief}

用户补充：
{answers_block}

请输出 JSON：
{{
  "questions": ["最多 3 个有助于缩小视觉方向的问题"],
  "terms": [
    {{
      "title": "短标题",
      "query": "中文语义搜图短语，必须具体、可视觉化",
      "axis": "object_detail | environment | character | lighting | mood | composition | material | era",
      "reason": "为什么这个探针对找参考有价值"
    }}
  ],
  "filters": [
    {{
      "field": "scene_location",
      "value": "indoor",
      "optional": false,
      "reason": "为什么这个结构化筛选适合"
    }}
  ]
}}

允许的结构化字段和值：
{field_notes}

规则：
1. terms 输出 12 到 20 条。
2. filters 输出 2 到 8 条，只输出对找图有实际帮助的稳定视觉条件。
3. 如果条件只是可能相关但不必要，optional 必须为 true。
4. field/value 只能使用上面列出的英文枚举值。
5. axis 必须准确：object_detail/material/character 是独立物件或主体参考，不依赖结构化筛选；environment/lighting/mood/composition/era 是场景语境参考，可以叠加结构化筛选。
6. 不要输出职业、身份、艺术家、IP、精确年代、剧情事件等高幻觉条件。
7. 不要输出 Markdown，不要解释 JSON 之外的内容。
""".strip()


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


def _build_creative_nodes_prompt(
    *,
    project_brief: str,
    parent_title: str,
    parent_note: str,
    language: str,
) -> str:
    note_block = parent_note if parent_note else (
        "No parent-node detail has been provided." if language == "en" else "该节点暂无补充说明。"
    )
    if language == "en":
        return f"""
Project brief:
{project_brief or "-"}

Current node:
{parent_title or "-"}

Current node detail:
{note_block}

Split the current node into useful child reference nodes for an illustrator.

Return JSON:
{{
  "nodes": [
    {{
      "title": "short node title",
      "note": "what visual reference this node should collect",
      "search_query": "concrete English semantic image-search phrase"
    }}
  ]
}}

Rules:
1. Output 4 to 8 child nodes.
2. Each node must represent a visually useful reference bucket: setting, object, character pose, lighting, material, composition, mood, or detail.
3. The search_query must be concrete enough for text-image embedding search. Avoid story-only abstractions.
4. Do not output Markdown or any text outside JSON.
""".strip()
    return f"""
项目主题：
{project_brief or "-"}

当前节点：
{parent_title or "-"}

当前节点说明：
{note_block}

请把当前节点拆成适合插画师收集参考图的子节点。

请输出 JSON：
{{
  "nodes": [
    {{
      "title": "简短节点名",
      "note": "这个节点应该收集什么视觉参考",
      "search_query": "具体的中文语义搜图短语"
    }}
  ]
}}

规则：
1. 输出 4 到 8 个子节点。
2. 每个节点必须是对找参考图有用的视觉篮子：环境、物件、角色姿态、光线、材质、构图、气氛或局部细节。
3. search_query 必须能直接用于图文 embedding 搜图，不要只有剧情或抽象词。
4. 不要输出 Markdown，不要解释 JSON 之外的内容。
""".strip()


def _build_creative_node_note_prompt(
    *,
    project_brief: str,
    project_extra: str,
    project_outline: str = "",
    node_title: str,
    current_note: str,
    node_path: str,
    language: str,
) -> str:
    current_note_block = current_note if current_note else (
        "No detail has been provided yet." if language == "en" else "用户尚未填写具体内容。"
    )
    node_path_block = node_path if node_path else node_title
    extra_block = project_extra if project_extra else (
        "No extra project context has been provided." if language == "en" else "用户未填写补充信息。"
    )
    outline_block = project_outline if project_outline else (
        "No project node tree has been provided." if language == "en" else "未提供项目节点树。"
    )
    search_scope_block = _creative_node_search_scope_guidance(
        project_extra=project_extra,
        current_note=current_note,
        node_title=node_title,
        node_path=node_path,
        language=language,
    )
    if language == "en":
        return f"""
Project brief:
{project_brief or "-"}

Extra project context:
{extra_block}

Current fixed planning node:
{node_path_block or "-"}

Existing node detail, if any. Treat it as the strongest constraint:
{current_note_block}

Current project node tree:
{outline_block}

Complete this node as an illustrator's reference-planning field. If existing node detail is provided, treat it as fixed information, then add missing visual decisions around it. Do not merely rewrite the existing sentence.

Search-query scope guidance:
{search_scope_block}

Return JSON:
{{"note":"one concise paragraph about what this node should define visually","search_query":"concrete English semantic image-search phrase"}}

Rules:
1. Do not create or rename nodes.
2. The project brief and extra context are hard constraints. Do not add an era, weather, location, object, character, action, or mood that conflicts with them.
3. If existing node detail is provided, preserve it as the highest-priority user constraint. You may infer and add related environment, objects, lighting, atmosphere, material, action, or composition, but must not contradict, remove, or replace the user's content.
4. Keep the note specific to this node only.
5. The note may stay project-aware. The search_query must obey the search-query scope guidance even when the note mentions project-specific context.
6. Do not copy all note details into search_query. Put only the reference target for this node in search_query.
7. Avoid story-only abstractions; describe visible environment, objects, action, lighting, mood, material, or composition.
8. The note must be a complete, useful sentence. Do not output placeholders such as "...", "TBD", or "same as above".
9. Do not output Markdown, analysis, reasoning, or text outside JSON.
10. If your model supports thinking mode, disable it and output only the final JSON.

/no_think
""".strip()
    return f"""
项目主题：
{project_brief or "-"}

补充信息：
{extra_block}

当前固定规划节点：
{node_path_block or "-"}

现有节点说明（如果有，优先级最高）：
{current_note_block}

当前项目节点树：
{outline_block}

请把这个节点补成插画师能直接拿来找参考图的规划内容。如果用户已经写了节点说明，必须把这些内容当作已经确定的事实、约束和出发点，再补充用户没有写出的相关视觉信息；不是围绕原句扩写，也不是只重复、改写或输出占位符。

搜索范围建议：
{search_scope_block}

输出 JSON：
{{"note":"一段简洁说明，说明这个节点要明确哪些视觉参考","search_query":"具体的中文语义搜图短语"}}

规则：
1. 不要创建节点，不要改节点名。
2. “项目主题”和“补充信息”是硬约束，不要加入与它们冲突的时代、天气、地点、物件、角色、动作或氛围。
3. 如果“现有节点说明”里有用户已写内容，它是最高优先级约束。允许基于它合理联想并补充相关的环境、物件、光线、气氛、材质、动作或构图，但不得推翻、删除、替换或违背。
4. 只补当前节点，不要把其他节点的内容混进来。
5. note 可以保留项目语境；search_query 必须严格服从“搜索范围建议”，即使 note 里提到了项目主题、世界观、地点、角色或事件。
6. 不要把 note 的所有细节复制进 search_query。search_query 只写当前节点要找的参考目标。
7. search_query 必须使用简体中文短语，不要输出英文，除非是不可翻译的专有名词。
8. 避免只有剧情或抽象词，要落到可见环境、物件、动作、光线、气氛、材质或构图。
9. note 必须是一句完整、有信息量的说明。不要输出“...”“待补充”“同上”这类占位内容。
10. 不要输出 Markdown、分析、推理、思考过程或 JSON 之外的内容。
11. 如果模型支持 Thinking/Reasoning 模式，请关闭它，只输出最终 JSON。

/no_think
""".strip()


def _creative_node_search_scope_guidance(
    *,
    project_extra: str = "",
    current_note: str = "",
    node_title: str,
    node_path: str,
    language: str,
) -> str:
    target = f"{node_path} / {node_title}".lower()
    user_context = f"{project_extra}\n{current_note}".lower()
    is_root_node = bool(node_title.strip()) and node_path.strip() in {"", node_title.strip()}

    def has_any(keywords: tuple[str, ...]) -> bool:
        return any(keyword.lower() in target for keyword in keywords)

    if language == "en":
        base = (
            "Node note and search_query serve different jobs. The note may explain how this node fits the project. "
            "The search_query should retrieve useful reference images for this node, not recreate the whole final image. "
            "Keep only the visual constraints necessary for this node; broaden the query when generic references are more useful."
        )
        if is_root_node:
            return (
                f"{base}\n"
                "For the root/top project node, fully respect the project brief, extra context, and existing node detail. "
                "The search_query should represent the fused theme and constraints as a whole."
            )
        if has_any(("world", "worldview", "setting", "era", "technology")):
            return (
                f"{base}\n"
                "For worldview/setting nodes, remove concrete event/action terms, but keep the world's visual language: era, technology, material, weathering, class, terrain, props, vehicles, architecture, and social-order cues. "
                "You may broaden to adjacent inspiration objects that fit the world, even if they are not in the project brief."
            )
        if has_any(("character", "person", "people", "protagonist", "secondary", "costume", "pose")):
            return (
                f"{base}\n"
                "For character/person nodes, split the reference need into layers. Identity, costume, and silhouette may keep necessary project anchors such as astronaut or flight suit. "
                "Body type, pose, eating gestures, facial expression, and group interaction should use broader human-reference terms when useful, such as people eating together, seated posture, or table conversation. "
                "Do not include the whole world/space/station setting unless the node is specifically about the costume or role identity."
            )
        if has_any(("event", "action", "behavior", "gesture")):
            return (
                f"{base}\n"
                "For event/action nodes, search the action and interaction itself first. Keep only theme anchors that change the mechanics of the action; otherwise omit worldbuilding, location, and role identity."
            )
        if has_any(("mood", "atmosphere", "color", "tone", "lighting")):
            return (
                f"{base}\n"
                "For atmosphere or lighting nodes, search mood, color, contrast, and light behavior directly. Avoid subject or world terms unless required."
            )
        if has_any(("composition", "camera", "shot", "angle", "framing", "layout")):
            return (
                f"{base}\n"
                "For composition nodes, search camera angle, framing, spatial layering, and layout. Avoid project-specific nouns unless the composition depends on them."
            )
        if has_any(("time", "season", "weather")):
            return (
                f"{base}\n"
                "For time/weather nodes, if user context already provides a time, weather, or season, search that condition directly without worldbuilding or event terms. "
                "If no such detail is provided, infer a fitting time/weather from the project, then still search the time/weather/light condition itself."
            )
        if has_any(("location", "place", "environment", "scene")):
            return (
                f"{base}\n"
                "For location/environment nodes, split natural and cultural references. Natural terrain from user context should be searched directly without worldbuilding or event terms. "
                "Cultural or built environments should keep necessary worldview anchors, such as medieval castle rather than generic/future castle. "
                "Keep this node about the place itself; put standalone props, vehicles, weapons, tools, furniture, or containers in object/prop nodes when available."
            )
        if has_any(("object", "prop", "weapon", "tool", "vehicle", "furniture", "item", "equipment")):
            return (
                f"{base}\n"
                "For object/prop nodes, search important visible objects separately from location and characters. Keep worldview anchors that affect object design, such as medieval weapon, muddy cart, field banner, wooden shield, or period tableware. "
                "Do not include the concrete event/action unless the object is defined by its active state."
            )
        return base

    base_zh = (
        "节点说明和 search_query 分工不同：节点说明可以解释这个节点如何服务项目，"
        "search_query 的用途是给当前节点找参考图，不是复原最终画面。"
        "只保留当前节点必需的可见约束；如果普通人、普通场景或通用构图参考更有用，就主动泛化搜索词。"
    )
    if is_root_node:
        return (
            f"{base_zh}\n"
            "父节点/最顶层节点：充分尊重项目主题、补充信息和已有节点说明。"
            "search_query 应体现主题与补充信息融合后的整体画面，让相似于总主题的图片被呈现。"
        )
    if has_any(("世界观", "设定", "时代", "技术", "社会", "秩序")):
        return (
            f"{base_zh}\n"
            "世界观/设定节点：不要体现具体事件或动作，例如不要把“战斗、吃饭、追逐”这类事件写进 search_query。"
            "要体现世界的造型特征和质感：时代感、材质、脏污程度、天气痕迹、地貌、建筑、载具、工具、阶层和社会秩序。"
            "允许基于补充信息做灵感泛化，例如“中世纪、泥泞”可以联想到城堡、马车、木栅栏、水车、破旗帜，即使主题没有明说这些物件。"
        )
    if has_any(("人物", "角色", "主角", "次要角色", "服装", "姿态")):
        return (
            f"{base_zh}\n"
            "人物/角色节点：把参考需求拆开。身份、服装、轮廓可以保留必要的主题锚点，例如航天员、舱内服、轻便宇航服；"
            "体格、姿态、吃饭动作、表情、多人互动则应该使用更通用的人类参考，例如多人围坐吃饭、餐桌交流、手部取餐、坐姿表情。"
            "search_query 按“造型锚点 + 通用动作/体态”组织，不要写背景、地点和世界观环境词。"
            "可以写：航天员舱内服、多人围坐吃饭、手部取餐、坐姿表情、群体互动。"
            "不要写：空间站、太空舱背景、太空、失重、漂浮水杯，除非当前节点标题明确是服装、装备或身份设定。"
        )
    if has_any(("事件", "动作", "行为", "互动")):
        return (
            f"{base_zh}\n"
            "事件/动作节点：search_query 必须体现事件本身，但不需要体现世界观、时间、地点或身份。"
            "search_query 必须只包含动作/互动词；禁止包含时代、地点、时间、阵营身份。"
            "例如主题是“几个对立阵营的士兵在战斗”，事件节点应搜索“多人战斗、混战、近身格斗、兵器交锋、冲突动作”；"
            "禁止写“中世纪、田野、泥泞、傍晚、士兵、战场”。主题是吃饭时，则搜索“几个人一起吃饭、餐桌互动、传递食物、手部动作、表情交流”。"
        )
    if has_any(("氛围", "气氛", "情绪", "色调", "光影", "光线")):
        return (
            f"{base_zh}\n"
            "氛围/光线节点：基本不需要体现世界观。note 和 search_query 应联想适合事件的灯光、质感和特效，"
            "例如烟雾、火花、尘土、泥水飞溅、逆光、残阳、冷暖对比、低能见度、高反差阴影。"
            "search_query 必须只包含氛围、灯光、质感、特效词；禁止包含时代、地点、人物身份和具体事件。"
            "例如不要写“中世纪、田野、士兵、战斗、战场”，写“烟雾、火花、尘土、泥水飞溅、残阳逆光、冷暖对比、低能见度”。"
        )
    if has_any(("构图", "镜头", "视角", "画面", "布局", "景别")):
        return (
            f"{base_zh}\n"
            "构图节点：基本不需要体现世界观。note 和 search_query 应联想适合事件的镜头、视角位置、景别、动线和画面层次，"
            "例如低机位、俯视、广角群像、前中后景、对角线冲突、包围关系、纵深透视、混乱人群布局。"
            "构图参考必须是场景类、空间类图片，用来看摄像机在场景空间里的摆放；禁止把 search_query 写成单个角色、肖像、半身、站姿、表情或角色设定参考。"
            "search_query 必须只包含场景空间构图和镜头词；禁止包含时代、地点、人物身份和具体事件。"
            "例如不要写“中世纪、田野、士兵、战斗、战场、单人角色、人物特写”，"
            "写“场景空间构图、镜头位置、广角环境、前中后景、群像布局、空间纵深、环境遮挡、摄像机视角”。"
        )
    if has_any(("时间", "天气", "季节", "昼夜")):
        return (
            f"{base_zh}\n"
            "时间/天气节点：如果用户已有节点说明或补充信息里提供了时间、季节、昼夜或天气，就按用户描述直接搜索这些条件，"
            "不需要带世界观、地点或事件。例如补充信息有“傍晚”，search_query 只需要“傍晚、夕阳、暮色、低角度暖光”等。"
            "如果用户没有描述，则由 AI 根据主题和补充信息补一个相得益彰的时间设定，但 search_query 仍优先搜索时间/光线本身。"
        )
    if has_any(("地点", "场景", "环境", "空间")):
        return (
            f"{base_zh}\n"
            "地点/环境节点：如果用户已有节点说明或补充信息描述的是自然环境，就按自然环境直接搜索，不带世界观或事件，"
            "例如“田野”只搜田野、泥地、草坡、农地、地形关系；禁止写中世纪、战斗、士兵、战场、傍晚。"
            "如果描述的是人文/建筑环境，则必须带世界观锚点，例如“城堡”应写“中世纪城堡”，不能变成未来城堡。"
            "如果用户没有描述，则 AI 应根据主题和补充信息补一个相得益彰的地点设定。"
            "地点节点只负责环境本身；独立道具、武器、载具、家具、工具、器皿和环境附属物优先放到“物件”节点。"
        )
    if has_any(("物件", "道具", "武器", "工具", "载具", "家具", "器皿", "装备", "物品")):
        return (
            f"{base_zh}\n"
            "物件/道具节点：把重要可见物从地点和人物里分出来搜索。search_query 可以保留影响物件造型的世界观锚点，"
            "也可以主动联想没有被主题明说但能强化世界质感的间接物件，例如破损工具、脚印车辙、旗帜标识、箱笼容器、丢弃衣物、权力符号、生活器具、维修设备、环境附属结构。"
            "例如中世纪武器、泥泞马车、木盾、破旗帜、水车、农具、粗木餐桌、陶碗、车辙、断裂绳索、临时路障。"
            "不要默认带具体事件动作，例如不要写“正在战斗”或“正在吃饭”，除非物件本身必须呈现使用状态。"
        )
    return base_zh


def _build_creative_project_copy_text_prompt(
    *,
    project_brief: str,
    nodes: list[dict[str, str]],
    language: str,
) -> str:
    clean_nodes: list[dict[str, str]] = []
    for node in nodes:
        clean_nodes.append(
            {
                "title": str(node.get("title", "")).strip()[:80],
                "path": str(node.get("path", "")).strip()[:240],
                "note": str(node.get("note", "")).strip()[:900],
                "search_query": str(node.get("search_query", "")).strip()[:240],
            }
        )
    nodes_json = json.dumps(clean_nodes, ensure_ascii=False, indent=2)
    if language == "en":
        return f"""
Project brief:
{project_brief or "-"}

Planning nodes JSON:
{nodes_json}

Write one vivid visual concept paragraph for an illustrator, 100-220 English words.
Respect the existing node information as hard constraints. Do not add characters, locations, weather, era, events, or mood that conflict with the nodes.

Output only the final copy text. Do not output JSON, Markdown, headings, explanation, or reasoning.

/no_think
""".strip()
    return f"""
项目主题：
{project_brief or "-"}

规划节点 JSON：
{nodes_json}

请根据这些固定节点写一段画面感强、可直接给插画师参考的中文文案，约 180 到 360 个汉字。
必须把已有节点信息当作硬约束，不要加入与节点说明冲突的角色、地点、天气、年代、事件或氛围。

只输出最终文案正文。不要输出 JSON、Markdown、标题、解释或思考过程。

/no_think
""".strip()


def _build_creative_project_seed_prompt(
    *,
    template_label: str,
    template_outline: str,
    language: str,
) -> str:
    random_packet = _creative_project_seed_random_packet(language)
    if language == "en":
        return f"""
Template:
{template_label or "-"}

Template node outline:
{template_outline or "-"}

Random creative draw for this request:
{random_packet}

Create one original illustration planning test prompt for this template.
Return JSON:
{{"title":"short project title","brief":"one concrete visual creative theme","extra":"extra constraints: era, weather, lighting, mood, materials, atmosphere"}}

The random draw must strongly affect the result. Do not treat it as optional decoration.
The brief must describe a drawable scene or design task, not a generic category.
The extra field should give useful constraints but leave room for later node completion.
Avoid defaulting to the same familiar combination: cyberpunk, neon rain, mechanics repairing vehicles, lone repair workers, vending machines, wet streets, blue-purple reflections, or near-future aircraft drivers, unless the random draw explicitly requires it.
Vary subject matter, era, location, social relationship, event type, mood, and visual materials across calls.
Do not copy the random draw as a list. Fuse it into one coherent project.
Do not output Markdown or reasoning.

/no_think
""".strip()
    return f"""
模板：
{template_label or "-"}

模板节点结构：
{template_outline or "-"}

本次随机创意抽签：
{random_packet}

请为这个模板生成一个原创的插画创作测试题。
输出 JSON：
{{"title":"简短项目名","brief":"一句具体、可绘制的创作主题","extra":"补充信息：时代、天气、光源、画面气质、材质、氛围等"}}

随机抽签必须强烈影响结果，不能只当作装饰性参考。
创作主题必须是具体画面或设计任务，不要写成泛泛的类别。
补充信息要给后续节点补全提供方向，但不要把所有节点都写死。
不要默认回到同一种熟悉组合：赛博朋克、霓虹雨夜、机械师修车、孤独维修工、自动售货机、潮湿街道、蓝紫色反光、近未来飞行器驾驶员；除非随机抽签明确要求。
每次都要主动变化题材、时代、地点、人物关系、事件类型、情绪和视觉材质。
不要把随机抽签原样列成清单，要融合成一个完整项目。
不要输出 Markdown、解释或推理过程。

/no_think
""".strip()


_CREATIVE_PROJECT_SEED_AXES_ZH: dict[str, tuple[str, ...]] = {
    "题材方向": (
        "古代仪式与公共生活",
        "荒野生存与临时秩序",
        "海洋、船只或港口劳动",
        "儿童、老人或家庭关系",
        "灾后重建与日常协作",
        "节庆、比赛或集体表演",
        "边境、驿站或迁徙队伍",
        "学院、工坊或知识传承",
        "异星生态或非人文明",
        "宗教、王权或民间信仰",
        "乡村生产、集市或手工业",
        "地下空间、矿区或考古现场",
    ),
    "核心对象或群体": (
        "三到七人的群体关系",
        "一支临时组成的小队",
        "两个阵营之间的微妙接触",
        "师徒、亲属或同事之间的协作",
        "一群动物、机器或非人角色",
        "人群中的一个关键行动者",
        "被围观、照料或审判的对象",
        "正在搬运、搭建或交换物品的人",
    ),
    "空间类型": (
        "开阔自然地形",
        "拥挤但有秩序的公共空间",
        "半毁坏的人造环境",
        "狭窄通道或夹层空间",
        "临时营地、棚屋或移动设施",
        "高处平台、桥梁或阶梯结构",
        "水边、冰面、泥地或风沙环境",
        "室内外交界的门廊、车厢或舱室",
    ),
    "事件类型": (
        "交接、谈判或分配资源",
        "训练、排练或考试",
        "抢修、转移或避险",
        "庆祝、祭祀或告别",
        "发现异常物、遗迹或生物",
        "等待某个信号或人物到来",
        "集体劳动中的小冲突",
        "安静日常里出现意外变化",
    ),
    "时代与世界质感": (
        "史前或远古感",
        "中世纪或前工业时代",
        "十九世纪工业化早期",
        "二十世纪现实主义",
        "近未来但非赛博都市",
        "幻想世界但材质写实",
        "异星殖民但生活化",
        "架空历史与民俗混合",
    ),
    "光线与时间": (
        "清晨斜光",
        "正午硬光",
        "傍晚余晖",
        "阴天漫射光",
        "火光或烛光",
        "雪地反光",
        "沙尘或雾气中的低对比光",
        "室内开口透入的自然光",
    ),
    "视觉材质": (
        "粗布、木材、泥土和旧金属",
        "石材、苔藓、水汽和磨损边缘",
        "陶器、绳索、皮革和手工痕迹",
        "玻璃、透明容器、液体和反射",
        "纸张、布幔、旗帜和尘埃",
        "冰雪、毛皮、骨骼和干裂表面",
        "湿泥、草根、车辙和脚印",
        "漆面剥落、临时补丁和重复使用痕迹",
    ),
    "情绪张力": (
        "安静但隐含危险",
        "紧张中的克制",
        "温暖日常与陌生环境并存",
        "庄重、缓慢、有仪式感",
        "混乱之后的恢复秩序",
        "喜悦表面下有压力",
        "孤立空间里的集体互助",
        "荒诞但可信的生活瞬间",
    ),
}


_CREATIVE_PROJECT_SEED_AXES_EN: dict[str, tuple[str, ...]] = {
    "genre direction": (
        "ancient ritual and public life",
        "wilderness survival and temporary order",
        "ocean, ships, or harbor labor",
        "children, elders, or family relationships",
        "post-disaster rebuilding and everyday cooperation",
        "festival, competition, or collective performance",
        "borderland, relay station, or migrating caravan",
        "academy, workshop, or knowledge transfer",
        "alien ecology or non-human civilization",
        "religion, monarchy, or folk belief",
        "rural production, markets, or craftwork",
        "underground space, mine, or archaeological site",
    ),
    "main subject": (
        "a group relationship of three to seven figures",
        "a temporary small team",
        "a delicate contact between two factions",
        "collaboration between mentor and apprentice, relatives, or coworkers",
        "a group of animals, machines, or non-human characters",
        "one key actor inside a crowd",
        "an object being watched, cared for, or judged",
        "people moving, building, or exchanging objects",
    ),
    "space type": (
        "open natural terrain",
        "crowded but orderly public space",
        "partially damaged built environment",
        "narrow passage or layered in-between space",
        "temporary camp, shed, or mobile facility",
        "high platform, bridge, or stair structure",
        "waterside, ice, mud, or windblown sand",
        "threshold between inside and outside",
    ),
    "event type": (
        "handover, negotiation, or resource distribution",
        "training, rehearsal, or examination",
        "repair, relocation, or emergency avoidance",
        "celebration, ritual, or farewell",
        "discovery of an anomalous object, ruin, or creature",
        "waiting for a signal or an arrival",
        "a small conflict inside collective labor",
        "an unexpected change inside a quiet daily moment",
    ),
    "era and world texture": (
        "prehistoric or ancient",
        "medieval or pre-industrial",
        "early industrial nineteenth century",
        "twentieth-century realism",
        "near future but not cyberpunk city",
        "fantasy world with realistic materials",
        "alien colony with everyday life",
        "alternate history mixed with folklore",
    ),
    "light and time": (
        "early morning slant light",
        "hard noon light",
        "evening afterglow",
        "overcast diffuse light",
        "firelight or candlelight",
        "snow bounce light",
        "low-contrast light in dust or fog",
        "natural light entering through an opening",
    ),
    "visual materials": (
        "coarse cloth, wood, mud, and old metal",
        "stone, moss, moisture, and worn edges",
        "ceramics, rope, leather, and handmade marks",
        "glass, transparent containers, liquid, and reflections",
        "paper, curtains, flags, and dust",
        "ice, fur, bones, and cracked surfaces",
        "wet mud, grass roots, ruts, and footprints",
        "peeling paint, field repairs, and reused parts",
    ),
    "emotional tension": (
        "quiet but dangerous",
        "restrained tension",
        "warm daily life inside a strange environment",
        "solemn, slow, and ritualistic",
        "order returning after chaos",
        "joy with pressure underneath",
        "collective help inside an isolated space",
        "absurd but believable slice of life",
    ),
}


def _creative_project_seed_random_packet(language: str) -> str:
    axes = _CREATIVE_PROJECT_SEED_AXES_EN if language == "en" else _CREATIVE_PROJECT_SEED_AXES_ZH
    axis_names = list(axes)
    selected_names = _CREATIVE_PROJECT_RANDOM.sample(axis_names, k=min(6, len(axis_names)))
    lines = [f"- random id: {_CREATIVE_PROJECT_RANDOM.randrange(100000, 999999)}"] if language == "en" else [
        f"- 随机编号：{_CREATIVE_PROJECT_RANDOM.randrange(100000, 999999)}"
    ]
    for name in selected_names:
        lines.append(f"- {name}: {_CREATIVE_PROJECT_RANDOM.choice(axes[name])}")
    return "\n".join(lines)


def _build_project_suggestion_prompt(
    *,
    brief: str,
    selected_terms: list[str],
    file_names: list[str],
    language: str,
) -> str:
    terms_text = "\n".join(f"- {term}" for term in selected_terms[:12]) or "-"
    files_text = "\n".join(f"- {name}" for name in file_names[:24]) or "-"
    if language == "en":
        return f"""
Creative brief:
{brief or "-"}

Selected semantic probes:
{terms_text}

Representative file names:
{files_text}

Return JSON:
{{"name":"short project name, 2-6 words","summary":"one concise sentence about the visual intent"}}

Keep the name concrete and usable as a sidebar project title.
Do not output Markdown.
""".strip()
    return f"""
创作主题：
{brief or "-"}

已选择的语义探针：
{terms_text}

代表性文件名：
{files_text}

请输出 JSON：
{{"name":"简短项目名，6-18 个字","summary":"一句话说明这组参考图的视觉意图"}}

项目名要具体，适合显示在左侧栏。
不要输出 Markdown。
""".strip()


def _build_group_naming_prompt(*, groups: list[dict[str, object]], language: str) -> str:
    lines: list[str] = []
    for index, group in enumerate(groups, start=1):
        file_names = group.get("file_names", [])
        if not isinstance(file_names, list):
            file_names = []
        badges = group.get("badges", [])
        if not isinstance(badges, list):
            badges = []
        lines.append(f"Group {index}:")
        lines.append("files:")
        lines.extend(f"- {name}" for name in file_names[:14] if isinstance(name, str))
        if badges:
            lines.append("intent labels:")
            lines.extend(f"- {badge}" for badge in badges[:8] if isinstance(badge, str))
    groups_text = "\n".join(lines) or "-"
    if language == "en":
        return f"""
The images below have already been clustered by visual embeddings.
Name each group based only on the file names and intent labels.

{groups_text}

Return JSON:
{{"groups":[{{"name":"short visual group name","summary":"one concise sentence"}}]}}

Return exactly one group object per input group, in the same order.
Do not output Markdown.
""".strip()
    return f"""
下面这些图片已经按图像 embedding 聚类完成。
请只根据文件名和意图标注，为每组命名。

{groups_text}

请输出 JSON：
{{"groups":[{{"name":"简短视觉组名","summary":"一句话说明这一组的共同参考价值"}}]}}

必须按输入顺序返回，每个输入组对应一个 group。
不要输出 Markdown。
""".strip()
