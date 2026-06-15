from __future__ import annotations

import json
import re
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
class CreativeProjectCopySuggestion:
    copy_text: str
    nodes: list[CreativeNodeSuggestion]


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
                    "Return strict JSON only."
                    if language == "en"
                    else "你只负责补全一个固定插画规划节点。不要生成子节点。只输出严格 JSON。"
                ),
            },
            {
                "role": "user",
                "content": _build_creative_node_note_prompt(
                    project_brief=clean_brief,
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
            prefer_json=True,
            response_format=_creative_node_note_response_format(),
            temperature=0.45,
            max_tokens=900,
        )
        return parse_creative_node_note_suggestion(content, language=language), model_name

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
                    "Respect any existing node information. Return strict JSON only."
                    if language == "en"
                    else "你负责根据固定创作节点写一段画面感强的插画概念文案。必须尊重已有节点信息。只输出严格 JSON。"
                ),
            },
            {
                "role": "user",
                "content": _build_creative_project_copy_prompt(
                    project_brief=clean_brief,
                    nodes=nodes,
                    language=language,
                ),
            },
        ]
        content = self._chat_completion(
            model_name=model_name,
            messages=messages,
            prefer_json=True,
            response_format=_creative_project_copy_response_format(),
            temperature=0.75,
            max_tokens=1800,
        )
        return parse_creative_project_copy_suggestion(content, language=language), model_name

    def _chat_completion(
        self,
        *,
        model_name: str,
        messages: list[dict[str, str]],
        prefer_json: bool,
        response_format: dict[str, object] | None = None,
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
            payload["response_format"] = response_format or _inspiration_response_format()
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
    payload = _load_json_object(content)
    note = _clean_project_text(payload.get("note"), max_length=900)
    search_query = _clean_project_text(payload.get("search_query"), max_length=240)
    if not note and not search_query:
        raise LLMProviderError("AI 没有返回可用的节点说明")
    if not search_query:
        search_query = note[:240]
    if not note:
        note = search_query
    return CreativeNodeNoteSuggestion(note=note, search_query=search_query)


def parse_creative_project_copy_suggestion(
    content: str,
    *,
    language: str = "zh",
) -> CreativeProjectCopySuggestion:
    payload = _load_json_object(content)
    fallback_copy = "A focused visual reference concept." if language == "en" else "一段围绕当前创作节点展开的视觉参考文案。"
    copy_text = _clean_project_text(payload.get("copy_text"), max_length=1600) or fallback_copy
    raw_nodes = payload.get("nodes", [])
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


def _creative_project_copy_response_format() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "eidory_creative_project_copy",
            "schema": {
                "type": "object",
                "properties": {
                    "copy_text": {"type": "string"},
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
                "required": ["copy_text", "nodes"],
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
    node_title: str,
    current_note: str,
    node_path: str,
    language: str,
) -> str:
    current_note_block = current_note if current_note else (
        "No detail has been provided yet." if language == "en" else "用户尚未填写具体内容。"
    )
    node_path_block = node_path if node_path else node_title
    if language == "en":
        return f"""
Project brief:
{project_brief or "-"}

Current fixed planning node:
{node_path_block or "-"}

Existing node detail:
{current_note_block}

Fill this node as an illustrator's reference-planning field.

Return JSON:
{{"note":"one concise paragraph about what this node should define visually","search_query":"concrete English semantic image-search phrase"}}

Rules:
1. Do not create or rename nodes.
2. Keep the note specific to this node only.
3. The search_query must be concrete enough for text-image embedding search.
4. Avoid story-only abstractions; describe visible environment, objects, action, lighting, mood, material, or composition.
5. Do not output Markdown or text outside JSON.
""".strip()
    return f"""
项目主题：
{project_brief or "-"}

当前固定规划节点：
{node_path_block or "-"}

现有节点说明：
{current_note_block}

请把这个节点补成插画师能直接拿来找参考图的规划内容。

输出 JSON：
{{"note":"一段简洁说明，说明这个节点要明确哪些视觉参考","search_query":"具体的中文语义搜图短语"}}

规则：
1. 不要创建节点，不要改节点名。
2. 只补当前节点，不要把其他节点的内容混进来。
3. search_query 必须能直接用于图文 embedding 搜图。
4. 避免只有剧情或抽象词，要落到可见环境、物件、动作、光线、气氛、材质或构图。
5. 不要输出 Markdown，不要解释 JSON 之外的内容。
""".strip()


def _build_creative_project_copy_prompt(
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
    has_any_detail = any(node["note"] or node["search_query"] for node in clean_nodes)
    nodes_json = json.dumps(clean_nodes, ensure_ascii=False, indent=2)
    if language == "en":
        mode_note = (
            "Some nodes already contain information. Do not contradict, overwrite, or ignore them. "
            "You may only fill empty nodes when it helps the copy."
            if has_any_detail
            else "All nodes are empty. Invent one coherent visual concept and fill every listed node."
        )
        return f"""
Project brief:
{project_brief or "-"}

Planning nodes JSON:
{nodes_json}

Task:
Write a concise visual concept copy for an illustrator. Also return node updates.

Return JSON:
{{
  "copy_text": "one vivid paragraph, 100-220 English words",
  "nodes": [
    {{"title": "existing node title exactly", "note": "node visual detail", "search_query": "concrete English image-search phrase"}}
  ]
}}

Rules:
1. {mode_note}
2. Keep node titles exactly the same as the provided titles.
3. If existing node information exists, the copy must preserve it as hard constraints.
4. Do not add characters, location, weather, era, or events that conflict with existing node notes.
5. Do not output Markdown or any text outside JSON.
""".strip()
    mode_note = (
        "部分节点已经有信息。不得推翻、篡改或忽略已有信息；只在不冲突时补全空节点。"
        if has_any_detail
        else "所有节点都为空。请随机生成一个统一、可画、画面感强的创作主题，并补全每个节点。"
    )
    return f"""
项目主题：
{project_brief or "-"}

规划节点 JSON：
{nodes_json}

任务：
为插画师写一段画面感强的短文，并返回节点补全。

输出 JSON：
{{
  "copy_text": "一段视觉性强的短文，约 180 到 360 个汉字",
  "nodes": [
    {{"title": "必须使用已有节点名", "note": "该节点的视觉信息", "search_query": "具体中文语义搜图短语"}}
  ]
}}

规则：
1. {mode_note}
2. 节点 title 必须和输入中的节点名完全一致。
3. 如果已有节点信息存在，文案必须把它当作硬约束。
4. 不要加入与已有节点说明冲突的角色、地点、天气、年代或事件。
5. 不要输出 Markdown，不要解释 JSON 之外的内容。
""".strip()


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
