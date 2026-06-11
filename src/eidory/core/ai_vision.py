from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from eidory.core.image_loader import MAX_AI_VISION_UPLOAD_BYTES, open_local_image


AI_VISION_PROMPT_VERSION = "scene-v1"

AI_VISION_FIELD_VALUES: dict[str, list[str]] = {
    "scene_location": ["indoor", "outdoor", "threshold", "unknown"],
    "environment_type": ["natural", "built", "mixed", "unknown"],
    "time_of_day": ["day", "night", "dawn_dusk", "artificial_light", "unknown"],
    "weather": ["sunny", "cloudy", "rain", "snow", "fog_haze", "not_visible", "unknown"],
    "shot_scale": ["close_up", "near", "medium", "long", "extreme_long", "unknown"],
    "view_angle": ["eye_level", "high_angle", "low_angle", "bird_eye", "unknown"],
}

AI_VISION_LIGHTING_VALUES = [
    "front_light",
    "side_light",
    "back_light",
    "top_light",
    "low_key",
    "high_key",
    "diffuse",
    "unclear",
]

AI_VISION_VALUE_ALIASES: dict[str, dict[str, str]] = {
    "scene_location": {
        "indoors": "indoor",
        "interior": "indoor",
        "outdoors": "outdoor",
        "exterior": "outdoor",
        "semi_outdoor": "threshold",
        "semi-outdoor": "threshold",
        "transitional": "threshold",
    },
    "environment_type": {
        "manmade": "built",
        "man-made": "built",
        "human_made": "built",
        "urban": "built",
        "city": "built",
    },
    "time_of_day": {
        "dawn": "dawn_dusk",
        "dusk": "dawn_dusk",
        "sunrise": "dawn_dusk",
        "sunset": "dawn_dusk",
        "evening": "dawn_dusk",
        "indoor_light": "artificial_light",
    },
    "weather": {
        "clear": "sunny",
        "clear_sky": "sunny",
        "overcast": "cloudy",
        "fog": "fog_haze",
        "haze": "fog_haze",
        "mist": "fog_haze",
        "not_applicable": "not_visible",
        "none": "not_visible",
    },
    "shot_scale": {
        "close-up": "close_up",
        "closeup": "close_up",
        "full_body": "long",
        "full_shot": "long",
        "wide": "long",
        "wide_shot": "long",
        "establishing_shot": "extreme_long",
        "very_long": "extreme_long",
    },
    "view_angle": {
        "normal": "eye_level",
        "straight_on": "eye_level",
        "front": "eye_level",
        "overhead": "bird_eye",
        "aerial": "bird_eye",
    },
    "lighting": {
        "artificial_light": "unclear",
        "ambient": "diffuse",
        "ambient_light": "diffuse",
        "soft_light": "diffuse",
        "flat_light": "diffuse",
        "silhouette": "back_light",
        "rim_light": "back_light",
    },
}

AI_VISION_FIELD_LABELS_ZH = {
    "scene_location": "室内外",
    "environment_type": "环境",
    "time_of_day": "时间",
    "weather": "天气",
    "shot_scale": "景别",
    "view_angle": "视角",
    "lighting": "光照",
}

AI_VISION_VALUE_LABELS_ZH = {
    "indoor": "室内",
    "outdoor": "室外",
    "threshold": "室内外过渡",
    "natural": "自然",
    "built": "人工",
    "mixed": "混合",
    "day": "白天",
    "night": "夜晚",
    "dawn_dusk": "清晨黄昏",
    "artificial_light": "人工光为主",
    "sunny": "晴",
    "cloudy": "阴",
    "rain": "雨",
    "snow": "雪",
    "fog_haze": "雾霾",
    "not_visible": "不可见",
    "close_up": "特写",
    "near": "近景",
    "medium": "中景",
    "long": "远景",
    "extreme_long": "大远景",
    "eye_level": "平视",
    "high_angle": "俯视",
    "low_angle": "仰视",
    "bird_eye": "鸟瞰",
    "front_light": "正面光",
    "side_light": "侧光",
    "back_light": "逆光",
    "top_light": "顶光",
    "low_key": "低调",
    "high_key": "高调",
    "diffuse": "散射光",
    "unclear": "光照不明显",
    "unknown": "无法判断",
}

AI_VISION_FIELD_LABELS_EN = {
    "scene_location": "Location",
    "environment_type": "Environment",
    "time_of_day": "Time",
    "weather": "Weather",
    "shot_scale": "Scale",
    "view_angle": "Angle",
    "lighting": "Lighting",
}

AI_VISION_VALUE_LABELS_EN = {
    "indoor": "Indoor",
    "outdoor": "Outdoor",
    "threshold": "Threshold",
    "natural": "Natural",
    "built": "Built",
    "mixed": "Mixed",
    "day": "Day",
    "night": "Night",
    "dawn/dusk": "Dawn/Dusk",
    "dawn_dusk": "Dawn/Dusk",
    "artificial_light": "Artificial light",
    "sunny": "Sunny",
    "cloudy": "Cloudy",
    "rain": "Rain",
    "snow": "Snow",
    "fog_haze": "Fog/Haze",
    "not_visible": "Not visible",
    "close_up": "Close-up",
    "near": "Near",
    "medium": "Medium",
    "long": "Long",
    "extreme_long": "Extreme long",
    "eye_level": "Eye level",
    "high_angle": "High angle",
    "low_angle": "Low angle",
    "bird_eye": "Bird's-eye",
    "front_light": "Front light",
    "side_light": "Side light",
    "back_light": "Back light",
    "top_light": "Top light",
    "low_key": "Low key",
    "high_key": "High key",
    "diffuse": "Diffuse",
    "unclear": "Unclear",
    "unknown": "Unknown",
}


@dataclass(frozen=True)
class AIVisionAnalysis:
    scene_location: str
    environment_type: str
    time_of_day: str
    weather: str
    shot_scale: str
    view_angle: str
    lighting: list[str]
    confidence: dict[str, float]
    notes: str
    raw_json: dict[str, Any]


class AIVisionProviderError(RuntimeError):
    pass


class AIVisionProvider:
    def __init__(
        self,
        *,
        base_url: str,
        model_name: str | None,
        api_key: str = "",
        service_name: str = "LM Studio",
        temperature: float = 0.1,
        timeout_seconds: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = (model_name or "").strip()
        self.api_key = api_key.strip()
        self.service_name = service_name.strip() or "LLM"
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.provider_name = self.service_name

    def analyze_image(self, image_path: str | Path) -> AIVisionAnalysis:
        path = Path(image_path)
        if not path.exists():
            raise AIVisionProviderError(f"源文件不存在：{path}")
        model_name = self.model_name or self._first_available_model()
        raw = self._chat_completion(model_name=model_name, image_path=path)
        return parse_ai_vision_analysis(raw)

    def resolved_model_name(self) -> str:
        if self.model_name:
            return self.model_name
        self.model_name = self._first_available_model()
        return self.model_name

    def _chat_completion(self, *, model_name: str, image_path: Path) -> str:
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是视觉元数据标注器。只输出严格 JSON，不要 Markdown，不要解释。"
                        "不要识别人物身份、艺术家、IP、年代、职业或故事剧情。"
                        "只能根据画面可见信息选择受控字段；看不清或不适用必须用 unknown/not_visible/unclear。"
                        "confidence 必须是 0 到 1 的数字。"
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_ai_vision_prompt()},
                        {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                    ],
                },
            ],
            "temperature": self.temperature,
            "max_tokens": 900,
            "response_format": ai_vision_response_format(),
        }
        headers = self._headers()
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            if response.status_code in {400, 422}:
                payload.pop("response_format", None)
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise AIVisionProviderError(f"{self.service_name} 请求失败：{exc}") from exc
        try:
            message = response.json()["choices"][0]["message"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise AIVisionProviderError("视觉模型返回格式无效") from exc
        content = str(message.get("content") or "").strip()
        if not content:
            content = str(message.get("reasoning_content") or "").strip()
        if not content:
            raise AIVisionProviderError("视觉模型返回了空内容")
        return content

    def _first_available_model(self) -> str:
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self._headers(),
                timeout=5,
            )
            response.raise_for_status()
            models = response.json().get("data", [])
            if models:
                model_id = models[0].get("id")
                if isinstance(model_id, str) and model_id.strip():
                    return model_id.strip()
        except requests.RequestException:
            pass
        return "local-vision-model"

    def _headers(self) -> dict[str, str]:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}


def build_ai_vision_prompt() -> str:
    allowed = {
        **AI_VISION_FIELD_VALUES,
        "lighting": AI_VISION_LIGHTING_VALUES,
    }
    return (
        "分析这张图，只返回一个 JSON 对象。\n"
        "关键定义：\n"
        "- indoor 只用于人造建筑内部、房间内部、车船舱内等人工封闭空间。\n"
        "- outdoor 用于露天自然/城市/街道/山水环境。\n"
        "- threshold 用于门口、窗边、阳台、廊下、半开放棚屋、洞口、峡谷/洞穴这类室内外边界或半封闭空间。\n"
        "- 不要把自然峡谷、洞穴、水潭、岩壁包围的空间直接判成 indoor。\n"
        "- weather 只有在天空或外部天气线索可见时判断；否则用 not_visible 或 unknown。\n"
        "字段必须严格如下：\n"
        f"- scene_location 取值 {AI_VISION_FIELD_VALUES['scene_location']}\n"
        f"- environment_type 取值 {AI_VISION_FIELD_VALUES['environment_type']}\n"
        f"- time_of_day 取值 {AI_VISION_FIELD_VALUES['time_of_day']}\n"
        f"- weather 取值 {AI_VISION_FIELD_VALUES['weather']}\n"
        f"- shot_scale 取值 {AI_VISION_FIELD_VALUES['shot_scale']}\n"
        f"- view_angle 取值 {AI_VISION_FIELD_VALUES['view_angle']}\n"
        f"- lighting 可以多选，取值 {AI_VISION_LIGHTING_VALUES}\n"
        "每个单选字段输出 {\"value\":\"...\",\"confidence\":0.0,\"note\":\"...\"}。\n"
        "lighting 输出 [{\"value\":\"...\",\"confidence\":0.0,\"note\":\"...\"}]。\n"
        "notes 输出一句很短的中文说明，专门写不确定或容易误判的地方。\n"
        "不要输出允许值之外的值。\n"
        f"允许值全集：{json.dumps(allowed, ensure_ascii=False)}"
    )


def ai_vision_response_format() -> dict[str, Any]:
    scalar = {
        "type": "object",
        "properties": {
            "value": {"type": "string"},
            "confidence": {"type": "number"},
            "note": {"type": "string"},
        },
        "required": ["value", "confidence", "note"],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "scene_location": scalar,
            "environment_type": scalar,
            "time_of_day": scalar,
            "weather": scalar,
            "shot_scale": scalar,
            "view_angle": scalar,
            "lighting": {"type": "array", "items": scalar, "minItems": 1, "maxItems": 4},
            "notes": {"type": "string"},
        },
        "required": [
            "scene_location",
            "environment_type",
            "time_of_day",
            "weather",
            "shot_scale",
            "view_angle",
            "lighting",
            "notes",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "eidory_ai_vision_tags",
            "strict": True,
            "schema": schema,
        },
    }


def parse_ai_vision_analysis(raw: str) -> AIVisionAnalysis:
    content = raw.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:].strip()
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise AIVisionProviderError("视觉模型没有返回 JSON 对象")
    try:
        data = json.loads(content[start : end + 1])
    except json.JSONDecodeError as exc:
        raise AIVisionProviderError(f"视觉模型 JSON 无效：{exc}") from exc
    return normalize_ai_vision_analysis(data)


def normalize_ai_vision_analysis(data: dict[str, Any]) -> AIVisionAnalysis:
    values: dict[str, str] = {}
    confidence: dict[str, float] = {}
    for field, allowed in AI_VISION_FIELD_VALUES.items():
        item = data.get(field)
        if not isinstance(item, dict):
            raise AIVisionProviderError(f"{field} 缺失")
        value = normalize_ai_vision_value(field, item.get("value"))
        if value not in allowed:
            raise AIVisionProviderError(f"{field} 取值无效：{value}")
        conf = item.get("confidence")
        if not isinstance(conf, (int, float)) or not 0 <= float(conf) <= 1:
            raise AIVisionProviderError(f"{field} 置信度无效：{conf}")
        values[field] = value
        confidence[field] = float(conf)

    lighting_raw = data.get("lighting")
    if not isinstance(lighting_raw, list) or not lighting_raw:
        raise AIVisionProviderError("lighting 缺失")
    lighting: list[str] = []
    for item in lighting_raw:
        if not isinstance(item, dict):
            raise AIVisionProviderError("lighting 项无效")
        value = normalize_ai_vision_value("lighting", item.get("value"))
        if value not in AI_VISION_LIGHTING_VALUES:
            raise AIVisionProviderError(f"lighting 取值无效：{value}")
        conf = item.get("confidence")
        if not isinstance(conf, (int, float)) or not 0 <= float(conf) <= 1:
            raise AIVisionProviderError(f"lighting 置信度无效：{conf}")
        if value not in lighting:
            lighting.append(value)
        confidence[f"lighting:{value}"] = float(conf)

    return AIVisionAnalysis(
        scene_location=values["scene_location"],
        environment_type=values["environment_type"],
        time_of_day=values["time_of_day"],
        weather=values["weather"],
        shot_scale=values["shot_scale"],
        view_angle=values["view_angle"],
        lighting=lighting,
        confidence=confidence,
        notes=str(data.get("notes") or "").strip()[:1000],
        raw_json=data,
    )


def normalize_ai_vision_value(field: str, value: object) -> str:
    normalized = str(value or "").strip().lower().replace(" ", "_")
    aliases = AI_VISION_VALUE_ALIASES.get(field, {})
    return aliases.get(normalized, normalized)


def image_to_data_url(path: Path) -> str:
    try:
        from PIL import Image, ImageOps

        with open_local_image(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
            for quality in (88, 78, 68, 58):
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=quality, optimize=True)
                payload = buffer.getvalue()
                if len(payload) <= MAX_AI_VISION_UPLOAD_BYTES or quality == 58:
                    if len(payload) > MAX_AI_VISION_UPLOAD_BYTES:
                        raise AIVisionProviderError(
                            f"AI 视觉缩略图仍过大：{len(payload):,} bytes"
                        )
                    encoded = base64.b64encode(payload).decode("ascii")
                    return f"data:image/jpeg;base64,{encoded}"
    except AIVisionProviderError:
        raise
    except Exception as exc:
        raise AIVisionProviderError(f"无法生成 AI 视觉缩略图：{path.name}: {exc}") from exc
    raise AIVisionProviderError(f"无法生成 AI 视觉缩略图：{path.name}")


def ai_vision_label(field: str, value: str, *, language: str = "zh") -> str:
    field_labels = AI_VISION_FIELD_LABELS_EN if language == "en" else AI_VISION_FIELD_LABELS_ZH
    value_labels = AI_VISION_VALUE_LABELS_EN if language == "en" else AI_VISION_VALUE_LABELS_ZH
    field_label = field_labels.get(field, field)
    value_label = value_labels.get(value, value)
    return f"{field_label}: {value_label}"
