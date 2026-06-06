#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import html
import json
import mimetypes
import os
import random
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


DEFAULT_DB_PATH = Path.home() / "Library/Application Support/Eidory/eidory.sqlite3"
DEFAULT_OUTPUT_DIR = Path.home() / "Library/Application Support/Eidory/ai_vision_trials"
DEFAULT_COLLECTION_PATH = "创作参考/全局参考"

ALLOWED_VALUES: dict[str, list[str]] = {
    "scene_location": ["indoor", "outdoor", "threshold", "unknown"],
    "environment_type": ["natural", "built", "mixed", "unknown"],
    "time_of_day": ["day", "night", "dawn_dusk", "artificial_light", "unknown"],
    "weather": ["sunny", "cloudy", "rain", "snow", "fog_haze", "not_visible", "unknown"],
    "shot_scale": ["close_up", "near", "medium", "long", "extreme_long", "unknown"],
    "view_angle": ["eye_level", "high_angle", "low_angle", "bird_eye", "unknown"],
}

LIGHTING_VALUES = [
    "front_light",
    "side_light",
    "back_light",
    "top_light",
    "low_key",
    "high_key",
    "diffuse",
    "unclear",
]

LABEL_ZH = {
    "scene_location": {
        "indoor": "室内",
        "outdoor": "室外",
        "threshold": "室内外过渡",
        "unknown": "无法判断",
    },
    "environment_type": {
        "natural": "自然",
        "built": "人工",
        "mixed": "混合",
        "unknown": "无法判断",
    },
    "time_of_day": {
        "day": "白天",
        "night": "夜晚",
        "dawn_dusk": "清晨黄昏",
        "artificial_light": "人工光为主",
        "unknown": "无法判断",
    },
    "weather": {
        "sunny": "晴",
        "cloudy": "阴",
        "rain": "雨",
        "snow": "雪",
        "fog_haze": "雾霾",
        "not_visible": "不可见",
        "unknown": "无法判断",
    },
    "shot_scale": {
        "close_up": "特写",
        "near": "近景",
        "medium": "中景",
        "long": "远景",
        "extreme_long": "大远景",
        "unknown": "无法判断",
    },
    "view_angle": {
        "eye_level": "平视",
        "high_angle": "俯视",
        "low_angle": "仰视",
        "bird_eye": "鸟瞰",
        "unknown": "无法判断",
    },
    "lighting": {
        "front_light": "正面光",
        "side_light": "侧光",
        "back_light": "逆光",
        "top_light": "顶光",
        "low_key": "低调",
        "high_key": "高调",
        "diffuse": "散射光",
        "unclear": "光照不明显",
    },
}


@dataclass(frozen=True)
class TrialImage:
    id: int
    file_path: str
    file_name: str
    width: int | None
    height: int | None
    file_size: int
    collection_paths: str
    thumbnail_path: str | None


def main() -> int:
    args = parse_args()
    if args.from_jsonl:
        input_jsonl = Path(args.from_jsonl).expanduser()
        if not input_jsonl.exists():
            print(f"jsonl not found: {input_jsonl}", file=sys.stderr)
            return 2
        output_dir = Path(args.output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        results = read_jsonl(input_jsonl)
        summary = build_summary(results, elapsed_seconds=0)
        output_html = output_dir / f"{input_jsonl.stem}_review.html"
        output_summary = output_dir / f"{input_jsonl.stem}_review_summary.json"
        output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        output_html.write_text(
            render_html(results, summary, review_key=input_jsonl.stem),
            encoding="utf-8",
        )
        print(f"html:  {output_html}")
        print(f"summary: {output_summary}")
        return 0

    db_path = Path(args.db).expanduser()
    if not db_path.exists():
        print(f"database not found: {db_path}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_jsonl = output_dir / f"ai_vision_trial_{run_id}.jsonl"
    output_html = output_dir / f"ai_vision_trial_{run_id}.html"
    output_summary = output_dir / f"ai_vision_trial_{run_id}_summary.json"

    settings = load_settings(db_path)
    base_url = args.base_url or settings.get("llm.lm_studio.base_url") or "http://localhost:1234/v1"
    api_key = args.api_key if args.api_key is not None else settings.get("llm.lm_studio.api_key", "")
    model = args.model or settings.get("llm.lm_studio.model") or first_available_model(base_url, api_key)
    temperature = args.temperature
    if temperature is None:
        temperature = safe_float(settings.get("llm.temperature"), 0.1)

    all_images = list_collection_images(db_path, args.collection_path)
    if not all_images:
        print(f"no images found under collection path: {args.collection_path}", file=sys.stderr)
        return 2

    sample = choose_sample(all_images, args.limit, args.seed)
    print(
        f"collection={args.collection_path} available={len(all_images)} sample={len(sample)} "
        f"model={model} base_url={base_url}"
    )

    results: list[dict[str, Any]] = []
    started = time.time()
    for index, image in enumerate(sample, start=1):
        print(f"[{index}/{len(sample)}] {image.file_name}", flush=True)
        record = {
            "image": {
                "id": image.id,
                "file_path": image.file_path,
                "file_name": image.file_name,
                "width": image.width,
                "height": image.height,
                "file_size": image.file_size,
                "collection_paths": image.collection_paths,
                "thumbnail_path": image.thumbnail_path,
            },
            "trial": {
                "collection_path": args.collection_path,
                "model": model,
                "base_url": base_url,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
            "status": "pending",
            "analysis": None,
            "raw_response": None,
            "error": None,
        }
        try:
            raw = analyze_image(
                base_url=base_url,
                model=model,
                api_key=api_key,
                image_path=Path(image.file_path),
                timeout=args.timeout,
                temperature=temperature,
            )
            parsed = parse_analysis(raw)
            record["status"] = "ok"
            record["analysis"] = parsed
            record["raw_response"] = raw
        except Exception as exc:  # noqa: BLE001 - this is an experiment runner.
            record["status"] = "failed"
            record["error"] = str(exc)
        append_jsonl(output_jsonl, record)
        results.append(record)

    summary = build_summary(results, elapsed_seconds=time.time() - started)
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    output_html.write_text(render_html(results, summary, review_key=output_jsonl.stem), encoding="utf-8")
    print(f"jsonl: {output_jsonl}")
    print(f"html:  {output_html}")
    print(f"summary: {output_summary}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an isolated LM Studio visual metadata trial for an Eidory collection."
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--collection-path", default=DEFAULT_COLLECTION_PATH)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=120, help="0 means use every image in the collection subtree.")
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--from-jsonl",
        default=None,
        help="Render an interactive review HTML from an existing trial JSONL without calling the model.",
    )
    return parser.parse_args()


def load_settings(db_path: Path) -> dict[str, str]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT key, value FROM app_settings").fetchall()
    return {str(key): str(value) for key, value in rows}


def list_collection_images(db_path: Path, collection_path: str) -> list[TrialImage]:
    sql = """
        WITH RECURSIVE tree(id, name, parent_id, path) AS (
            SELECT id, name, parent_id, name
            FROM collections
            WHERE parent_id IS NULL
            UNION ALL
            SELECT c.id, c.name, c.parent_id, tree.path || '/' || c.name
            FROM collections c
            JOIN tree ON c.parent_id = tree.id
        ),
        target(id) AS (
            SELECT id FROM tree WHERE path = ?
        ),
        subtree(id) AS (
            SELECT id FROM target
            UNION ALL
            SELECT c.id
            FROM collections c
            JOIN subtree s ON c.parent_id = s.id
        ),
        linked AS (
            SELECT DISTINCT i.id
            FROM images i
            JOIN image_collections ic ON ic.image_id = i.id
            JOIN subtree s ON s.id = ic.collection_id
            WHERE i.is_missing = 0
              AND lower(i.file_ext) IN ('.jpg', '.jpeg', '.png', '.webp')
        )
        SELECT
            i.id,
            i.file_path,
            i.file_name,
            i.width,
            i.height,
            i.file_size,
            i.thumbnail_path,
            GROUP_CONCAT(tree.path, '；') AS collection_paths
        FROM linked
        JOIN images i ON i.id = linked.id
        JOIN image_collections ic ON ic.image_id = i.id
        JOIN tree ON tree.id = ic.collection_id
        GROUP BY i.id
        ORDER BY i.id
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, (collection_path,)).fetchall()
    return [
        TrialImage(
            id=int(row["id"]),
            file_path=str(row["file_path"]),
            file_name=str(row["file_name"]),
            width=row["width"],
            height=row["height"],
            file_size=int(row["file_size"]),
            collection_paths=str(row["collection_paths"] or ""),
            thumbnail_path=row["thumbnail_path"],
        )
        for row in rows
        if Path(str(row["file_path"])).exists()
    ]


def choose_sample(images: list[TrialImage], limit: int, seed: int) -> list[TrialImage]:
    if limit <= 0 or limit >= len(images):
        return images
    groups: dict[str, list[TrialImage]] = {}
    for image in images:
        key = image.collection_paths.split("；")[0] if image.collection_paths else "未归类"
        groups.setdefault(key, []).append(image)
    rng = random.Random(seed)
    for group_images in groups.values():
        rng.shuffle(group_images)
    ordered: list[TrialImage] = []
    while len(ordered) < limit:
        made_progress = False
        for key in sorted(groups):
            if groups[key]:
                ordered.append(groups[key].pop())
                made_progress = True
                if len(ordered) >= limit:
                    break
        if not made_progress:
            break
    return ordered


def analyze_image(
    *,
    base_url: str,
    model: str,
    api_key: str,
    image_path: Path,
    timeout: int,
    temperature: float,
) -> str:
    if not image_path.exists():
        raise FileNotFoundError(str(image_path))
    image_data_url = image_to_data_url(image_path)
    payload: dict[str, Any] = {
        "model": model,
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
                    {
                        "type": "text",
                        "text": build_prompt(),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                ],
            },
        ],
        "temperature": temperature,
        "max_tokens": 900,
        "response_format": vision_response_format(),
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=payload,
        headers=headers,
        timeout=timeout,
    )
    if response.status_code in {400, 422}:
        payload.pop("response_format", None)
        response = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout,
        )
    response.raise_for_status()
    data = response.json()
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"invalid response shape: {data}") from exc
    content = str(message.get("content") or "").strip()
    if not content:
        content = str(message.get("reasoning_content") or "").strip()
    if not content:
        raise ValueError("empty model response")
    return content


def build_prompt() -> str:
    allowed = {
        **ALLOWED_VALUES,
        "lighting": LIGHTING_VALUES,
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
        "- scene_location: 室内/室外/室内外过渡/无法判断。取值 "
        f"{ALLOWED_VALUES['scene_location']}\n"
        "- environment_type: 自然/人工/混合/无法判断。取值 "
        f"{ALLOWED_VALUES['environment_type']}\n"
        "- time_of_day: 白天/夜晚/清晨黄昏/人工光为主/无法判断。取值 "
        f"{ALLOWED_VALUES['time_of_day']}\n"
        "- weather: 晴/阴/雨/雪/雾霾/不可见/无法判断。取值 "
        f"{ALLOWED_VALUES['weather']}\n"
        "- shot_scale: 特写/近景/中景/远景/大远景/无法判断。取值 "
        f"{ALLOWED_VALUES['shot_scale']}\n"
        "- view_angle: 平视/俯视/仰视/鸟瞰/无法判断。取值 "
        f"{ALLOWED_VALUES['view_angle']}\n"
        f"- lighting: 可以多选，取值 {LIGHTING_VALUES}\n"
        "每个单选字段输出 {\"value\":\"...\",\"confidence\":0.0,\"note\":\"...\"}。\n"
        "lighting 输出 [{\"value\":\"...\",\"confidence\":0.0,\"note\":\"...\"}]。\n"
        "notes 输出一句很短的中文说明，专门写不确定或容易误判的地方。\n"
        "不要输出允许值之外的值。\n"
        f"允许值全集：{json.dumps(allowed, ensure_ascii=False)}"
    )


def vision_response_format() -> dict[str, Any]:
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
    lighting = {
        "type": "array",
        "items": scalar,
        "minItems": 1,
        "maxItems": 4,
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
            "lighting": lighting,
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


def parse_analysis(raw: str) -> dict[str, Any]:
    content = raw.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:].strip()
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("model did not return JSON object")
    data = json.loads(content[start : end + 1])
    validate_analysis(data)
    return data


def validate_analysis(data: dict[str, Any]) -> None:
    for key, values in ALLOWED_VALUES.items():
        value = str(data.get(key, {}).get("value", ""))
        if value not in values:
            raise ValueError(f"{key} has invalid value: {value}")
        confidence = data[key].get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError(f"{key} has invalid confidence: {confidence}")
    lighting = data.get("lighting")
    if not isinstance(lighting, list) or not lighting:
        raise ValueError("lighting must be a non-empty list")
    for item in lighting:
        value = str(item.get("value", ""))
        if value not in LIGHTING_VALUES:
            raise ValueError(f"lighting has invalid value: {value}")
        confidence = item.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError(f"lighting has invalid confidence: {confidence}")


def image_to_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    if mime_type.startswith("image/"):
        try:
            from PIL import Image, ImageOps

            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=90, optimize=True)
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                return f"data:image/jpeg;base64,{encoded}"
        except Exception:
            pass
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def first_available_model(base_url: str, api_key: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = requests.get(f"{base_url.rstrip('/')}/models", headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        models = data.get("data", [])
        if models:
            model_id = models[0].get("id")
            if isinstance(model_id, str) and model_id.strip():
                return model_id.strip()
    except requests.RequestException:
        pass
    return "local-model"


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
    return records


def build_summary(results: list[dict[str, Any]], *, elapsed_seconds: float) -> dict[str, Any]:
    ok = [record for record in results if record["status"] == "ok"]
    failed = [record for record in results if record["status"] != "ok"]
    fields: dict[str, dict[str, Any]] = {}
    for key, values in ALLOWED_VALUES.items():
        counts = {value: 0 for value in values}
        confidence_sum = 0.0
        for record in ok:
            analysis = record.get("analysis") or {}
            value = str((analysis.get(key) or {}).get("value") or "")
            if value in counts:
                counts[value] += 1
            confidence_sum += float((analysis.get(key) or {}).get("confidence") or 0)
        fields[key] = {
            "counts": counts,
            "average_confidence": round(confidence_sum / len(ok), 3) if ok else 0,
        }
    lighting_counts = {value: 0 for value in LIGHTING_VALUES}
    for record in ok:
        for item in (record.get("analysis") or {}).get("lighting", []) or []:
            value = str(item.get("value") or "")
            if value in lighting_counts:
                lighting_counts[value] += 1
    return {
        "total": len(results),
        "ok": len(ok),
        "failed": len(failed),
        "json_success_rate": round(len(ok) / len(results), 4) if results else 0,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "fields": fields,
        "lighting_counts": lighting_counts,
        "errors": [record.get("error") for record in failed[:20]],
    }


def render_html(results: list[dict[str, Any]], summary: dict[str, Any], *, review_key: str) -> str:
    cards = "\n".join(render_card(record) for record in results)
    summary_json = html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
    storage_key = html.escape(f"eidory-ai-vision-review:{review_key}", quote=True)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Eidory AI Vision Trial</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #242a31;
      color: #e7ebf1;
    }}
    body {{ margin: 0; padding: 24px; }}
    h1 {{ font-size: 22px; margin: 0 0 16px; }}
    .review-panel {{
      position: sticky;
      top: 0;
      z-index: 20;
      background: rgba(36, 42, 49, 0.96);
      backdrop-filter: blur(10px);
      border: 1px solid #46515f;
      border-radius: 8px;
      padding: 12px;
      margin: 0 0 16px;
      box-shadow: 0 10px 24px rgba(0, 0, 0, 0.22);
    }}
    .review-stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 10px;
      font-size: 13px;
    }}
    .stat {{
      background: #323a44;
      border: 1px solid #4a5664;
      border-radius: 6px;
      padding: 6px 8px;
    }}
    .panel-actions {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    button {{
      background: #3b4551;
      color: #e7ebf1;
      border: 1px solid #596575;
      border-radius: 6px;
      padding: 7px 12px;
      font: inherit;
      cursor: pointer;
    }}
    button:hover {{ background: #465261; }}
    button.active[data-value="usable"] {{ background: #2f7d4f; border-color: #48a26c; }}
    button.active[data-value="ambiguous"] {{ background: #8a6a26; border-color: #b18a38; }}
    button.active[data-value="wrong"] {{ background: #923f46; border-color: #be5a63; }}
    button.secondary {{ font-size: 12px; padding: 6px 9px; }}
    pre {{
      white-space: pre-wrap;
      background: #171b20;
      border: 1px solid #3e4855;
      border-radius: 6px;
      padding: 12px;
      margin-bottom: 18px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
      gap: 14px;
    }}
    .card {{
      background: #2d343d;
      border: 1px solid #46515f;
      border-radius: 8px;
      overflow: hidden;
    }}
    .card.review-usable {{ border-color: #48a26c; }}
    .card.review-ambiguous {{ border-color: #b18a38; }}
    .card.review-wrong {{ border-color: #be5a63; }}
    .image {{
      height: 260px;
      background: #15191e;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    img {{ max-width: 100%; max-height: 260px; object-fit: contain; }}
    .body {{ padding: 12px; }}
    .name {{ font-weight: 700; margin-bottom: 6px; }}
    .meta {{ color: #aeb7c4; font-size: 12px; line-height: 1.35; margin-bottom: 10px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    td {{
      border-top: 1px solid #404a56;
      padding: 5px 3px;
      vertical-align: top;
    }}
    td:first-child {{ color: #aeb7c4; width: 92px; }}
    .error {{ color: #ffadad; }}
    .note {{ color: #c6ced9; font-size: 12px; margin-top: 8px; }}
    .value {{ font-weight: 600; }}
    .confidence {{ color: #97d08a; }}
    .review-controls {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 6px;
      margin: 10px 0;
    }}
    .review-controls button {{ padding: 8px 6px; }}
  </style>
</head>
<body>
  <h1>Eidory AI Vision Trial</h1>
  <section class="review-panel">
    <div class="review-stats">
      <div class="stat">总数：<strong id="stat-total">0</strong></div>
      <div class="stat">已审：<strong id="stat-reviewed">0</strong></div>
      <div class="stat">可用：<strong id="stat-usable">0</strong></div>
      <div class="stat">模糊：<strong id="stat-ambiguous">0</strong></div>
      <div class="stat">错误：<strong id="stat-wrong">0</strong></div>
      <div class="stat">错误率：<strong id="stat-error-rate">0%</strong></div>
    </div>
    <div class="panel-actions">
      <button class="secondary" type="button" id="export-review">导出审查结果 JSON</button>
      <button class="secondary" type="button" id="copy-review">复制审查结果</button>
      <button class="secondary" type="button" id="reset-review">清空本页审查</button>
    </div>
  </section>
  <pre>{summary_json}</pre>
  <div class="grid">
    {cards}
  </div>
  <script>
    const storageKey = "{storage_key}";
    const labels = {{
      usable: "可用",
      ambiguous: "模糊",
      wrong: "错误"
    }};
    function loadReview() {{
      try {{
        return JSON.parse(localStorage.getItem(storageKey) || "{{}}");
      }} catch (_error) {{
        return {{}};
      }}
    }}
    function saveReview(review) {{
      localStorage.setItem(storageKey, JSON.stringify(review));
    }}
    function setCardState(card, value) {{
      card.classList.remove("review-usable", "review-ambiguous", "review-wrong");
      for (const button of card.querySelectorAll(".review-controls button")) {{
        button.classList.toggle("active", button.dataset.value === value);
      }}
      if (value) {{
        card.classList.add("review-" + value);
      }}
    }}
    function collectExportPayload() {{
      const review = loadReview();
      const records = [];
      for (const card of document.querySelectorAll(".card[data-image-id]")) {{
        const imageId = card.dataset.imageId;
        records.push({{
          image_id: Number(imageId),
          file_name: card.dataset.fileName,
          review: review[imageId] || "",
        }});
      }}
      const total = records.length;
      const usable = records.filter((item) => item.review === "usable").length;
      const ambiguous = records.filter((item) => item.review === "ambiguous").length;
      const wrong = records.filter((item) => item.review === "wrong").length;
      return {{
        storage_key: storageKey,
        exported_at: new Date().toISOString(),
        summary: {{
          total,
          reviewed: usable + ambiguous + wrong,
          usable,
          ambiguous,
          wrong,
          error_rate: total ? wrong / total : 0,
        }},
        records,
      }};
    }}
    function updateStats() {{
      const review = loadReview();
      const cards = Array.from(document.querySelectorAll(".card[data-image-id]"));
      let usable = 0;
      let ambiguous = 0;
      let wrong = 0;
      for (const card of cards) {{
        const value = review[card.dataset.imageId] || "";
        setCardState(card, value);
        if (value === "usable") usable += 1;
        if (value === "ambiguous") ambiguous += 1;
        if (value === "wrong") wrong += 1;
      }}
      const reviewed = usable + ambiguous + wrong;
      document.getElementById("stat-total").textContent = String(cards.length);
      document.getElementById("stat-reviewed").textContent = String(reviewed);
      document.getElementById("stat-usable").textContent = String(usable);
      document.getElementById("stat-ambiguous").textContent = String(ambiguous);
      document.getElementById("stat-wrong").textContent = String(wrong);
      const rate = reviewed ? Math.round((wrong / reviewed) * 1000) / 10 : 0;
      document.getElementById("stat-error-rate").textContent = rate + "%";
    }}
    document.addEventListener("click", async (event) => {{
      const button = event.target.closest("button");
      if (!button) return;
      const reviewButton = button.closest(".review-controls button");
      if (reviewButton) {{
        const card = button.closest(".card[data-image-id]");
        const review = loadReview();
        const imageId = card.dataset.imageId;
        const value = button.dataset.value;
        if (review[imageId] === value) {{
          delete review[imageId];
        }} else {{
          review[imageId] = value;
        }}
        saveReview(review);
        updateStats();
        return;
      }}
      if (button.id === "export-review") {{
        const payload = collectExportPayload();
        const blob = new Blob([JSON.stringify(payload, null, 2)], {{type: "application/json"}});
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = storageKey.replace(/[^a-zA-Z0-9_-]+/g, "_") + ".json";
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
        return;
      }}
      if (button.id === "copy-review") {{
        const payload = collectExportPayload();
        await navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
        button.textContent = "已复制";
        setTimeout(() => button.textContent = "复制审查结果", 1200);
        return;
      }}
      if (button.id === "reset-review") {{
        if (confirm("清空这个审查页的所有人工判断？")) {{
          localStorage.removeItem(storageKey);
          updateStats();
        }}
      }}
    }});
    updateStats();
  </script>
</body>
</html>
"""


def render_card(record: dict[str, Any]) -> str:
    image = record["image"]
    preview_path = image.get("thumbnail_path") or image.get("file_path")
    preview_uri = Path(str(preview_path)).as_uri() if preview_path and Path(str(preview_path)).exists() else ""
    title = html.escape(str(image.get("file_name") or "-"))
    meta = html.escape(
        f"id {image.get('id')} | {image.get('width') or '-'} x {image.get('height') or '-'} | "
        f"{image.get('collection_paths') or '-'}"
    )
    if record["status"] != "ok":
        content = f'<div class="error">{html.escape(str(record.get("error") or "failed"))}</div>'
    else:
        analysis = record["analysis"]
        rows = [
            render_row("室内外", "scene_location", analysis),
            render_row("环境", "environment_type", analysis),
            render_row("时间", "time_of_day", analysis),
            render_row("天气", "weather", analysis),
            render_row("景别", "shot_scale", analysis),
            render_row("视角", "view_angle", analysis),
            render_lighting_row(analysis),
        ]
        notes = html.escape(str(analysis.get("notes") or ""))
        content = f"<table>{''.join(rows)}</table><div class=\"note\">{notes}</div>"
    image_html = f'<img src="{html.escape(preview_uri)}" alt="">' if preview_uri else "无预览"
    image_id = html.escape(str(image.get("id") or ""))
    file_name = html.escape(str(image.get("file_name") or ""), quote=True)
    return f"""
    <section class="card" data-image-id="{image_id}" data-file-name="{file_name}">
      <div class="image">{image_html}</div>
      <div class="body">
        <div class="name">{title}</div>
        <div class="meta">{meta}</div>
        <div class="review-controls">
          <button type="button" data-value="usable">可用</button>
          <button type="button" data-value="ambiguous">模糊</button>
          <button type="button" data-value="wrong">错误</button>
        </div>
        {content}
      </div>
    </section>
    """


def render_row(label: str, key: str, analysis: dict[str, Any]) -> str:
    item = analysis.get(key) or {}
    value = str(item.get("value") or "unknown")
    note = str(item.get("note") or "")
    confidence = float(item.get("confidence") or 0)
    zh = LABEL_ZH[key].get(value, value)
    return (
        f"<tr><td>{html.escape(label)}</td><td>"
        f"<span class=\"value\">{html.escape(zh)}</span> "
        f"<span class=\"confidence\">{confidence:.2f}</span>"
        f"<div class=\"note\">{html.escape(note)}</div>"
        "</td></tr>"
    )


def render_lighting_row(analysis: dict[str, Any]) -> str:
    parts = []
    for item in analysis.get("lighting", []) or []:
        value = str(item.get("value") or "unclear")
        confidence = float(item.get("confidence") or 0)
        zh = LABEL_ZH["lighting"].get(value, value)
        parts.append(f"{html.escape(zh)} <span class=\"confidence\">{confidence:.2f}</span>")
    return f"<tr><td>光照</td><td>{'；'.join(parts)}</td></tr>"


def safe_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


if __name__ == "__main__":
    raise SystemExit(main())
