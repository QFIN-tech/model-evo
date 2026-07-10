from __future__ import annotations

from typing import Any

DEFAULT_REPORT_LANGUAGE = "zh-CN"
SUPPORTED_REPORT_LANGUAGES = {"zh-CN", "en-US"}

_ALIASES = {
    "zh": "zh-CN",
    "zh-cn": "zh-CN",
    "zh_cn": "zh-CN",
    "chinese": "zh-CN",
    "中文": "zh-CN",
    "en": "en-US",
    "en-us": "en-US",
    "en_us": "en-US",
    "english": "en-US",
}


def normalize_report_language(value: Any) -> tuple[str, str, dict[str, Any] | None]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return DEFAULT_REPORT_LANGUAGE, "default", None
    raw = str(value).strip()
    language = _ALIASES.get(raw.lower(), raw)
    if language in SUPPORTED_REPORT_LANGUAGES:
        return language, "explicit", None
    return (
        DEFAULT_REPORT_LANGUAGE,
        "default_after_unsupported",
        {
            "code": "UNSUPPORTED_REPORT_LANGUAGE",
            "level": "warning",
            "blocking": False,
            "message": (
                f"Unsupported report language `{raw}`; fallback to {DEFAULT_REPORT_LANGUAGE}."
            ),
            "suggested_fix": "Use zh-CN or en-US.",
        },
    )


def normalize_report_preferences(value: Any) -> tuple[dict[str, Any], dict[str, Any] | None]:
    raw = value if isinstance(value, dict) else {}
    language, source, warning = normalize_report_language(raw.get("language"))
    preferences = dict(raw)
    preferences["language"] = language
    preferences["language_source"] = raw.get("language_source") or source
    return preferences, warning


def is_zh(language: str) -> bool:
    return language == "zh-CN"
