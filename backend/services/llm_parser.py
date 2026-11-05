"""UNIT 1 + 3: natural language -> validated JSON {category, class_id, count}.

Two strategies:
  * Gemini LLM (zero-shot or few-shot prompt) when GEMINI_API_KEY is set.
  * A deterministic rule-based parser used as fallback, so the app also works
    offline / without a key. The response always says which one produced the result.
"""
import json
import re
from typing import Optional

import httpx
from pydantic import BaseModel, Field, ValidationError

import config

# ---- synonyms -> Fashion-MNIST class id --------------------------------------
SYNONYMS = {
    0: ["t-shirt", "tshirt", "t shirt", "tee", "tees", "top", "tops"],
    1: ["trouser", "pant", "pants", "jeans", "slacks"],
    2: ["pullover", "sweater", "jumper", "hoodie", "sweatshirt"],
    3: ["dress", "gown", "frock"],
    4: ["coat", "jacket", "parka", "overcoat"],
    5: ["sandal", "flip flop", "flip-flop", "slipper"],
    6: ["shirt", "blouse"],
    7: ["sneaker", "trainer", "running shoe", "shoe"],
    8: ["handbag", "bag", "purse", "backpack", "tote"],
    9: ["ankle boot", "boot", "bootie"],
}
NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "single": 1, "two": 2, "pair": 2, "three": 3, "four": 4,
                "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "eleven": 11, "twelve": 12}


class ParsedRequest(BaseModel):
    """The validated contract between the LLM and the generation model."""
    category: str
    class_id: int = Field(ge=0, le=9)
    count: int = Field(ge=1, le=config.MAX_IMAGES)


class ParseError(ValueError):
    pass


def category_to_id(text: str) -> Optional[int]:
    """Longest synonym wins ('ankle boot' before 'boot', 'shirt' vs 't-shirt')."""
    text = text.lower()
    best, best_len = None, 0
    for cid, words in SYNONYMS.items():
        for w in words:
            plural = r"(?:e?s)?"
            if re.search(rf"\b{re.escape(w)}{plural}\b", text) and len(w) > best_len:
                best, best_len = cid, len(w)
    return best


def extract_count(text: str) -> int:
    text = text.lower()
    m = re.search(r"\b(\d{1,3})\b", text)
    if m:
        return int(m.group(1))
    for word, val in NUMBER_WORDS.items():
        if word in ("a", "an") :
            continue
        if re.search(rf"\b{word}\b", text):
            return val
    return 1


def rule_based_parse(prompt: str) -> dict:
    cid = category_to_id(prompt)
    if cid is None:
        raise ParseError("I couldn't find a fashion category. Try: T-shirt, trouser, pullover, "
                         "dress, coat, sandal, shirt, sneaker, bag or ankle boot.")
    return {"category": config.CLASS_NAMES[cid].lower(), "class_id": cid, "count": extract_count(prompt)}


# ---- LLM prompts (Unit 1: zero-shot vs few-shot) -----------------------------
CLASS_LIST = "\n".join(f"{i} = {n}" for i, n in enumerate(config.CLASS_NAMES))
ZERO_SHOT = ("You convert fashion image requests into JSON. Categories:\n" + CLASS_LIST +
             '\nReturn ONLY JSON: {"category": "<name>", "class_id": <int>, "count": <int>}. '
             'If no category matches, return {"category": null, "class_id": null, "count": 0}.\n'
             "Request: ")
FEW_SHOT = ZERO_SHOT.rsplit("Request: ", 1)[0] + (
    'Examples:\nRequest: Show me four ankle boots\n{"category": "ankle boot", "class_id": 9, "count": 4}\n'
    'Request: Make a couple of dresses\n{"category": "dress", "class_id": 3, "count": 2}\n'
    'Request: I want a handbag\n{"category": "bag", "class_id": 8, "count": 1}\n'
    'Request: draw a spaceship\n{"category": null, "class_id": null, "count": 0}\nRequest: ')


def _gemini_call(prompt_text: str, timeout: float = 15.0) -> dict:
    model_name = config.GEMINI_MODEL.replace("models/", "")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    body = {"contents": [{"parts": [{"text": prompt_text}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
    r = httpx.post(url, json=body, headers={"x-goog-api-key": config.GEMINI_API_KEY}, timeout=timeout)
    r.raise_for_status()
    text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    text = re.sub(r"^```(?:json)?|```$", "", text.strip()).strip()
    return json.loads(text)


def _validate(raw: dict) -> ParsedRequest:
    if raw.get("category") is None or raw.get("class_id") is None:
        raise ParseError("The request does not mention a supported fashion category.")
    try:
        parsed = ParsedRequest(**raw)
    except ValidationError as exc:
        msg = "; ".join(e["msg"] for e in exc.errors())
        raise ParseError(f"Invalid request values: {msg} (max {config.MAX_IMAGES} images).")
    parsed.category = config.CLASS_NAMES[parsed.class_id].lower()  # trust the id, not LLM spelling
    return parsed


def parse_prompt(prompt: str, strategy: str = "few_shot", compare: bool = False) -> dict:
    prompt = (prompt or "").strip()
    if len(prompt) < 3:
        raise ParseError("Please enter a request such as 'Show me four ankle boots'.")
    if len(prompt) > 300:
        raise ParseError("Request is too long (max 300 characters).")

    def run_llm(strat):
        template = FEW_SHOT if strat == "few_shot" else ZERO_SHOT
        return _gemini_call(template + prompt)

    result = {"prompt": prompt, "strategy": strategy, "source": "rule-based", "llm_available": bool(config.GEMINI_API_KEY)}
    comparison = {}
    if config.GEMINI_API_KEY:
        strategies = ["zero_shot", "few_shot"] if compare else [strategy]
        for strat in strategies:
            try:
                raw = run_llm(strat)
                comparison[strat] = {"raw": raw, "valid": True}
                comparison[strat]["parsed"] = _validate(raw).model_dump()
            except ParseError as exc:
                comparison[strat] = {"raw": raw, "valid": False, "error": str(exc)}
            except Exception as exc:                      # network, quota, bad JSON ...
                comparison[strat] = {"raw": None, "valid": False, "error": f"LLM call failed: {type(exc).__name__}"}
        chosen = comparison.get(strategy)
        if chosen and chosen["valid"]:
            result.update(source=f"gemini:{config.GEMINI_MODEL}", **chosen["parsed"])
            result["comparison"] = comparison if compare else None
            return result
        if chosen and not chosen["valid"] and chosen.get("raw") is not None:
            raise ParseError(chosen["error"])    # LLM answered but request was invalid
        result["llm_error"] = chosen["error"] if chosen else None
    # fallback
    parsed = _validate(rule_based_parse(prompt))
    result.update(parsed.model_dump())
    result["comparison"] = comparison or None
    return result
