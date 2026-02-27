import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from google import genai
from google.genai import types

APP_TITLE = "Claim Validator (Experimental)"
MAX_FILES = 3
SUPPORTED_TYPES = ["mp4", "mov", "m4a", "mp3", "wav", "webm", "mpeg", "mpga"]
DEFAULT_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """
You are a strict claim-validation engine for short-form social content.

NON-NEGOTIABLE RULES:
1) Use Google Search grounding tools in this request. Prefer grounded evidence over prior knowledge.
2) Do NOT diagnose any person psychologically or medically. Never produce clinical diagnosis from social content.
3) Distinguish clearly between:
   - FACTUAL CLAIM (verifiable)
   - PROFESSIONAL/TECHNICAL CLAIM (requires domain-specific evidence)
   - INTERPRETATION (inference)
   - OPINION (subjective)
   - RHETORICAL/MANIPULATIVE framing (persuasion, overgeneralization, fear language, Barnum-style language, false authority, cherry-picking)
4) If evidence is insufficient, say so explicitly with status NOT_ENOUGH_EVIDENCE.
5) Do not overstate certainty. Add confidence (high/medium/low) and a short reason.
6) Cite sources used for each validated claim.
7) Separate what is true from what is misleading in the same statement (partial truth handling).
8) Be concise, structured, and product-like. No motivational language.
9) Do NOT shame creators. Validate claims, not people.
10) When the topic involves mental health / psychology:
    - Avoid diagnostic language toward the viewer or creator.
    - Clarify when terms are pop-psychology vs recognized clinical/professional terminology.
    - Mention when a concept exists but is used inaccurately.

OUTPUT FORMAT (STRICT):
- Return STRICT JSON only.
- Do NOT wrap in ``` fences.
- Do NOT add any prose before or after the JSON.
- Do NOT repeat the JSON twice.
- LIMITS: max 5 claims total; max 2 sources per claim.
- IMPORTANT: To reduce formatting failures, do NOT include full URLs in the JSON.
  Use source entries with title + source_hint only (domain or publisher name) + evidence_note.
  Grounding metadata will be captured separately by the application.

Return JSON matching this schema exactly:
{
  "summary": {
    "overall_verdict": "mostly_accurate|mixed|mostly_misleading|not_enough_evidence",
    "why": "short explanation",
    "confidence": "high|medium|low"
  },
  "claims": [
    {
      "claim_text": "...",
      "claim_type": "factual|professional|interpretation|opinion|rhetorical",
      "status": "accurate|misleading|partially_accurate|not_enough_evidence|opinion_not_verifiable",
      "explanation": "short, concrete",
      "confidence": "high|medium|low",
      "red_flags": ["barnum_effect", "overgeneralization", "false_authority", "cherry_picking", "none"],
      "sources": [
        {"title": "...", "source_hint": "publisher/domain", "evidence_note": "..."}
      ]
    }
  ],
  "creator_style_signals": {
    "uses_absolute_language": true,
    "uses_emotional_certainty": false,
    "uses_professional_sounding_terms": true,
    "notes": "short"
  },
  "user_relevance": {
    "requested_comparison_done": true,
    "note": "If the user asked whether it applies to them, explain only in non-diagnostic behavioral terms and mention limits."
  }
}
""".strip()

NICHE_RULESETS = {
    "general": "General claim validation. Prioritize official primary sources and high-trust institutions relevant to topic.",
    "psychology": (
        "Psychology niche rules: prioritize APA, DSM/ICD-related references, major health systems, universities, and peer-reviewed sources. "
        "Do not diagnose. Distinguish clinical terms from pop-psychology language. Flag Barnum/Forer effects and overgeneralization when present."
    ),
    "nutrition": (
        "Nutrition niche rules: prioritize WHO, NIH, national health authorities, registered-dietitian consensus resources, and peer-reviewed studies/meta-analyses. "
        "Distinguish mechanistic speculation from clinical evidence."
    ),
    "finance": (
        "Finance niche rules: prioritize regulators, official company filings, central banks, and major financial data providers. "
        "Avoid personalized financial advice. Distinguish opinions/forecasts from factual statements."
    ),
}

def get_api_key() -> str:
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = None
    if not key:
        key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY. Add it to Streamlit secrets (preferred) or environment variables.")
    return key

def init_client() -> genai.Client:
    return genai.Client(api_key=get_api_key())

def build_user_prompt(user_context: str, niche: str, compare_to_user: bool) -> str:
    niche_rules = NICHE_RULESETS.get(niche, NICHE_RULESETS["general"])
    comparison_instruction = (
        "The user asked for a non-diagnostic comparison to themselves. After claim validation, include a brief behavioral-pattern comparison only in general terms, with clear limits and no diagnosis."
        if compare_to_user else "No personal comparison requested."
    )
    return (
        "Task: Validate claims in the uploaded media files.\n"
        f"Niche: {niche}\n"
        f"Niche rules: {niche_rules}\n"
        f"User context (short explanation): {user_context or 'No extra context provided.'}\n"
        f"{comparison_instruction}\n"
        "Important: Use grounding with Google Search and cite evidence in the JSON sources arrays."
    )

def build_media_part(file_obj, mime_type: Optional[str]) -> Any:
    file_obj.seek(0)
    data = file_obj.read()
    if not data:
        raise ValueError(f"File '{file_obj.name}' is empty")
    return types.Part.from_bytes(data=data, mime_type=(mime_type or "application/octet-stream"))

def extract_text_from_response(resp: Any) -> str:
    if hasattr(resp, "text") and resp.text:
        return resp.text
    try:
        candidates = getattr(resp, "candidates", None) or []
        for c in candidates:
            content = getattr(c, "content", None)
            if not content:
                continue
            parts = getattr(content, "parts", None) or []
            texts = [getattr(p, "text", "") for p in parts if getattr(p, "text", None)]
            if texts:
                return "\n".join(texts)
    except Exception:
        pass
    return ""

def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()

def _sanitize_common_corruptions(s: str) -> str:
    # Remove accidental nested markdown fence starts that often appear inside long strings.
    s = s.replace("```json", "").replace("```", "")
    # Remove raw control chars except standard whitespace.
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", s)
    return s

def safe_json_parse(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not text:
        return None, "Empty model response"
    s = _sanitize_common_corruptions(_strip_fences(text))
    first = s.find("{")
    if first == -1:
        return None, f"JSON parse failed: no '{{' found. Raw response starts with: {text[:300]}"
    s = s[first:]

    decoder = json.JSONDecoder()
    try:
        obj, end = decoder.raw_decode(s)
        if isinstance(obj, dict):
            return obj, None
    except Exception:
        pass

    # Fallback: cut first balanced JSON object and parse
    depth = 0
    in_str = False
    esc = False
    end_idx = None
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            elif ch in "\n\r":
                # Replace raw newlines inside strings with escaped newline marker
                pass
            continue
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break

    if end_idx is None:
        return None, f"JSON parse failed: could not match braces. Raw response starts with: {text[:300]}"

    candidate = s[:end_idx]
    # Escape literal newlines that may have slipped into strings by a conservative pass
    out = []
    in_str = False
    esc = False
    for ch in candidate:
        if in_str:
            if esc:
                out.append(ch)
                esc = False
                continue
            if ch == "\\":
                out.append(ch)
                esc = True
                continue
            if ch == '"':
                out.append(ch)
                in_str = False
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            out.append(ch)
        else:
            out.append(ch)
            if ch == '"':
                in_str = True
    candidate = "".join(out)

    try:
        obj = json.loads(candidate)
        if not isinstance(obj, dict):
            return None, "Top-level JSON must be an object/dict"
        return obj, None
    except Exception as e2:
        return None, f"JSON parse failed: {e2}. Raw response starts with: {text[:300]}"

def validate_output_shape(obj: Dict[str, Any]) -> List[str]:
    issues = []
    for key in ["summary", "claims", "creator_style_signals", "user_relevance"]:
        if key not in obj:
            issues.append(f"Missing top-level key: {key}")
    if "claims" in obj and not isinstance(obj.get("claims"), list):
        issues.append("'claims' must be a list")
    return issues

def grounding_links_from_response(response: Any) -> List[Dict[str, str]]:
    links = []
    try:
        cands = getattr(response, "candidates", None) or []
        if not cands:
            return links
        gm = getattr(cands[0], "grounding_metadata", None)
        if gm is None:
            return links

        # try common shapes
        chunks = getattr(gm, "grounding_chunks", None) or getattr(gm, "groundingChunks", None) or []
        for ch in chunks:
            web = getattr(ch, "web", None) or {}
            uri = getattr(web, "uri", None) if hasattr(web, "uri") else (web.get("uri") if isinstance(web, dict) else None)
            title = getattr(web, "title", None) if hasattr(web, "title") else (web.get("title") if isinstance(web, dict) else None)
            if uri or title:
                links.append({"title": title or "", "url": uri or ""})

        # dedupe
        dedup = []
        seen = set()
        for l in links:
            key = (l.get("title",""), l.get("url",""))
            if key in seen:
                continue
            seen.add(key)
            dedup.append(l)
        return dedup[:30]
    except Exception:
        return links

def run_validation(client: genai.Client, uploaded_files: List[Any], user_context: str, niche: str, compare_to_user: bool) -> Tuple[Dict[str, Any], str]:
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    media_parts = [build_media_part(f, getattr(f, "type", None)) for f in uploaded_files]
    user_prompt = build_user_prompt(user_context, niche, compare_to_user)
    contents: List[Any] = [user_prompt, *media_parts]

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[grounding_tool],
        temperature=0.1,
        top_p=0.95,
        max_output_tokens=3072,  # shorter to reduce corruption/duplication
    )

    started = time.time()
    response = client.models.generate_content(model=DEFAULT_MODEL, contents=contents, config=config)
    latency = time.time() - started

    raw_text = extract_text_from_response(response)
    parsed, parse_err = safe_json_parse(raw_text)
    grounding_links = grounding_links_from_response(response)

    meta = {
        "model": DEFAULT_MODEL,
        "latency_seconds": round(latency, 2),
        "grounding_requested": True,
        "media_mode": "inline_parts",
        "raw_text_preview": (raw_text[:500] if raw_text else ""),
        "grounding_links_count": len(grounding_links),
    }

    if parse_err:
        return {"_meta": meta, "_error": parse_err, "_raw_model_text": raw_text, "_grounding_links": grounding_links}, raw_text

    parsed["_meta"] = meta
    if grounding_links:
        parsed["_grounding_links"] = grounding_links

    shape_issues = validate_output_shape(parsed)
    if shape_issues:
        parsed.setdefault("_warnings", []).extend(shape_issues)
    return parsed, raw_text

def compare_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    comparison = {"files_analyzed": len(results), "overall_verdicts": [], "common_red_flags": {}, "claim_count_total": 0, "notes": []}
    for idx, r in enumerate(results, start=1):
        summary = r.get("summary", {}) if isinstance(r, dict) else {}
        comparison["overall_verdicts"].append({"file_index": idx, "overall_verdict": summary.get("overall_verdict", "unknown"), "confidence": summary.get("confidence", "unknown")})
        claims = r.get("claims", []) if isinstance(r, dict) else []
        comparison["claim_count_total"] += len(claims) if isinstance(claims, list) else 0
        if isinstance(claims, list):
            for c in claims:
                for rf in (c.get("red_flags") or []):
                    comparison["common_red_flags"][rf] = comparison["common_red_flags"].get(rf, 0) + 1
    verdict_set = {v["overall_verdict"] for v in comparison["overall_verdicts"]}
    comparison["notes"].append("All analyzed files have the same overall verdict." if len(verdict_set) == 1 else "Files differ in overall verdict; review claim-level evidence.")
    if comparison["common_red_flags"]:
        top_flag = max(comparison["common_red_flags"], key=comparison["common_red_flags"].get)
        comparison["notes"].append(f"Most frequent red flag across files: {top_flag}")
    return comparison

def render_result_card(idx: int, result: Dict[str, Any]) -> None:
    st.subheader(f"תוצאה #{idx}")
    if "_error" in result:
        st.error(result["_error"])
        if result.get("_grounding_links"):
            with st.expander("Grounding links (debug)"):
                st.json(result["_grounding_links"])
        if result.get("_raw_model_text"):
            with st.expander("Raw model response"):
                st.code(result["_raw_model_text"])
        return

    meta = result.get("_meta", {})
    summary = result.get("summary", {})
    st.write({
        "overall_verdict": summary.get("overall_verdict"),
        "confidence": summary.get("confidence"),
        "why": summary.get("why"),
        "latency_seconds": meta.get("latency_seconds"),
        "grounding_requested": meta.get("grounding_requested"),
        "grounding_links_count": meta.get("grounding_links_count"),
    })

    claims = result.get("claims", [])
    if isinstance(claims, list) and claims:
        for i, c in enumerate(claims, start=1):
            with st.expander(f"Claim {i}: {str(c.get('claim_text', ''))[:120]}"):
                st.json(c)
    else:
        st.info("No claim list returned.")

    if result.get("_grounding_links"):
        with st.expander("Grounding links (from API metadata)"):
            st.json(result["_grounding_links"][:15])

    with st.expander("Creator style signals"):
        st.json(result.get("creator_style_signals", {}))
    with st.expander("User relevance (non-diagnostic)"):
        st.json(result.get("user_relevance", {}))
    if result.get("_warnings"):
        st.warning("Warnings: " + "; ".join(result["_warnings"]))
    with st.expander("Full JSON"):
        st.json(result)

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Experimental private-use tool for validating claims in short-form videos/audio with grounded evidence.")

    with st.sidebar:
        st.header("הגדרות")
        niche = st.selectbox("נישה", options=["general", "psychology", "nutrition", "finance"], index=1)
        compare_to_user = st.checkbox("השוואה כללית אליי (לא אבחון)", value=False)
        st.markdown("---")
        st.markdown('**מפתח API (חובה):** `st.secrets["GEMINI_API_KEY"]`')
        st.caption("הערה: עם Grounding tool אי אפשר להשתמש response_mime_type=application/json (מגבלת API), לכן יש parser קשיח.")

    st.info("מוצר ניסיוני: בודק טענות ולא שופט אנשים. חובה מקור/ציטוט, חובה 'לא ידוע' כשאין ראיות, ואיסור אבחון.")

    consent = st.checkbox("אני מאשר/ת שהקבצים שהעליתי הם לשימוש פרטי שלי ושיש לי הרשאה להעלותם לבדיקה", value=False)
    user_context = st.text_area("הסבר קצר מה המשתמש רוצה לבדוק (מומלץ)", placeholder="לדוגמה: 'בדקו אם המונח הפסיכולוגי שהיא משתמשת בו אמיתי ואם היא עושה הכללה מוגזמת'", height=90)

    files = st.file_uploader("העלה עד 3 קבצי וידאו/אודיו על אותה טענה", type=SUPPORTED_TYPES, accept_multiple_files=True)
    if files and len(files) > MAX_FILES:
        st.error(f"אפשר להעלות עד {MAX_FILES} קבצים בלבד.")
        return

    c1, c2 = st.columns([1, 1])
    with c1:
        run_btn = st.button("בדוק טענות (Grounding חובה)", type="primary", use_container_width=True)
    with c2:
        st.button("נקה תוצאות", use_container_width=True, on_click=lambda: (st.session_state.pop("last_results", None), st.session_state.pop("last_comparison", None)))

    if run_btn:
        if not consent:
            st.error("יש לאשר שימוש פרטי והרשאת העלאה לפני הרצה.")
            st.stop()
        if not files:
            st.error("לא הועלו קבצים.")
            st.stop()
        try:
            client = init_client()
        except Exception as e:
            st.error(str(e))
            st.code('GEMINI_API_KEY = "YOUR_KEY_HERE"', language="toml")
            st.stop()

        all_results: List[Dict[str, Any]] = []
        progress = st.progress(0)
        status = st.empty()
        for idx, f in enumerate(files, start=1):
            status.write(f"מעבד קובץ {idx}/{len(files)}: {f.name}")
            try:
                res, _ = run_validation(client, [f], user_context, niche, compare_to_user)
                res["_file_name"] = f.name
                all_results.append(res)
            except Exception as e:
                all_results.append({"_file_name": f.name, "_error": str(e)})
            progress.progress(idx / len(files))

        st.session_state["last_results"] = all_results
        st.session_state["last_comparison"] = compare_results(all_results)
        status.success("סיום ניתוח")

    if "last_results" in st.session_state:
        st.markdown("## השוואה בין הסרטונים")
        st.json(st.session_state.get("last_comparison", {}))
        st.markdown("## תוצאות מפורטות")
        for i, r in enumerate(st.session_state["last_results"], start=1):
            st.markdown(f"### קובץ: {r.get('_file_name', f'#{i}')}")
            render_result_card(i, r)

        export_payload = {"comparison": st.session_state.get("last_comparison", {}), "results": st.session_state.get("last_results", [])}
        st.download_button("הורד JSON של התוצאות", data=json.dumps(export_payload, ensure_ascii=False, indent=2), file_name="claim_validation_results.json", mime="application/json")

    with st.expander("מה בדיוק הערך של הכלי מול צ'אט רגיל"):
        st.markdown("""
- **Pipeline קשיח**: פורמט קבוע + הפרדת טענות + אימות + מקורות + ודאות.
- **Guardrails**: איסור אבחון, חובה מקורות, חובה "לא ידוע" כשאין ראיות.
- **התאמה לנישה**: כללי אימות שונים לפסיכולוגיה/תזונה/פיננסים.
- **השוואה בין עד 3 קבצים**: עקביות, דגלים חוזרים, פסקי דין שונים.
- **פלט מובנה**: JSON קבוע לשימוש בהמשך, לא תשובת צ'אט כללית.
        """)

if __name__ == "__main__":
    main()
