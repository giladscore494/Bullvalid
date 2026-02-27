import json
import os
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
Your job is to evaluate claims from uploaded media (video/audio) and brief user context.

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
6) Quote/trace sources for evidence-backed conclusions. Cite sources used for each validated claim.
7) Separate what is true from what is misleading in the same statement (partial truth handling).
8) Be concise, structured, and product-like. No motivational language.
9) Do NOT shame creators. Validate claims, not people.
10) When the topic involves mental health / psychology:
    - Avoid diagnostic language toward the viewer or creator.
    - Clarify when terms are pop-psychology vs recognized clinical/professional terminology.
    - Mention when a concept exists but is used inaccurately.

OUTPUT FORMAT:
Return STRICT JSON only (no markdown) matching this schema:
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
        {"title": "...", "url": "...", "evidence_note": "..."}
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
    """Read Gemini API key from Streamlit secrets first, then optional env fallback."""
    # Primary path (what the user asked for): Streamlit secrets
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = None

    # Optional fallback for local dev convenience
    if not key:
        key = os.getenv("GEMINI_API_KEY")

    if not key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY. Add it to Streamlit secrets (preferred) or environment variables."
        )
    return key


def init_client() -> genai.Client:
    key = get_api_key()
    return genai.Client(api_key=key)


def build_user_prompt(user_context: str, niche: str, compare_to_user: bool) -> str:
    niche_rules = NICHE_RULESETS.get(niche, NICHE_RULESETS["general"])
    comparison_instruction = (
        "The user asked for a non-diagnostic comparison to themselves. After claim validation, include a brief behavioral-pattern comparison only in general terms, with clear limits and no diagnosis."
        if compare_to_user
        else "No personal comparison requested."
    )
    return (
        f"Task: Validate claims in the uploaded media files.\\n"
        f"Niche: {niche}\\n"
        f"Niche rules: {niche_rules}\\n"
        f"User context (short explanation): {user_context or 'No extra context provided.'}\\n"
        f"{comparison_instruction}\\n"
        f"Important: Use grounding with Google Search and cite evidence in the JSON sources arrays."
    )


def upload_file_to_gemini(client: genai.Client, file_obj, mime_type: Optional[str]) -> Any:
    """Uploads a local uploaded file to Gemini Files API and returns the file reference."""
    # streamlit UploadedFile exposes .read(); reset pointer before sending.
    file_obj.seek(0)
    data = file_obj.read()
    if not data:
        raise ValueError(f"File '{file_obj.name}' is empty")

    # The SDK accepts bytes via types.Part.from_bytes OR file upload API. We use file upload API for larger media.
    uploaded = client.files.upload(
        file=(file_obj.name, data, mime_type or "application/octet-stream")
    )
    return uploaded


def extract_text_from_response(resp: Any) -> str:
    if hasattr(resp, "text") and resp.text:
        return resp.text
    # Fallback if SDK returns parts only
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


def safe_json_parse(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not text:
        return None, "Empty model response"
    stripped = text.strip()
    # Handle accidental markdown fences
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        return json.loads(stripped), None
    except Exception as e:
        return None, f"JSON parse failed: {e}. Raw response starts with: {text[:300]}"


def validate_output_shape(obj: Dict[str, Any]) -> List[str]:
    issues = []
    for key in ["summary", "claims", "creator_style_signals", "user_relevance"]:
        if key not in obj:
            issues.append(f"Missing top-level key: {key}")
    if "claims" in obj and not isinstance(obj.get("claims"), list):
        issues.append("'claims' must be a list")
    return issues


def run_validation(client: genai.Client, uploaded_files: List[Any], user_context: str, niche: str, compare_to_user: bool) -> Tuple[Dict[str, Any], str]:
    grounding_tool = types.Tool(google_search=types.GoogleSearch())

    # Upload all files first
    gemini_files = []
    for f in uploaded_files:
        mime = getattr(f, "type", None)
        gemini_files.append(upload_file_to_gemini(client, f, mime))

    user_prompt = build_user_prompt(user_context, niche, compare_to_user)

    # Build multimodal contents list: prompt + file refs
    contents: List[Any] = [user_prompt]
    for gf in gemini_files:
        # SDK accepts uploaded file object directly in contents for multimodal generate_content
        contents.append(gf)

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[grounding_tool],
        temperature=0.1,
        top_p=0.95,
        max_output_tokens=4096,
    )

    started = time.time()
    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=contents,
        config=config,
    )
    latency = time.time() - started

    raw_text = extract_text_from_response(response)
    parsed, parse_err = safe_json_parse(raw_text)

    result: Dict[str, Any] = {
        "_meta": {
            "model": DEFAULT_MODEL,
            "latency_seconds": round(latency, 2),
            "grounding_requested": True,
            "raw_text_preview": (raw_text[:500] if raw_text else ""),
        }
    }

    # Try to pull grounding metadata / source rendering if present
    try:
        cands = getattr(response, "candidates", None) or []
        if cands:
            gm = getattr(cands[0], "grounding_metadata", None)
            if gm is not None:
                result["_meta"]["grounding_metadata_present"] = True
                result["_meta"]["grounding_metadata"] = str(gm)
            else:
                result["_meta"]["grounding_metadata_present"] = False
    except Exception:
        result["_meta"]["grounding_metadata_present"] = None

    if parse_err:
        result["_error"] = parse_err
        result["_raw_model_text"] = raw_text
        return result, raw_text

    shape_issues = validate_output_shape(parsed)
    if shape_issues:
        parsed.setdefault("_warnings", []).extend(shape_issues)

    # Merge metadata with parsed result
    parsed["_meta"] = result["_meta"]
    return parsed, raw_text


def compare_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Basic cross-file consistency summary without DB."""
    comparison = {
        "files_analyzed": len(results),
        "overall_verdicts": [],
        "common_red_flags": {},
        "claim_count_total": 0,
        "notes": []
    }
    for idx, r in enumerate(results, start=1):
        summary = r.get("summary", {}) if isinstance(r, dict) else {}
        comparison["overall_verdicts"].append({
            "file_index": idx,
            "overall_verdict": summary.get("overall_verdict", "unknown"),
            "confidence": summary.get("confidence", "unknown"),
        })
        claims = r.get("claims", []) if isinstance(r, dict) else []
        comparison["claim_count_total"] += len(claims) if isinstance(claims, list) else 0
        if isinstance(claims, list):
            for c in claims:
                for rf in (c.get("red_flags") or []):
                    comparison["common_red_flags"][rf] = comparison["common_red_flags"].get(rf, 0) + 1

    verdict_set = {v["overall_verdict"] for v in comparison["overall_verdicts"]}
    if len(verdict_set) == 1:
        comparison["notes"].append("All analyzed files have the same overall verdict.")
    else:
        comparison["notes"].append("Files differ in overall verdict; review claim-level evidence.")

    if comparison["common_red_flags"]:
        top_flag = max(comparison["common_red_flags"], key=comparison["common_red_flags"].get)
        comparison["notes"].append(f"Most frequent red flag across files: {top_flag}")
    return comparison


def render_result_card(idx: int, result: Dict[str, Any]) -> None:
    st.subheader(f"תוצאה #{idx}")

    if "_error" in result:
        st.error(result["_error"])
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
        "grounding_metadata_present": meta.get("grounding_metadata_present"),
    })

    claims = result.get("claims", [])
    if isinstance(claims, list) and claims:
        for i, c in enumerate(claims, start=1):
            with st.expander(f"Claim {i}: {c.get('claim_text', '')[:120]}"):
                st.json(c)
    else:
        st.info("No claim list returned.")

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
        niche = st.selectbox(
            "נישה",
            options=["general", "psychology", "nutrition", "finance"],
            index=1,
            help="התאמת כללי אימות לפי תחום (חדות בנישה במקום מודל כללי בלבד).",
        )
        compare_to_user = st.checkbox(
            "השוואה כללית אליי (לא אבחון)",
            value=False,
            help="אם מסומן, המודל יתייחס רק ברמת דפוסים כלליים וללא אבחון אישי/קליני.",
        )
        st.markdown("---")
        st.markdown("**מפתח API (חובה):** `st.secrets[\"GEMINI_API_KEY\"]`")

    st.info(
        "מוצר ניסיוני: בודק טענות ולא שופט אנשים. חובה מקור/ציטוט, חובה 'לא ידוע' כשאין ראיות, ואיסור אבחון."
    )

    consent = st.checkbox(
        "אני מאשר/ת שהקבצים שהעליתי הם לשימוש פרטי שלי ושיש לי הרשאה להעלותם לבדיקה",
        value=False,
    )

    user_context = st.text_area(
        "הסבר קצר מה המשתמש רוצה לבדוק (מומלץ)",
        placeholder="לדוגמה: 'בדקו אם המונח הפסיכולוגי שהיא משתמשת בו אמיתי ואם היא עושה הכללה מוגזמת'",
        height=90,
    )

    files = st.file_uploader(
        "העלה עד 3 קבצי וידאו/אודיו על אותה טענה",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        help="פורמטים נפוצים נתמכים. עד 3 קבצים להשוואה על אותה טענה/נושא.",
    )

    if files and len(files) > MAX_FILES:
        st.error(f"אפשר להעלות עד {MAX_FILES} קבצים בלבד.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        run_btn = st.button("בדוק טענות (Grounding חובה)", type="primary", use_container_width=True)
    with col2:
        st.button("נקה תוצאות", use_container_width=True, on_click=lambda: st.session_state.pop("last_results", None))

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
            st.code(
                '[general]\n# .streamlit/secrets.toml\nGEMINI_API_KEY = "YOUR_KEY_HERE"',
                language="toml",
            )
            st.stop()

        all_results: List[Dict[str, Any]] = []
        progress = st.progress(0)
        status = st.empty()

        for idx, f in enumerate(files, start=1):
            status.write(f"מעבד קובץ {idx}/{len(files)}: {f.name}")
            try:
                res, _ = run_validation(client, [f], user_context=user_context, niche=niche, compare_to_user=compare_to_user)
                res["_file_name"] = f.name
                all_results.append(res)
            except Exception as e:
                all_results.append({"_file_name": f.name, "_error": str(e)})
            progress.progress(idx / len(files))

        comparison = compare_results(all_results)
        st.session_state["last_results"] = all_results
        st.session_state["last_comparison"] = comparison
        status.success("סיום ניתוח")

    if "last_results" in st.session_state:
        st.markdown("## השוואה בין הסרטונים")
        st.json(st.session_state.get("last_comparison", {}))

        st.markdown("## תוצאות מפורטות")
        for i, r in enumerate(st.session_state["last_results"], start=1):
            st.markdown(f"### קובץ: {r.get('_file_name', f'#{i}')}")
            render_result_card(i, r)

        # Export buttons (critical extra I added)
        export_payload = {
            "comparison": st.session_state.get("last_comparison", {}),
            "results": st.session_state.get("last_results", []),
        }
        st.download_button(
            "הורד JSON של התוצאות",
            data=json.dumps(export_payload, ensure_ascii=False, indent=2),
            file_name="claim_validation_results.json",
            mime="application/json",
        )

    with st.expander("מה בדיוק הערך של הכלי מול צ'אט רגיל"):
        st.markdown(
            """
- **Pipeline קשיח**: תמלול/ניתוח/אימות/ציטוטים/ודאות בפורמט קבוע.
- **Guardrails**: איסור אבחון, חובה מקורות, חובה "לא ידוע" כשאין ראיות.
- **התאמה לנישה**: כללי אימות שונים לפסיכולוגיה/תזונה/פיננסים.
- **השוואה בין עד 3 קבצים**: עקביות, דגלים חוזרים, פסקי דין שונים.
- **פלט מובנה**: JSON קבוע לשימוש בהמשך, לא תשובת צ'אט כללית.
            """
        )


if __name__ == "__main__":
    main()
