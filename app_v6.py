import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from google import genai
from google.genai import types

APP_TITLE = "Claim Validator (Experimental v6)"
MAX_FILES = 3
SUPPORTED_TYPES = ["mp4", "mov", "m4a", "mp3", "wav", "webm", "mpeg", "mpga"]
DEFAULT_MODEL = "gemini-2.5-flash"

# --------------------------
# PROMPTS (Two-stage pipeline)
# --------------------------

EXTRACT_SYSTEM_PROMPT = """
You extract claims from short-form social content (video/audio).

Rules:
- Output JSON only. No markdown fences. No prose before/after.
- No diagnosis. Do not classify people clinically.
- Be concise.
- Return at most 4 claims.
- Claims should be atomic and checkable.
- Include transcript_summary and creator framing notes.
- If a statement is opinion-only, mark as opinion.
- Do not include sources at this stage.

Schema:
{
  "transcript_summary": "short summary of what was said",
  "claims": [
    {
      "claim_text": "...",
      "claim_type_guess": "factual|professional|interpretation|opinion|rhetorical",
      "why_checkable": "short reason"
    }
  ],
  "creator_style_signals": {
    "uses_absolute_language": true,
    "uses_emotional_certainty": false,
    "uses_professional_sounding_terms": true,
    "notes": "short"
  }
}
""".strip()

VALIDATE_CLAIM_SYSTEM_PROMPT = """
You validate ONE claim from short-form content with Google Search grounding.

NON-NEGOTIABLE:
1) Use Google Search grounding in this request.
2) No diagnosis. No medical/psychological diagnosis of a person.
3) Distinguish claim vs interpretation vs opinion when relevant.
4) If evidence is insufficient, return NOT_ENOUGH_EVIDENCE.
5) Do not overstate certainty.
6) Output JSON only. No markdown fences. No prose before/after.
7) Keep output compact.
8) Do NOT include full URLs in JSON (the app reads grounding metadata separately).

Schema:
{
  "claim_text": "...",
  "claim_type": "factual|professional|interpretation|opinion|rhetorical",
  "status": "accurate|misleading|partially_accurate|not_enough_evidence|opinion_not_verifiable",
  "explanation": "short concrete explanation",
  "confidence": "high|medium|low",
  "red_flags": ["barnum_effect|overgeneralization|false_authority|cherry_picking|none"],
  "sources": [
    {"title":"...", "source_hint":"publisher/domain", "evidence_note":"..."}
  ]
}
Limits:
- max 2 sources
- max 140 words in explanation
""".strip()

SUMMARY_SYSTEM_PROMPT = """
You are composing a final verdict summary from validated claims.

Rules:
- JSON only. No markdown. No prose before/after.
- No diagnosis.
- Keep it short and product-like.
- Use these verdict labels only:
  mostly_accurate | mixed | mostly_misleading | not_enough_evidence

Schema:
{
  "summary": {
    "overall_verdict": "mostly_accurate|mixed|mostly_misleading|not_enough_evidence",
    "why": "short explanation",
    "confidence": "high|medium|low"
  },
  "user_relevance": {
    "requested_comparison_done": true,
    "note": "non-diagnostic only"
  }
}
""".strip()

NICHE_RULESETS = {
    "general": "General claim validation. Prefer primary/high-trust sources.",
    "psychology": "Psychology: distinguish pop-psychology vs clinical/professional terminology. No diagnosis. Flag Barnum/Forer and overgeneralization.",
    "nutrition": "Nutrition: prefer WHO/NIH/guidelines/meta-analyses. Distinguish evidence levels.",
    "finance": "Finance: prefer regulators/filings/official data. No personalized advice.",
}


# --------------------------
# Utilities
# --------------------------

def get_api_key() -> str:
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = None
    if not key:
        key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY. Add it to Streamlit secrets or environment variables.")
    return key


def init_client() -> genai.Client:
    return genai.Client(api_key=get_api_key())


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
        for c in getattr(resp, "candidates", []) or []:
            content = getattr(c, "content", None)
            if not content:
                continue
            texts = [getattr(p, "text", "") for p in (getattr(content, "parts", None) or []) if getattr(p, "text", None)]
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


def _sanitize(s: str) -> str:
    s = _strip_fences(s)
    s = s.replace("```json", "").replace("```", "")
    # raw control chars
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", s)
    return s


def _find_balanced_json_prefix(s: str) -> Optional[str]:
    first = s.find("{")
    if first == -1:
        return None
    s = s[first:]
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[: i + 1]
    return None


def _escape_newlines_inside_strings(candidate: str) -> str:
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
    return "".join(out)


def safe_json_parse(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not text:
        return None, "Empty model response"
    s = _sanitize(text)

    # 1) direct raw_decode
    try:
        first = s.find("{")
        if first != -1:
            obj, _ = json.JSONDecoder().raw_decode(s[first:])
            if isinstance(obj, dict):
                return obj, None
    except Exception:
        pass

    # 2) handle repeated json starts by truncating before second start marker
    starts = [m.start() for m in re.finditer(r'(?=\{\s*"[\w_]+")|(?=```json\s*\{)', text)]
    if len(starts) >= 2:
        # Try each region until parse succeeds
        for i in range(len(starts) - 1):
            region = _sanitize(text[starts[i]:starts[i+1]])
            prefix = _find_balanced_json_prefix(region)
            if prefix:
                try:
                    obj = json.loads(_escape_newlines_inside_strings(prefix))
                    if isinstance(obj, dict):
                        return obj, None
                except Exception:
                    continue

    # 3) generic balanced prefix
    prefix = _find_balanced_json_prefix(s)
    if prefix:
        try:
            obj = json.loads(_escape_newlines_inside_strings(prefix))
            if isinstance(obj, dict):
                return obj, None
        except Exception as e:
            return None, f"JSON parse failed: {e}. Raw response starts with: {text[:300]}"

    return None, f"JSON parse failed: could not match braces. Raw response starts with: {text[:300]}"


def grounding_links_from_response(response: Any) -> List[Dict[str, str]]:
    links = []
    try:
        cands = getattr(response, "candidates", None) or []
        if not cands:
            return links
        gm = getattr(cands[0], "grounding_metadata", None)
        if gm is None:
            return links
        chunks = getattr(gm, "grounding_chunks", None) or getattr(gm, "groundingChunks", None) or []
        for ch in chunks:
            web = getattr(ch, "web", None) or {}
            uri = getattr(web, "uri", None) if hasattr(web, "uri") else (web.get("uri") if isinstance(web, dict) else None)
            title = getattr(web, "title", None) if hasattr(web, "title") else (web.get("title") if isinstance(web, dict) else None)
            if uri or title:
                links.append({"title": title or "", "url": uri or ""})
    except Exception:
        pass
    out, seen = [], set()
    for l in links:
        k = (l.get("title",""), l.get("url",""))
        if k in seen:
            continue
        seen.add(k)
        out.append(l)
    return out[:20]


def call_model(
    client: genai.Client,
    system_prompt: str,
    contents: List[Any],
    use_grounding: bool,
    max_output_tokens: int,
    temperature: float = 0.1,
) -> Tuple[Any, str]:
    tools = [types.Tool(google_search=types.GoogleSearch())] if use_grounding else None
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=tools,
        temperature=temperature,
        top_p=0.95,
        max_output_tokens=max_output_tokens,
    )
    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=contents,
        config=config,
    )
    raw_text = extract_text_from_response(response)
    return response, raw_text


# --------------------------
# Two-stage pipeline
# --------------------------

def stage_a_extract(
    client: genai.Client,
    media_parts: List[Any],
    user_context: str,
    niche: str,
    compare_to_user: bool
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    niche_rules = NICHE_RULESETS.get(niche, NICHE_RULESETS["general"])
    cmp_text = (
        "User asked for non-diagnostic comparison to self. Mention only general behavioral fit/limits later."
        if compare_to_user else "No personal comparison requested."
    )
    user_prompt = (
        "Task: extract claims from uploaded media.\n"
        f"Niche rules: {niche_rules}\n"
        f"User context: {user_context or 'None'}\n"
        f"{cmp_text}\n"
        "Return extraction JSON only."
    )
    resp, raw = call_model(
        client=client,
        system_prompt=EXTRACT_SYSTEM_PROMPT,
        contents=[user_prompt, *media_parts],
        use_grounding=False,  # extraction should be lightweight and stable
        max_output_tokens=1200,
        temperature=0.0,
    )
    parsed, err = safe_json_parse(raw)
    debug = {
        "raw": raw,
        "error": err,
        "meta": {
            "grounding_used": False,
            "raw_preview": raw[:400] if raw else "",
        }
    }
    if err:
        return None, debug

    claims = parsed.get("claims", [])
    if not isinstance(claims, list):
        parsed["claims"] = []
    else:
        parsed["claims"] = claims[:4]
    return parsed, debug


def stage_b_validate_one_claim(
    client: genai.Client,
    claim_text: str,
    claim_type_guess: str,
    transcript_summary: str,
    niche: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    niche_rules = NICHE_RULESETS.get(niche, NICHE_RULESETS["general"])
    user_prompt = (
        f"Niche rules: {niche_rules}\n"
        f"Transcript summary (context only): {transcript_summary[:600]}\n"
        f"Claim to validate: {claim_text}\n"
        f"Claim type guess from extractor: {claim_type_guess}\n"
        "Validate this single claim and return compact JSON."
    )

    # primary attempt
    resp, raw = call_model(
        client=client,
        system_prompt=VALIDATE_CLAIM_SYSTEM_PROMPT,
        contents=[user_prompt],
        use_grounding=True,
        max_output_tokens=900,
        temperature=0.1,
    )
    parsed, err = safe_json_parse(raw)
    fallback_used = False

    # fallback: even smaller schema if needed
    if err:
        fallback_used = True
        tiny_prompt = (
            "Return JSON only, tiny schema:\n"
            '{"claim_text":"...","status":"accurate|misleading|partially_accurate|not_enough_evidence|opinion_not_verifiable","reason":"...","confidence":"high|medium|low","red_flags":["barnum_effect|overgeneralization|false_authority|cherry_picking|none"]}\n'
            f"Claim: {claim_text}\n"
            f"Context summary: {transcript_summary[:300]}\n"
            f"Niche rules: {niche_rules}\n"
            "Use Google Search grounding. No diagnosis. No URLs."
        )
        resp2, raw2 = call_model(
            client=client,
            system_prompt="You validate one claim with Google Search grounding. JSON only. No markdown.",
            contents=[tiny_prompt],
            use_grounding=True,
            max_output_tokens=500,
            temperature=0.1,
        )
        parsed2, err2 = safe_json_parse(raw2)
        if not err2:
            parsed = {
                "claim_text": parsed2.get("claim_text", claim_text),
                "claim_type": claim_type_guess if claim_type_guess in {"factual","professional","interpretation","opinion","rhetorical"} else "professional",
                "status": parsed2.get("status", "not_enough_evidence"),
                "explanation": parsed2.get("reason", ""),
                "confidence": parsed2.get("confidence", "low"),
                "red_flags": parsed2.get("red_flags", ["none"]),
                "sources": [],
            }
            raw = raw2
            resp = resp2
            err = None
        else:
            err = f"primary+fallback failed | last: {err2}"
            raw = f"[PRIMARY]\n{raw}\n\n[FALLBACK]\n{raw2}"

    links = grounding_links_from_response(resp)
    debug = {
        "raw": raw,
        "error": err,
        "grounding_links": links,
        "meta": {
            "grounding_used": True,
            "fallback_used": fallback_used,
            "raw_preview": raw[:400] if raw else "",
        }
    }

    if err:
        return {
            "claim_text": claim_text,
            "claim_type": claim_type_guess if claim_type_guess in {"factual","professional","interpretation","opinion","rhetorical"} else "professional",
            "status": "not_enough_evidence",
            "explanation": "Validation call output was malformed; fallback result unavailable.",
            "confidence": "low",
            "red_flags": ["none"],
            "sources": [],
            "_error": err,
        }, debug

    # normalize shape
    normalized = {
        "claim_text": parsed.get("claim_text", claim_text),
        "claim_type": parsed.get("claim_type", claim_type_guess or "professional"),
        "status": parsed.get("status", "not_enough_evidence"),
        "explanation": parsed.get("explanation", parsed.get("reason", "")),
        "confidence": parsed.get("confidence", "low"),
        "red_flags": parsed.get("red_flags", ["none"]) if isinstance(parsed.get("red_flags"), list) else ["none"],
        "sources": parsed.get("sources", []) if isinstance(parsed.get("sources"), list) else [],
    }
    # attach grounding metadata links separately (more reliable than model-embedded URLs)
    if links:
        normalized["_grounding_links"] = links[:8]
    return normalized, debug


def compose_summary(
    client: genai.Client,
    validated_claims: List[Dict[str, Any]],
    creator_style_signals: Dict[str, Any],
    compare_to_user: bool
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    compact_claims = []
    for c in validated_claims:
        compact_claims.append({
            "claim_text": c.get("claim_text"),
            "status": c.get("status"),
            "confidence": c.get("confidence"),
            "red_flags": c.get("red_flags"),
            "explanation": c.get("explanation"),
        })
    prompt = (
        f"validated_claims={json.dumps(compact_claims, ensure_ascii=False)}\n"
        f"creator_style_signals={json.dumps(creator_style_signals or {}, ensure_ascii=False)}\n"
        f"user_compare_requested={str(compare_to_user).lower()}\n"
        "Return final summary JSON only."
    )
    resp, raw = call_model(
        client=client,
        system_prompt=SUMMARY_SYSTEM_PROMPT,
        contents=[prompt],
        use_grounding=False,
        max_output_tokens=500,
        temperature=0.0,
    )
    parsed, err = safe_json_parse(raw)
    debug = {"raw": raw, "error": err}
    if err:
        # deterministic fallback summary
        statuses = [c.get("status") for c in validated_claims]
        if not statuses:
            verdict = "not_enough_evidence"
            why = "No valid claim results were produced."
            conf = "low"
        else:
            misleading_like = sum(s in {"misleading"} for s in statuses)
            partial = sum(s in {"partially_accurate"} for s in statuses)
            accurate = sum(s in {"accurate"} for s in statuses)
            nee = sum(s in {"not_enough_evidence"} for s in statuses)
            if nee == len(statuses):
                verdict = "not_enough_evidence"
            elif misleading_like and accurate:
                verdict = "mixed"
            elif misleading_like and not accurate and not partial:
                verdict = "mostly_misleading"
            elif accurate >= len(statuses) - 1:
                verdict = "mostly_accurate"
            else:
                verdict = "mixed"
            why = "Auto-generated fallback summary due to malformed summary output."
            conf = "low"
        parsed = {
            "summary": {"overall_verdict": verdict, "why": why, "confidence": conf},
            "user_relevance": {
                "requested_comparison_done": bool(compare_to_user),
                "note": "non-diagnostic only"
            }
        }
    return parsed, debug


def run_validation_pipeline(
    client: genai.Client,
    uploaded_files: List[Any],
    user_context: str,
    niche: str,
    compare_to_user: bool
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    started = time.time()
    media_parts = [build_media_part(f, getattr(f, "type", None)) for f in uploaded_files]
    debug_bundle: Dict[str, Any] = {"stage_a": {}, "stage_b": [], "stage_c": {}}

    # Stage A: extraction (no grounding)
    extracted, dbg_a = stage_a_extract(client, media_parts, user_context, niche, compare_to_user)
    debug_bundle["stage_a"] = dbg_a

    if not extracted:
        total = round(time.time() - started, 2)
        return {
            "_error": "Stage A extraction failed (malformed model output).",
            "_meta": {"total_latency_seconds": total, "pipeline": "two_stage_v6", "stage_a_ok": False},
            "_debug": debug_bundle,
        }, debug_bundle

    transcript_summary = extracted.get("transcript_summary", "")
    extracted_claims = extracted.get("claims", []) if isinstance(extracted.get("claims"), list) else []
    creator_style = extracted.get("creator_style_signals", {})

    # If extractor returns no claims, still return structured result
    if not extracted_claims:
        summary_obj, dbg_c = compose_summary(client, [], creator_style, compare_to_user)
        debug_bundle["stage_c"] = dbg_c
        total = round(time.time() - started, 2)
        result = {
            "summary": summary_obj.get("summary", {"overall_verdict":"not_enough_evidence","why":"No checkable claims extracted.","confidence":"low"}),
            "claims": [],
            "creator_style_signals": creator_style,
            "user_relevance": summary_obj.get("user_relevance", {"requested_comparison_done": bool(compare_to_user), "note":"non-diagnostic only"}),
            "_meta": {"total_latency_seconds": total, "pipeline": "two_stage_v6", "stage_a_ok": True, "stage_b_count": 0},
        }
        return result, debug_bundle

    # Stage B: validate each claim with grounding (one-by-one)
    validated_claims: List[Dict[str, Any]] = []
    for c in extracted_claims[:4]:
        claim_text = str(c.get("claim_text", "")).strip()
        claim_type_guess = str(c.get("claim_type_guess", "professional")).strip()
        if not claim_text:
            continue
        validated, dbg_b = stage_b_validate_one_claim(
            client=client,
            claim_text=claim_text,
            claim_type_guess=claim_type_guess,
            transcript_summary=transcript_summary,
            niche=niche,
        )
        validated_claims.append(validated)
        debug_bundle["stage_b"].append(dbg_b)

    # Stage C: summary composition (no grounding)
    summary_obj, dbg_c = compose_summary(client, validated_claims, creator_style, compare_to_user)
    debug_bundle["stage_c"] = dbg_c

    total = round(time.time() - started, 2)
    result = {
        "summary": summary_obj.get("summary", {"overall_verdict":"mixed","why":"Summary generation fallback used.","confidence":"low"}),
        "claims": validated_claims,
        "creator_style_signals": creator_style,
        "user_relevance": summary_obj.get("user_relevance", {"requested_comparison_done": bool(compare_to_user), "note":"non-diagnostic only"}),
        "transcript_summary": transcript_summary,
        "_meta": {
            "total_latency_seconds": total,
            "pipeline": "two_stage_v6",
            "stage_a_ok": True,
            "stage_b_count": len(validated_claims),
            "grounding_requested_stage_b": True,
        },
    }
    return result, debug_bundle


# --------------------------
# UI helpers
# --------------------------

def compare_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    comparison = {
        "files_analyzed": len(results),
        "overall_verdicts": [],
        "common_red_flags": {},
        "claim_count_total": 0,
        "notes": [],
    }
    for idx, r in enumerate(results, start=1):
        summary = r.get("summary", {}) if isinstance(r, dict) else {}
        comparison["overall_verdicts"].append(
            {
                "file_index": idx,
                "overall_verdict": summary.get("overall_verdict", "unknown"),
                "confidence": summary.get("confidence", "unknown"),
            }
        )
        claims = r.get("claims", []) if isinstance(r, dict) else []
        if isinstance(claims, list):
            comparison["claim_count_total"] += len(claims)
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
        if result.get("_meta"):
            st.json(result["_meta"])
        if result.get("_debug"):
            with st.expander("Debug (stages)"):
                st.json(result["_debug"])
        return

    meta = result.get("_meta", {})
    summary = result.get("summary", {})
    st.write({
        "overall_verdict": summary.get("overall_verdict"),
        "confidence": summary.get("confidence"),
        "why": summary.get("why"),
        "total_latency_seconds": meta.get("total_latency_seconds"),
        "pipeline": meta.get("pipeline"),
        "stage_b_count": meta.get("stage_b_count"),
    })

    if result.get("transcript_summary"):
        with st.expander("Transcript summary (extracted)"):
            st.write(result["transcript_summary"])

    claims = result.get("claims", [])
    if isinstance(claims, list) and claims:
        for i, c in enumerate(claims, start=1):
            with st.expander(f"Claim {i}: {str(c.get('claim_text', ''))[:120]}"):
                st.json(c)
    else:
        st.info("No claims returned.")

    with st.expander("Creator style signals"):
        st.json(result.get("creator_style_signals", {}))

    with st.expander("User relevance (non-diagnostic)"):
        st.json(result.get("user_relevance", {}))

    # show debug only when needed
    if result.get("_debug"):
        with st.expander("Pipeline debug"):
            st.json(result["_debug"])

    with st.expander("Full JSON"):
        st.json(result)


# --------------------------
# Streamlit app
# --------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Two-stage pipeline (v6): extraction first, then claim-by-claim grounded validation. More stable than one big JSON response.")

    with st.sidebar:
        st.header("הגדרות")
        niche = st.selectbox("נישה", ["general", "psychology", "nutrition", "finance"], index=1)
        compare_to_user = st.checkbox("השוואה כללית אליי (לא אבחון)", value=False)
        show_debug = st.checkbox("הצג Debug של השלבים", value=False)
        st.markdown("---")
        st.markdown('**מפתח API (חובה):** `st.secrets["GEMINI_API_KEY"]`')
        st.caption("Grounding מופעל בשלב אימות הטענות בלבד (שלב B), כדי לשפר יציבות.")

    st.info("בודק טענות (לא אנשים): pipeline קשיח + guardrails + אימות claim-by-claim עם Grounding.")

    consent = st.checkbox("אני מאשר/ת שהקבצים לשימוש פרטי שלי ויש לי הרשאה להעלותם לבדיקה", value=False)

    user_context = st.text_area(
        "הסבר קצר מה המשתמש רוצה לבדוק",
        placeholder="לדוגמה: בדקו אם המונח הפסיכולוגי אמיתי ואם יש הכללה מוגזמת",
        height=90,
    )

    files = st.file_uploader(
        "העלה עד 3 קבצי וידאו/אודיו על אותה טענה",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
    )

    if files and len(files) > MAX_FILES:
        st.error(f"אפשר להעלות עד {MAX_FILES} קבצים בלבד.")
        return

    c1, c2 = st.columns(2)
    with c1:
        run_btn = st.button("בדוק טענות (v6, דו-שלבי)", type="primary", use_container_width=True)
    with c2:
        st.button(
            "נקה תוצאות",
            use_container_width=True,
            on_click=lambda: (st.session_state.pop("last_results", None), st.session_state.pop("last_comparison", None))
        )

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
                result, debug = run_validation_pipeline(
                    client=client,
                    uploaded_files=[f],
                    user_context=user_context,
                    niche=niche,
                    compare_to_user=compare_to_user,
                )
                result["_file_name"] = f.name
                if show_debug:
                    result["_debug"] = debug
                all_results.append(result)
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

    with st.expander("למה v6 אמור להיות יציב יותר"):
        st.markdown("""
- במקום JSON גדול אחד (שנוטה להישבר), יש **2 שלבים**:
  1) חילוץ טענות קצר (ללא Grounding)
  2) אימות כל טענה בנפרד עם Grounding
- אם טענה אחת נשברת, שאר הטענות עדיין ממשיכות.
- ה-JSON הסופי מורכב בקוד (יותר דטרמיניסטי).
- יש fallback קטן לכל טענה אם הפלט הראשי נשבר.
        """)


if __name__ == "__main__":
    main()
