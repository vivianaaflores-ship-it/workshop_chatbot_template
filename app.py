import ast
import copy
import json
import math
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests
import streamlit as st

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from docx import Document
except Exception:
    Document = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

APP_ROOT = Path(__file__).parent
CONFIG_DIR = APP_ROOT / "config"
DATA_DIR = APP_ROOT / "company_data"
LOGO_DIR = APP_ROOT / "company_logo"
SUPPORTED_LOGO_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]

MODEL_ALIASES = {
    "glm5-1": "glm-5-1",
    "glm-5.1": "glm-5-1",
    "glm-5-1": "glm-5-1",
    "hypernoba": "hypernova-60b",
    "hypernova": "hypernova-60b",
    "hypernova-60b": "hypernova-60b",
}


def load_json(path: Path, fallback: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.warning(f"Could not read {path.name}. Using defaults. Error: {exc}")
        return fallback


def get_secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)


def normalize_model(model: str) -> str:
    return MODEL_ALIASES.get(model.strip().lower(), model.strip())


def find_company_logo() -> Optional[Path]:
    """Return the first logo image saved in company_logo/."""
    if not LOGO_DIR.exists():
        return None
    for path in sorted(LOGO_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_LOGO_EXTENSIONS:
            return path
    return None


def read_text_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in [".txt", ".md", ".markdown", ".json", ".csv"]:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf" and PdfReader:
        reader = PdfReader(str(path))
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)
    if suffix == ".docx" and Document:
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    return ""


def read_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    data = uploaded_file.getvalue()

    if suffix in [".txt", ".md", ".markdown", ".json", ".csv"]:
        return data.decode("utf-8", errors="ignore")

    temp_path = APP_ROOT / f".tmp_{uuid.uuid4().hex}{suffix}"
    try:
        temp_path.write_bytes(data)
        return read_text_file(temp_path)
    finally:
        try:
            temp_path.unlink()
        except Exception:
            pass


def chunk_text(source: str, text: str, max_chars: int = 1200) -> List[Dict[str, str]]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    chunks = []
    for i in range(0, len(cleaned), max_chars):
        chunk = cleaned[i : i + max_chars]
        chunks.append({"source": source, "text": chunk})
    return chunks


def load_company_knowledge(uploaded_files=None) -> List[Dict[str, str]]:
    chunks: List[Dict[str, str]] = []

    if DATA_DIR.exists():
        for path in sorted(DATA_DIR.iterdir()):
            if path.name.startswith(".") or path.is_dir():
                continue
            text = read_text_file(path)
            chunks.extend(chunk_text(path.name, text))

    if uploaded_files:
        for uploaded in uploaded_files:
            text = read_uploaded_file(uploaded)
            chunks.extend(chunk_text(uploaded.name, text))

    return chunks


def tokenize(text: str) -> set:
    words = re.findall(r"[a-zA-ZÀ-ÿ0-9_]{3,}", text.lower())
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "are", "you", "your", "our",
        "para", "por", "con", "que", "los", "las", "una", "uno", "del", "como", "esta", "este",
    }
    return {w for w in words if w not in stop}


def search_chunks(query: str, chunks: List[Dict[str, str]], top_k: int = 4) -> List[Dict[str, str]]:
    q = tokenize(query)
    if not q or not chunks:
        return []

    scored: List[Tuple[int, Dict[str, str]]] = []
    for chunk in chunks:
        t = tokenize(chunk["text"])
        score = len(q.intersection(t))
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]


def build_system_prompt(company: Dict[str, Any], personality: Dict[str, Any], model_cfg: Dict[str, Any]) -> str:
    lines = [
        f"You are {company.get('bot_name', 'the assistant')}.",
        f"Company/project: {company.get('company_name', 'Student company')}.",
        f"Bot goal: {company.get('bot_goal', 'Help users with questions about the company.')}.",
        f"Target users: {company.get('target_users', 'Customers or employees')}.",
        f"Main product/service: {company.get('product_service', 'Not specified')}.",
        "",
        "Personality and answer style:",
        f"- Role: {personality.get('role', 'Helpful company assistant')}",
        f"- Tone: {personality.get('tone', 'friendly and clear')}",
        f"- Language: {personality.get('language', 'same language as the user')}",
        f"- Answer length: {personality.get('answer_length', 'short but complete')}",
    ]

    for item in personality.get("do", []):
        lines.append(f"- Do: {item}")
    for item in personality.get("dont", []):
        lines.append(f"- Do not: {item}")
    for item in personality.get("safety_rules", []):
        lines.append(f"- Safety rule: {item}")

    lines.extend([
        "",
        "Use the company knowledge snippets when they are relevant.",
        "If you do not know the answer from the information available, say so clearly and suggest the next best step.",
        "Do not invent company policies, prices, legal conditions, or technical procedures.",
    ])

    if model_cfg.get("workshop_mode", True):
        lines.append("This is a workshop prototype, not a production system.")

    return "\n".join(lines)


ALLOWED_MATH_NAMES = {
    name: getattr(math, name)
    for name in [
        "ceil", "floor", "sqrt", "sin", "cos", "tan", "log", "log10", "exp", "pow", "pi", "e"
    ]
}


def safe_calculator(expression: str) -> str:
    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub,
        ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd, ast.Load, ast.Call, ast.Name,
    )
    try:
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return "Calculator error: expression contains unsupported syntax."
            if isinstance(node, ast.Name) and node.id not in ALLOWED_MATH_NAMES:
                return f"Calculator error: unsupported name '{node.id}'."
        result = eval(compile(tree, "<calculator>", "eval"), {"__builtins__": {}}, ALLOWED_MATH_NAMES)
        return str(result)
    except Exception as exc:
        return f"Calculator error: {exc}"


def format_company_datetime(company: Dict[str, Any]) -> str:
    tz_name = company.get("timezone", "Europe/Madrid")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz_name = "Europe/Madrid"
        tz = ZoneInfo(tz_name)
    now = datetime.now(tz)
    return now.strftime(f"%Y-%m-%d %H:%M:%S %Z ({tz_name})")


def execute_tool(name: str, arguments: Dict[str, Any], chunks: List[Dict[str, str]], company: Dict[str, Any]) -> str:
    if name == "calculator":
        return safe_calculator(str(arguments.get("expression", "")))

    if name == "current_datetime":
        return format_company_datetime(company)

    if name == "search_company_data":
        query = str(arguments.get("query", ""))
        results = search_chunks(query, chunks, top_k=4)
        if not results:
            return "No relevant company data found."
        return "\n\n".join(f"Source: {r['source']}\n{r['text']}" for r in results)

    if name == "create_support_ticket":
        issue = str(arguments.get("issue_summary", "No summary provided"))
        email = str(arguments.get("user_email", "not provided"))
        ticket_id = "TICKET-" + uuid.uuid4().hex[:8].upper()
        return (
            f"Created demo support ticket {ticket_id}. "
            f"Issue: {issue}. User email: {email}. "
            f"Escalation contact: {company.get('contact_email', 'not configured')}."
        )

    return f"Unknown tool: {name}"


def build_tools(tools_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    available = []
    tools = tools_cfg.get("tools", {})

    if tools.get("calculator", {}).get("enabled", False):
        available.append({
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Calculate a simple math expression, such as prices, discounts, or measurements.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "A simple math expression, for example '120*0.21'"}
                    },
                    "required": ["expression"],
                },
            },
        })

    if tools.get("current_datetime", {}).get("enabled", False):
        available.append({
            "type": "function",
            "function": {
                "name": "current_datetime",
                "description": "Get the current date and time in the company timezone (Europe/Madrid, Spain).",
                "parameters": {"type": "object", "properties": {}},
            },
        })

    if tools.get("search_company_data", {}).get("enabled", False):
        available.append({
            "type": "function",
            "function": {
                "name": "search_company_data",
                "description": "Search the company knowledge base (files in company_data) for relevant information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for in the company documents"}
                    },
                    "required": ["query"],
                },
            },
        })

    if tools.get("create_support_ticket", {}).get("enabled", False):
        available.append({
            "type": "function",
            "function": {
                "name": "create_support_ticket",
                "description": "Create a demo support ticket when a user needs human help.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "issue_summary": {"type": "string", "description": "Short summary of the problem"},
                        "user_email": {"type": "string", "description": "User email if they provided one"},
                    },
                    "required": ["issue_summary"],
                },
            },
        })

    return available


def active_tool_names(tools_cfg: Dict[str, Any]) -> List[str]:
    return [name for name, cfg in tools_cfg.get("tools", {}).items() if cfg.get("enabled")]


def make_runtime_tools_config(tools_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Let students toggle tools from the UI without editing JSON permanently."""
    runtime_cfg = copy.deepcopy(tools_cfg)
    tool_items = runtime_cfg.get("tools", {})

    if "tool_enabled_overrides" not in st.session_state:
        st.session_state.tool_enabled_overrides = {
            name: bool(cfg.get("enabled", False)) for name, cfg in tool_items.items()
        }

    for name, cfg in tool_items.items():
        if name not in st.session_state.tool_enabled_overrides:
            st.session_state.tool_enabled_overrides[name] = bool(cfg.get("enabled", False))
        cfg["enabled"] = bool(st.session_state.tool_enabled_overrides[name])

    return runtime_cfg


def empty_usage_metrics(model: str) -> Dict[str, Any]:
    return {
        "response_time_s": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "usage_source": "unavailable",
        "model": model,
        "api_calls": 0,
    }


def parse_api_usage(usage: Dict[str, Any], model: str) -> Dict[str, Any]:
    if usage.get("total_tokens") is None and usage.get("prompt_tokens") is None:
        metrics = empty_usage_metrics(model)
        metrics["usage_source"] = "unavailable"
        return metrics

    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "usage_source": "api",
        "model": model,
    }


def compactif_chat(
    messages: List[Dict[str, Any]],
    model_cfg: Dict[str, Any],
    tools: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    api_key = get_secret("COMPACTIF_API_KEY")
    base_url = get_secret("COMPACTIF_BASE_URL", model_cfg.get("base_url", "https://api.compactif.ai/v1"))
    model = normalize_model(model_cfg.get("default_model", "glm-5-1"))
    started = time.perf_counter()

    if not api_key:
        content = (
            "Demo mode: add COMPACTIF_API_KEY in Streamlit secrets to connect the real model. "
            "Token usage and cost are only shown when the API returns real usage data."
        )
        latency = time.perf_counter() - started
        metrics = empty_usage_metrics(model)
        metrics["response_time_s"] = latency
        return {"demo_mode": True, "content": content}, metrics

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(model_cfg.get("temperature", 0.4)),
        "max_tokens": int(model_cfg.get("max_tokens", 800)),
    }

    if tools and model_cfg.get("use_tools", True):
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    latency = time.perf_counter() - started

    if response.status_code >= 400:
        raise RuntimeError(f"Compactif API error {response.status_code}: {response.text[:500]}")

    data = response.json()
    message = data["choices"][0]["message"]
    usage_metrics = parse_api_usage(data.get("usage") or {}, model)

    return message, {
        "response_time_s": latency,
        "prompt_tokens": usage_metrics["prompt_tokens"],
        "completion_tokens": usage_metrics["completion_tokens"],
        "total_tokens": usage_metrics["total_tokens"],
        "usage_source": usage_metrics["usage_source"],
        "model": model,
        "api_calls": 1,
    }


def merge_metrics(total: Dict[str, Any], part: Dict[str, Any]) -> Dict[str, Any]:
    total["response_time_s"] = float(total.get("response_time_s", 0)) + float(part.get("response_time_s", 0))
    total["prompt_tokens"] = int(total.get("prompt_tokens", 0)) + int(part.get("prompt_tokens", 0))
    total["completion_tokens"] = int(total.get("completion_tokens", 0)) + int(part.get("completion_tokens", 0))
    total["total_tokens"] = int(total.get("total_tokens", 0)) + int(part.get("total_tokens", 0))
    total["api_calls"] = int(total.get("api_calls", 0)) + int(part.get("api_calls", 0))
    if part.get("usage_source") != "api":
        total["usage_source"] = "unavailable"
    total["model"] = part.get("model", total.get("model"))
    return total


def chat_with_tools(
    messages: List[Dict[str, Any]],
    model_cfg: Dict[str, Any],
    tools: List[Dict[str, Any]],
    chunks: List[Dict[str, str]],
    company: Dict[str, Any],
    tool_names: List[str],
    max_rounds: int = 3,
) -> Tuple[str, Dict[str, Any]]:
    current_messages = list(messages)
    metrics = {
        "response_time_s": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "usage_source": "api",
        "api_calls": 0,
        "model": normalize_model(model_cfg.get("default_model", "glm-5-1")),
        "tools_sent": tool_names,
        "tools_called": [],
    }

    for _ in range(max_rounds):
        message, part_metrics = compactif_chat(current_messages, model_cfg, tools)
        metrics = merge_metrics(metrics, part_metrics)

        if isinstance(message, dict) and message.get("demo_mode"):
            return message["content"], metrics

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return message.get("content") or "I could not generate a response.", metrics

        current_messages.append(message)
        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception:
                args = {}

            metrics["tools_called"].append(name)
            result = execute_tool(name, args, chunks, company)
            current_messages.append({
                "role": "tool",
                "tool_call_id": call.get("id"),
                "name": name,
                "content": result,
            })

    return "I used the available tools, but I need a bit more information to finish the answer.", metrics


def estimate_request_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    model_cfg: Dict[str, Any],
) -> Optional[float]:
    rates = model_cfg.get("pricing", {}).get("models", {}).get(normalize_model(model))
    if not rates:
        return None
    input_rate = float(rates.get("input", 0))
    output_rate = float(rates.get("output", 0))
    return (prompt_tokens / 1_000_000) * input_rate + (completion_tokens / 1_000_000) * output_rate


def format_cost_usd(cost: Optional[float]) -> str:
    if cost is None:
        return "—"
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.4f}"


def attach_request_cost(metrics: Dict[str, Any], model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    metrics = dict(metrics)
    if metrics.get("usage_source") == "api":
        metrics["request_cost_usd"] = estimate_request_cost(
            str(metrics.get("model", "")),
            int(metrics.get("prompt_tokens", 0)),
            int(metrics.get("completion_tokens", 0)),
            model_cfg,
        )
    else:
        metrics["request_cost_usd"] = None
    return metrics


def metrics_text(metrics: Optional[Dict[str, Any]]) -> str:
    if not metrics:
        return ""
    tools_called = metrics.get("tools_called") or []
    called_text = ", ".join(tools_called) if tools_called else "none"
    if metrics.get("usage_source") != "api":
        return (
            f"{metrics.get('response_time_s', 0):.2f}s · "
            f"tokens n/a · "
            f"model {metrics.get('model', 'unknown')} · "
            f"tools: {called_text}"
        )
    cost_text = format_cost_usd(metrics.get("request_cost_usd"))
    return (
        f"{metrics.get('response_time_s', 0):.2f}s · "
        f"{metrics.get('total_tokens', 0)} tokens · "
        f"cost {cost_text} · "
        f"model {metrics.get('model', 'unknown')} · "
        f"tools: {called_text}"
    )


def reset_chat(welcome: str):
    st.session_state.messages = [{"role": "assistant", "content": welcome}]
    st.session_state.pop("pending_question", None)


def inject_custom_css():
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1rem;
                padding-bottom: 1.5rem;
                max-width: 1100px;
                padding-left: 2rem;
                padding-right: 2rem;
            }
            [data-testid="stSidebar"] {
                background: #FFFFFF;
                border-right: 1px solid #E2E8F0;
            }
            [data-testid="stSidebar"] .block-container {
                padding-top: 1.25rem;
            }
            .sidebar-section {
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #64748B;
                margin: 1.25rem 0 0.5rem 0;
            }
            .metrics-pill {
                display: inline-block;
                margin-top: 0.35rem;
                padding: 0.3rem 0.7rem;
                border-radius: 8px;
                background: #F1F5F9;
                color: #64748B;
                font-size: 0.75rem;
                font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            }
            .hint-label {
                font-size: 0.82rem;
                color: #64748B;
                margin: 0.25rem 0 0.5rem 0;
            }
            div[data-testid="stChatMessage"] {
                border-radius: 12px;
            }
            div[data-testid="stChatInput"] textarea {
                border-radius: 12px !important;
            }
            .stButton > button[kind="secondary"] {
                border-radius: 999px;
                border: 1px solid #E2E8F0;
                background: #FFFFFF;
                color: #334155;
                font-size: 0.84rem;
            }
            [data-testid="stSidebar"] .stButton > button {
                border-radius: 10px;
                width: 100%;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(company: Dict[str, Any], logo_path: Optional[Path], chunk_count: int, demo_mode: bool):
    bot_name = company.get("bot_name", "Workshop Chatbot")
    bot_goal = company.get("bot_goal", "A simple chatbot template students can personalize.")
    company_name = company.get("company_name", "Student company")
    status = "Demo mode" if demo_mode else "Live"

    logo_col, info_col = st.columns([1, 10], vertical_alignment="center")
    with logo_col:
        if logo_path:
            st.image(str(logo_path), width=56)
        else:
            st.markdown("##### 🤖")

    with info_col:
        st.markdown(f"**{bot_name}**")
        st.caption(f"{company_name} · {chunk_count} knowledge chunks · {status}")
        st.caption(bot_goal)

    st.divider()


MODEL_CHOICES = ["glm-5-1", "hypernova-60b"]


def render_sidebar(
    company: Dict[str, Any],
    tools_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    make_runtime_tools_config(tools_cfg)

    with st.sidebar:
        st.markdown("### 🧪 Experiment controls")
        st.caption("Change the model and tools, ask the same question, and watch tokens and time change.")

        st.markdown('<div class="sidebar-section">Model</div>', unsafe_allow_html=True)
        current_model = normalize_model(model_cfg.get("default_model", "glm-5-1"))
        selected_model = st.selectbox(
            "Model",
            MODEL_CHOICES,
            index=MODEL_CHOICES.index(current_model) if current_model in MODEL_CHOICES else 0,
            label_visibility="collapsed",
            help="Switch models and re-ask the same question to compare token usage and response time.",
        )
        model_cfg["default_model"] = selected_model

        st.markdown('<div class="sidebar-section">Tools</div>', unsafe_allow_html=True)
        st.caption("Each enabled tool is sent to the model on every request, so it costs tokens even when unused.")
        for name, cfg in tools_cfg.get("tools", {}).items():
            default_value = bool(st.session_state.tool_enabled_overrides.get(name, cfg.get("enabled", False)))
            st.session_state.tool_enabled_overrides[name] = st.checkbox(
                name.replace("_", " "),
                value=default_value,
                help=cfg.get("student_note", ""),
                key=f"toggle_{name}",
            )

        runtime_tools_cfg = make_runtime_tools_config(tools_cfg)
        enabled = active_tool_names(runtime_tools_cfg)

        st.metric("Tools on", len(enabled))

        st.divider()
        if st.button("↺ Reset chat", use_container_width=True):
            reset_chat(company.get("welcome_message", "Hello! How can I help?"))
            st.rerun()
        if st.button("🧹 Clear comparison log", use_container_width=True):
            st.session_state.runs = []
            st.rerun()

        st.caption("Add data to `/company_data`, a logo to `/company_logo`, and edit `/config` — all in the repo.")

    return runtime_tools_cfg


def render_suggested_questions(company: Dict[str, Any]):
    examples = company.get("examples_of_good_questions") or []
    if not examples or len(st.session_state.messages) > 1:
        return

    st.markdown('<div class="hint-label">Try asking:</div>', unsafe_allow_html=True)
    cols = st.columns(min(len(examples), 3))
    for i, question in enumerate(examples):
        with cols[i % len(cols)]:
            if st.button(question, key=f"suggest_{i}", type="secondary", use_container_width=True):
                st.session_state.pending_question = question
                st.rerun()


def render_message_metrics(metrics: Optional[Dict[str, Any]]):
    caption = metrics_text(metrics)
    if caption:
        st.markdown(f'<div class="metrics-pill">{caption}</div>', unsafe_allow_html=True)


def record_run(question: str, metrics: Dict[str, Any]):
    if metrics.get("usage_source") != "api":
        return

    if "runs" not in st.session_state:
        st.session_state.runs = []

    tools_sent = metrics.get("tools_sent") or []
    tools_called = metrics.get("tools_called") or []
    cost = metrics.get("request_cost_usd")
    st.session_state.runs.append({
        "#": len(st.session_state.runs) + 1,
        "Question": (question[:40] + "…") if len(question) > 40 else question,
        "Model": metrics.get("model", "unknown"),
        "Tools on": len(tools_sent),
        "Time (s)": round(float(metrics.get("response_time_s", 0)), 2),
        "Prompt tok": int(metrics.get("prompt_tokens", 0)),
        "Completion tok": int(metrics.get("completion_tokens", 0)),
        "Total tok": int(metrics.get("total_tokens", 0)),
        "Cost ($)": round(float(cost), 6) if cost is not None else None,
        "Tools called": ", ".join(tools_called) if tools_called else "—",
    })


def render_pricing_reference(model_cfg: Dict[str, Any]):
    pricing_models = model_cfg.get("pricing", {}).get("models", {})
    if not pricing_models:
        return

    st.caption("Workshop models (USD per 1M tokens):")
    for model_name in MODEL_CHOICES:
        rates = pricing_models.get(model_name)
        if rates:
            st.caption(
                f"`{model_name}`: input ${rates.get('input', 0)}/M · output ${rates.get('output', 0)}/M"
            )

    with st.expander("All CompactifAI model rates"):
        rows = [
            {
                "Model": name,
                "Input ($/1M)": rates.get("input", 0),
                "Output ($/1M)": rates.get("output", 0),
            }
            for name, rates in pricing_models.items()
        ]
        if pd is not None:
            st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
        else:
            st.table(rows)
        source = model_cfg.get("pricing", {}).get("source")
        if source:
            st.caption(f"Source: {source}. Edit `config/model.json` if prices change.")


def render_comparison_lab(demo_mode: bool, model_cfg: Dict[str, Any]):
    runs = st.session_state.get("runs") or []

    if not runs:
        st.info("Ask a few questions in the **Chat** tab first. Each answer is logged here so you can compare tokens, time, and cost.")
        render_pricing_reference(model_cfg)
        with st.expander("How to run a good experiment"):
            st.markdown(
                "- **Tool cost:** ask one question with all tools OFF, then ON. Compare *Prompt tok* and *Cost ($)*.\n"
                "- **Model cost:** ask the *same* question on `glm-5-1`, then `hypernova-60b`. Compare *Total tok*, *Time (s)*, and *Cost ($)*.\n"
                "- **Tool use:** ask a math question with the calculator ON vs OFF and watch *Tools called*.\n"
                "- Keep the question identical so only one variable changes at a time."
            )
        return

    st.caption(
        "Compare rows to see how tools and models change token usage, response time, and cost. "
        "Cost = (prompt tokens × input rate + completion tokens × output rate) from `config/model.json`."
    )

    if len(runs) >= 2:
        last, prev = runs[-1], runs[-2]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Total tokens",
            last["Total tok"],
            delta=last["Total tok"] - prev["Total tok"],
            delta_color="inverse",
        )
        c2.metric(
            "Response time (s)",
            f'{last["Time (s)"]:.2f}',
            delta=round(last["Time (s)"] - prev["Time (s)"], 2),
            delta_color="inverse",
        )
        last_cost = last.get("Cost ($)") or 0.0
        prev_cost = prev.get("Cost ($)") or 0.0
        c3.metric(
            "Cost (USD)",
            format_cost_usd(last_cost),
            delta=round(last_cost - prev_cost, 6) if last.get("Cost ($)") is not None else None,
            delta_color="inverse",
        )
        c4.metric("Tools on", last["Tools on"], delta=last["Tools on"] - prev["Tools on"])
        st.caption("Deltas compare the latest run with the previous one. Lower tokens, time, and cost are better.")

    session_total = sum((r.get("Cost ($)") or 0.0) for r in runs)
    st.metric("Session total cost", format_cost_usd(session_total))

    if pd is not None:
        df = pd.DataFrame(runs).set_index("#")
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "⬇️ Download results (CSV)",
            df.to_csv().encode("utf-8"),
            file_name="workshop_token_comparison.csv",
            mime="text/csv",
        )
    else:
        st.table(runs)

    if demo_mode:
        st.info(
            "Demo mode: add `COMPACTIF_API_KEY` in secrets to get real model replies, "
            "token usage, and cost in the Comparison lab.",
            icon="ℹ️",
        )

    with st.expander("How to run a good experiment"):
        st.markdown(
            "- **Tool cost:** ask one question with all tools OFF, then ON. Compare *Prompt tok* and *Cost ($)*.\n"
            "- **Model cost:** ask the *same* question on `glm-5-1`, then `hypernova-60b`. Compare *Total tok*, *Time (s)*, and *Cost ($)*.\n"
            "- **Tool use:** ask a math question with the calculator ON vs OFF and watch *Tools called*.\n"
            "- Keep the question identical so only one variable changes at a time."
        )
    render_pricing_reference(model_cfg)


def render_chat_tab(
    company: Dict[str, Any],
    personality: Dict[str, Any],
    model_cfg: Dict[str, Any],
    runtime_tools_cfg: Dict[str, Any],
    chunks: List[Dict[str, str]],
):
    render_suggested_questions(company)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            render_message_metrics(msg.get("metrics"))

    user_input = st.session_state.pop("pending_question", None) or st.chat_input("Message the assistant…")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    relevant = search_chunks(user_input, chunks, top_k=4)
    knowledge_block = ""
    if relevant:
        knowledge_block = "\n\nRelevant company knowledge snippets:\n" + "\n\n".join(
            f"Source: {r['source']}\n{r['text']}" for r in relevant
        )

    system_prompt = build_system_prompt(company, personality, model_cfg) + knowledge_block

    api_messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    api_messages.extend([
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages[-10:]
    ])

    tools = build_tools(runtime_tools_cfg)
    tool_names = active_tool_names(runtime_tools_cfg)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, metrics = chat_with_tools(
                    api_messages,
                    model_cfg,
                    tools,
                    chunks,
                    company,
                    tool_names,
                )
                metrics = attach_request_cost(metrics, model_cfg)
            except Exception as exc:
                answer = f"There was an error calling the model: {exc}"
                metrics = attach_request_cost({
                    "response_time_s": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "usage_source": "unavailable",
                    "api_calls": 0,
                    "model": normalize_model(model_cfg.get("default_model", "glm-5-1")),
                    "tools_sent": tool_names,
                    "tools_called": [],
                }, model_cfg)
            st.markdown(answer)
            render_message_metrics(metrics)

    st.session_state.messages.append({"role": "assistant", "content": answer, "metrics": metrics})
    record_run(user_input, metrics)


def main():
    st.set_page_config(page_title="Workshop Chatbot Template", page_icon="🤖", layout="wide")

    inject_custom_css()

    company = load_json(CONFIG_DIR / "company.json", {})
    personality = load_json(CONFIG_DIR / "personality.json", {})
    tools_cfg = load_json(CONFIG_DIR / "tools.json", {})
    model_cfg = load_json(CONFIG_DIR / "model.json", {})

    logo_path = find_company_logo()
    chunks = load_company_knowledge()
    demo_mode = not bool(get_secret("COMPACTIF_API_KEY"))

    runtime_tools_cfg = render_sidebar(company, tools_cfg, model_cfg)
    render_header(company, logo_path, len(chunks), demo_mode)

    if "messages" not in st.session_state:
        reset_chat(company.get("welcome_message", "Hello! How can I help?"))

    tab_chat, tab_lab = st.tabs(["Chat", "Comparison lab"])

    with tab_chat:
        render_chat_tab(company, personality, model_cfg, runtime_tools_cfg, chunks)

    with tab_lab:
        render_comparison_lab(demo_mode, model_cfg)


if __name__ == "__main__":
    main()
