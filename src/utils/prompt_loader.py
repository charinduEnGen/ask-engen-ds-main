"""Loads agent/task prompt definitions from src/agents/agents.yml and
src/tasks/tasks.yml and renders their `{{token}}` placeholders.

Each task in tasks.yml is broken into `description` (human-readable summary,
not sent to the model), `input` (the interpolated context block),
`instructions` (the numbered steps), `output` (the response contract), and
`agent` (which agent's system prompt it pairs with — documentation only,
call sites in src.orchestrator.steps pass the agent name explicitly). The
LLM-facing prompt is `input` + `instructions` + `output` concatenated, via
render_task_prompt.

These YAML files are plain prompt text — there is no crewAI or agent
framework involved. Every LLM call in the pipeline is a plain function call
via src.utils.llm_client.
"""

from pathlib import Path

import yaml

_SRC_DIR = Path(__file__).resolve().parent.parent
_AGENTS_PATH = _SRC_DIR / "services" / "analytics" / "agents" / "agents.yml"
_TASKS_PATH = _SRC_DIR / "services" / "analytics" / "tasks" / "tasks.yml"


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_agent(name: str) -> dict:
    agents = _load_yaml(_AGENTS_PATH)
    if name not in agents:
        raise KeyError(f"Agent '{name}' not found in {_AGENTS_PATH}")
    return agents[name]


def load_task(name: str) -> dict:
    tasks = _load_yaml(_TASKS_PATH)
    if name not in tasks:
        raise KeyError(f"Task '{name}' not found in {_TASKS_PATH}")
    return tasks[name]


def render_template(template: str, **values: str) -> str:
    # Plain {{token}} substitution rather than str.format — task descriptions
    # embed literal `{` / `}` (JSON examples, pandas dict literals) that would
    # collide with format-string syntax.
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", str(value))
    return template


def render_task_prompt(name: str, **values: str) -> str:
    """Assembles a task's full LLM-facing prompt from its `static_context`
    (if present), `input`, `instructions`, and `output` sections (in that
    order) and renders `{{token}}` placeholders across all of them.
    `description` and `agent` are metadata for humans reading tasks.yml —
    never sent to the model.

    Only tasks with no `static_context` section (currently
    answer_enrichment_task, followups_task) should call this directly for
    their one-shot prompt. data_analysis_task instead splits into
    render_task_static_context + render_task_volatile_prompt so its static
    block can be served from an explicit Gemini context cache — see
    src.orchestrator.steps.run_data_processing_agent — but this function
    still concatenates all four sections as a correct (just uncached)
    fallback for providers that don't support caching."""
    task = load_task(name)
    sections = [task.get(part, "") for part in ("static_context", "input", "instructions", "output")]
    template = "\n\n".join(section.strip() for section in sections if section and section.strip())
    return render_template(template, **values)


def render_task_static_context(name: str, **values: str) -> str:
    """Renders a task's `static_context` section together with its
    (also-static) `instructions` and `output` sections — the part of the
    prompt that's request-invariant and therefore cacheable. Returns "" if
    the task has no `static_context` section, so callers can treat that as
    "nothing to cache, fall back to render_task_prompt" without a special
    case."""
    task = load_task(name)
    static = task.get("static_context", "")
    if not static.strip():
        return ""
    sections = [static, task.get("instructions", ""), task.get("output", "")]
    template = "\n\n".join(section.strip() for section in sections if section and section.strip())
    return render_template(template, **values)


def render_task_volatile_prompt(name: str, **values: str) -> str:
    """Renders just a task's `input` section — the per-request part not
    covered by render_task_static_context. Pair with it whenever caching is
    actually in effect; use render_task_prompt's full concatenation instead
    when it isn't (see that function's docstring)."""
    task = load_task(name)
    template = task.get("input", "").strip()
    return render_template(template, **values)


def build_agent_system_prompt(agent_name: str) -> str:
    agent = load_agent(agent_name)
    return (
        f"Role: {agent['role'].strip()}\n\n"
        f"Goal: {agent['goal'].strip()}\n\n"
        f"Backstory: {agent['backstory'].strip()}"
    )
