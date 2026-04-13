"""
summarizer.py
=============
Graduation Project Summary Generator — standalone module.

Generates a one-cohesive-paragraph summary for the top-5 recommended
graduation projects, tailored to the group's profile.

Setup:
    pip install openai

Usage (standalone test):
    export HF_TOKEN="your_token_here"
    python summarizer.py

Usage (from app.py):
    from summarizer import generate_summary
    summary = generate_summary(top5_projects, group_profile)
"""

import os
from openai import OpenAI

# ─────────────────────────────────────────────
#   MODEL CONFIGURATION
# ─────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_MODEL = os.environ.get(
    "HF_MODEL",
    "Qwen/Qwen2.5-1.5B-Instruct:featherless-ai"
)

_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
) if HF_TOKEN else None


# ─────────────────────────────────────────────
#   PROMPT TEMPLATE
# ─────────────────────────────────────────────

_PROMPT_TEMPLATE = """\
You are an academic assistant helping undergraduate students understand the collective meaning of a set of recommended graduation projects.

Write one cohesive paragraph (no headings, no lists) that synthesizes the TOP 5 projects as a group. Focus on the core ideas found in their abstracts: what these projects aim to achieve, the problems they address, and the innovative directions they represent. Highlight only the themes or intentions that appear across multiple projects, especially the recurring educational, technical, or user‑centered challenges they attempt to solve. Do NOT summarize each project individually; instead, extract the shared ideas that reveal the overall direction of the project set.

Use only the information provided. Do NOT add or assume technologies, methods, or goals that are not explicitly mentioned.

At the end of the paragraph, include a short closing statement explaining why these projects are suitable for the students reading this. Base this justification ONLY on the natural intersections between the project ideas and the group's stated interests, application domains, and RDIA focus. Address the reader directly using "you" and "your", and express the justification naturally within the flow of the text.

Maximum 180 words.

The set of recommended projects to the group you talk to is as follows:
{project_1_data}
{project_2_data}
{project_3_data}
{project_4_data}
{project_5_data}

The Profile of the Group you are talking to is as follows:
- Interests: {group_interests}
- Applications: {group_apps}
- RDIA: {group_rdia}

- Respond with one cohesive paragraph only.
- Do NOT speak about the groups like if they were third person they are the reader, so use "you" and "your" when addressing them.
"""


# ─────────────────────────────────────────────
#   HELPERS
# ─────────────────────────────────────────────

def _format_project(rank: int, project: dict) -> str:
    """Convert a project dict to a readable string for the prompt."""
    return (
        f"Rank: {rank}\n"
        f"Title: {project.get('title', '')}\n"
        f"Keywords: {', '.join(project.get('keywords', []))}\n"
        f"Application Domain(s): {', '.join(project.get('application_domains', project.get('application', [])))}\n"
        f"Interest Area(s): {', '.join(project.get('interest_domains', project.get('interest', [])))}\n"
        f"RDIA: {', '.join(project.get('rdia', []))}\n"
        f"Match Explanation: {project.get('explanation', '')}"
    )


def _build_prompt(top5: list[dict], group_profile: dict) -> str:
    """Fill the prompt template with project and group profile data."""
    return _PROMPT_TEMPLATE.format(
        project_1_data=_format_project(1, top5[0]),
        project_2_data=_format_project(2, top5[1]),
        project_3_data=_format_project(3, top5[2]),
        project_4_data=_format_project(4, top5[3]),
        project_5_data=_format_project(5, top5[4]),
        group_interests=", ".join(group_profile.get("selected_interests", [])),
        group_apps=", ".join(group_profile.get("selected_applications", [])),
        group_rdia=", ".join(group_profile.get("selected_rdia", [])),
    )


# ─────────────────────────────────────────────
#   PUBLIC API  ←  هذي الميثود اللي تستوردها في app.py
# ─────────────────────────────────────────────

def generate_summary(top5_projects: list[dict], group_profile: dict) -> str | None:
    """
    Generate a one-paragraph summary for the top-5 recommended projects.

    Args:
        top5_projects : list of the first 5 project dicts from recommended_projects
        group_profile : dict with keys selected_interests, selected_applications, selected_rdia

    Returns:
        Summary string, or None if HF_TOKEN is missing or the call fails.
    """
    if _client is None:
        print("[SUMMARIZER] HF_TOKEN not set — skipping summary generation.")
        return None

    if len(top5_projects) < 5:
        print("[SUMMARIZER] Fewer than 5 projects — skipping summary.")
        return None

    prompt = _build_prompt(top5_projects, group_profile)

    try:
        completion = _client.chat.completions.create(
            model=HF_MODEL,
            max_tokens=500,
            temperature=0.4,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"[SUMMARIZER] Model call failed: {e}")
        return None
