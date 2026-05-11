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
    summary = generate_summary(top5_projects)
"""

from openai import OpenAI

# ─────────────────────────────────────────────
#   MODEL CONFIGURATION
# ─────────────────────────────────────────────

HF_TOKEN = "enter your token here"
HF_MODEL = "Qwen/Qwen2.5-1.5B-Instruct:featherless-ai"

_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

# ─────────────────────────────────────────────
#   PROMPT TEMPLATE
# ─────────────────────────────────────────────

_PROMPT_TEMPLATE = """ You are an academic assistant helping undergraduate students understand the collective meaning of a set of recommended graduation projects.

Write one cohesive paragraph (no headings, no lists) that synthesizes the TOP 5 projects. Focus on the core ideas found in their abstracts: what these projects aim to achieve, the problems they address, and the innovative directions they represent. Highlight only the themes or intentions that appear across multiple projects, especially the recurring educational, technical, or user-centered challenges they attempt to solve. 

Strict constraints:
- Do NOT summarize each project individually; instead, extract the shared ideas that reveal the overall direction of the project set.
- Use only the information provided. Do NOT add or assume technologies, methods, or goals that are not explicitly mentioned.
- Maximum 50 words only.

Project Set:
{project_1_data}
{project_2_data}
{project_3_data}
{project_4_data}
{project_5_data}
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
        f"Abstract: {project.get('abstract', '')}\n"
    )


def _build_prompt(top5: list[dict]) -> str:
    """Fill the prompt template with project data."""
    return _PROMPT_TEMPLATE.format(
        project_1_data=_format_project(1, top5[0]),
        project_2_data=_format_project(2, top5[1]),
        project_3_data=_format_project(3, top5[2]),
        project_4_data=_format_project(4, top5[3]),
        project_5_data=_format_project(5, top5[4]),
    )


# ─────────────────────────────────────────────
#   PUBLIC API
# ─────────────────────────────────────────────

def generate_summary(top5_projects: list[dict]) -> str | None:
    """
    Generate a one-paragraph summary for the top-5 recommended projects.

    Args:
        top5_projects : list of the first 5 project dicts from recommended_projects

    Returns:
        Summary string, or None if generation fails.
    """
    if len(top5_projects) < 5:
        print("[SUMMARIZER] Fewer than 5 projects — skipping summary.")
        return None

    prompt = _build_prompt(top5_projects)

    try:
        completion = _client.chat.completions.create(
            model=HF_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.4,
        )
        return completion.choices[0].message.content.strip()

    except Exception as e:
        print(f"[SUMMARIZER] Model call failed: {e}")
        return None
