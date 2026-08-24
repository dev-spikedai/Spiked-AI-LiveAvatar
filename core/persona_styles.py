"""
Persona and answer-style prompt fragments.

The frontend stores user selections as ids (e.g. "balanced", "concise").
These ids are mapped here to the prompt text that gets injected into the
answer-generation system prompts. Keep ids in sync with
SpikedAI-Frontend/src/pages/settings.tsx (customerPersonas, answerStyles).
"""

from typing import Iterable, Optional


PERSONA_PROMPTS: dict[str, str] = {
    "balanced": (
        "Speak to a general B2B audience. Use clear, professional language. "
        "Balance business value with practical explanation. Avoid heavy jargon "
        "while maintaining credibility."
    ),
    "technical": (
        "Speak to a technical decision maker (CTO, VP Engineering, Tech Lead, "
        "Solution Architect). Use precise technical language. Focus on "
        "architecture, APIs/SDKs, scalability, performance, security, "
        "integrations, and deployment. Assume strong technical knowledge. "
        "Avoid simplifications."
    ),
    "finance": (
        "Speak to a finance executive (CFO, Controller, Budget Owner). Focus "
        "on ROI, TCO, pricing clarity, cost optimization, and financial risk. "
        "Use numbers and comparisons where possible. Avoid technical depth "
        "unless it impacts cost or returns."
    ),
    "executive": (
        "Speak to a senior executive (CEO, Founder, Managing Director). Focus "
        "on strategic impact, growth, competitive advantage, scalability, and "
        "long-term outcomes. Use simple language. Avoid technical or "
        "operational detail."
    ),
}


STYLE_PROMPTS: dict[str, str] = {
    "concise": (
        "Answer in a single short paragraph or a few sentences. State the "
        "core answer only. No headings, no conclusions, no examples, no "
        "extra context. Max 150 words."
    ),
    "in_depth": (
        "Provide a detailed, structured explanation. Include context, "
        "explanation, benefits, challenges, and comparisons where relevant. "
        "Use headings if helpful. Optimize for clarity and completeness."
    ),
    "points_format": (
        "Respond only using bullet points or numbered lists. Each point "
        "should cover one idea. Keep points concise and scannable. No long "
        "paragraphs."
    ),
    "with_analogy": (
        "Explain using a simple real-world analogy. Clearly map the analogy "
        "to the actual concept. Keep it accurate and easy to understand. "
        "Avoid unnecessary detail."
    ),
    "technical_terms": (
        "Define technical terms inline using: 'Term: Definition'. Keep "
        "definitions short and plain-language. Do not over-explain or repeat "
        "definitions unnecessarily."
    ),
    "sales_points": (
        "Frame the response as clear value propositions. Emphasize benefits, "
        "outcomes, and customer pain points. Use confident, persuasive, "
        "benefit-driven language."
    ),
    "case_study": (
        "Summarize a case study using SPSR: Situation, Problem, Solution, "
        "Result. Keep each section short. Focus on measurable outcomes and "
        "impact."
    ),
    "competitive_comparison": (
        "Compare against alternatives using bullets or a table. Evaluate "
        "features, performance, pricing, integrations, and support. Clearly "
        "highlight differentiators."
    ),
    "pricing_summary": (
        "Summarize pricing models, tiers, customization options, and TCO. "
        "Emphasize transparency, scalability, and value for money."
    ),
}


def render_persona(persona_id: Optional[str]) -> str:
    if not persona_id:
        return ""
    return PERSONA_PROMPTS.get(persona_id, "")


def render_styles(style_ids: Optional[Iterable[str]]) -> str:
    if not style_ids:
        return ""
    lines = [f"- {STYLE_PROMPTS[s]}" for s in style_ids if s in STYLE_PROMPTS]
    return "\n".join(lines)
