"""
Shared helper for calling the deep-reasoning model (Claude Fable 5).

Two layers of resilience:
  1. Server-side `fallbacks` (the `server-side-fallback-2026-06-01` beta) —
     covers ONLY safety-classifier declines. If Fable's content-policy
     check refuses a request, Anthropic reruns it on the fallback model
     inside the same call, transparently.
  2. Client-side retry (this module) — covers everything the server-side
     fallback does NOT: rate limits, overloads, 5xx errors, network
     failures, and an org not yet eligible for Fable (e.g. missing the
     30-day data-retention requirement). None of these fall back on their
     own — Anthropic returns them as-is. If the primary call raises any
     exception, this retries once, directly against the fallback model,
     before giving up.

Used by weekly_review, coin_researcher, thesis_generator, and
report_generator — anywhere the bot reasons deeply, as opposed to the
frequent Haiku analysis loop in claude_analyzer.
"""
from __future__ import annotations

import logging

import config

logger = logging.getLogger(__name__)


def call_deep_model(client, *, max_tokens: int, messages: list,
                    system: str | None = None, thinking: bool = False):
    """
    Call config.CLAUDE_DEEP_MODEL (Fable 5) with a server-side fallback for
    content refusals, and a client-side retry against
    config.CLAUDE_DEEP_FALLBACK (Opus 4.8) for any other failure.

    Returns the response object from whichever model actually served the
    request. Raises the retry's exception if both attempts fail.
    """
    kwargs = {"max_tokens": max_tokens, "messages": messages}
    if system is not None:
        kwargs["system"] = system
    if thinking:
        kwargs["thinking"] = {"type": "adaptive"}

    try:
        return client.beta.messages.create(
            model=config.CLAUDE_DEEP_MODEL,
            betas=["server-side-fallback-2026-06-01"],
            fallbacks=[{"model": config.CLAUDE_DEEP_FALLBACK}],
            **kwargs,
        )
    except Exception as exc:
        logger.warning(
            "%s unavailable (%s) — retrying directly on %s",
            config.CLAUDE_DEEP_MODEL, exc, config.CLAUDE_DEEP_FALLBACK,
        )
        return client.messages.create(model=config.CLAUDE_DEEP_FALLBACK, **kwargs)
