"""Environment configuration and the shared Anthropic client.

Mirrors read-leh's pipeline/config.py + pipeline/client.py pattern:
validate required env vars in one place, rather than reading os.environ
ad hoc elsewhere.
"""

import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

# Same model already funded via read-leh's Anthropic key - no new billing.
MODEL = "claude-haiku-4-5-20251001"


def require_env(name: str) -> str:
    """Returns a required environment variable.

    Args:
        name: The environment variable name.

    Returns:
        The variable's value.

    Raises:
        RuntimeError: If the variable is unset or empty.
    """
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set - check your .env file.")
    return value


def get_client() -> anthropic.Anthropic:
    """Returns an Anthropic client authenticated from ANTHROPIC_API_KEY.

    Returns:
        A configured `anthropic.Anthropic` client.

    Raises:
        RuntimeError: If ANTHROPIC_API_KEY is not set.
    """
    return anthropic.Anthropic(api_key=require_env("ANTHROPIC_API_KEY"))
