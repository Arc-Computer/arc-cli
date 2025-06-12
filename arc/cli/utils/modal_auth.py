"""Modal authentication helper for Arc CLI."""

import os
from pathlib import Path

from arc.cli.utils.console import ArcConsole

console = ArcConsole()


def setup_modal_auth(
    token_id: str | None = None, token_secret: str | None = None
) -> bool:
    """
    Setup Modal authentication with various methods.

    Priority order:
    1. Modal workspace deployment (MODAL_IDENTITY_TOKEN or MODAL_TASK_ID)
    2. Provided tokens (for shared/demo accounts)
    3. Environment variables (MODAL_TOKEN_ID, MODAL_TOKEN_SECRET)
    4. Arc-provided demo tokens (ARC_MODAL_TOKEN_ID, ARC_MODAL_TOKEN_SECRET)
    5. User's own Modal config (~/.modal.toml)

    Returns:
        True if authentication is configured, False otherwise
    """
    # Method 1: Check if running inside Modal workspace/container
    if os.environ.get("MODAL_IDENTITY_TOKEN") or os.environ.get("MODAL_TASK_ID"):
        console.print("Running inside Modal workspace deployment", style="info")
        return True
    # Method 2: Use provided tokens
    if token_id and token_secret:
        os.environ["MODAL_TOKEN_ID"] = token_id
        os.environ["MODAL_TOKEN_SECRET"] = token_secret
        console.print("Using provided Modal credentials", style="info")
        return True

    # Method 3: Check if already set in environment
    if os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"):
        console.print("Using existing Modal environment credentials", style="info")
        return True

    # Method 4: Check for Arc-provided demo tokens
    arc_token_id = os.environ.get("ARC_MODAL_TOKEN_ID")
    arc_token_secret = os.environ.get("ARC_MODAL_TOKEN_SECRET")

    if arc_token_id and arc_token_secret:
        os.environ["MODAL_TOKEN_ID"] = arc_token_id
        os.environ["MODAL_TOKEN_SECRET"] = arc_token_secret
        console.print("Using Arc demo Modal credentials", style="info")
        console.print(
            "Note: These are shared demo credentials with usage limits", style="warning"
        )
        return True

    # Method 5: Check for user's Modal config
    modal_config_path = Path.home() / ".modal.toml"
    if modal_config_path.exists():
        console.print("Using user's Modal configuration", style="info")
        return True

    # No authentication found
    return False


def get_modal_auth_instructions() -> str:
    """Get instructions for Modal authentication."""
    return """
Modal Authentication Required
────────────────────────────

To run scenarios on Modal, you need to authenticate. Choose one of:

1. Use your own Modal account (recommended for production):
   modal token new

2. Use Arc demo credentials (for testing):
   export ARC_MODAL_TOKEN_ID="<provided-token-id>"
   export ARC_MODAL_TOKEN_SECRET="<provided-token-secret>"

3. Set Modal tokens directly:
   export MODAL_TOKEN_ID="<your-token-id>"
   export MODAL_TOKEN_SECRET="<your-token-secret>"

For more information: https://modal.com/docs/guide/secrets
"""


def check_modal_quota() -> tuple[bool, str | None]:
    """
    Check if using shared Modal account and warn about quotas.

    Returns:
        (is_shared_account, warning_message)
    """
    if os.environ.get("ARC_MODAL_TOKEN_ID"):
        return True, (
            "You're using Arc's shared Modal account. "
            "This has usage limits and is for demo purposes only. "
            "For production use, please authenticate with your own Modal account."
        )
    return False, None
