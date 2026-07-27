"""Environment configuration and the shared Anthropic client.

Now provided by the shared `lehcore` library rather than implemented
here: this and readformeleh's `pipeline/client.py`/`pipeline/config.py`
had independently converged on the identical `get_client`/`require_env`
implementation, so it was extracted into one tested, shared copy (see
lehcore's README). Re-exported under the original names so
`rag/gate.py`/`rag/chain.py` keep working unchanged.
"""

from lehcore.client import DEFAULT_MODEL as MODEL
from lehcore.client import get_client, require_env

__all__ = ["MODEL", "get_client", "require_env"]
