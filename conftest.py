"""
Root-level conftest.py — DO NOT MOVE TO tests/ AND DO NOT DELETE.

Purpose: silence DeprecationWarning from websockets.legacy that fires when
alpaca-py imports its WebSocket stream clients during module load.

Why root-level (not tests/conftest.py):
pytest wraps each test session in warnings.catch_warnings(), which RESETS
all Python warning filters, then re-applies the -W flags from the CLI.
Any filterwarnings() call made inside a tests/ conftest — or in pyproject.toml
filterwarnings — is wiped out before test collection starts. A root-level
conftest.py is loaded earlier, but the same reset still occurs. The standard
filterwarnings mechanisms cannot win against a CLI -W flag.

The fix:
Pre-import the alpaca modules HERE, inside a local catch_warnings() block,
BEFORE pytest activates its session wrapper. Python caches every imported
module in sys.modules. Once the cache is populated, subsequent imports
(from test files) are no-ops — the module body never re-executes, so the
DeprecationWarning fires exactly once, during this pre-import, and is
silenced locally.

Deletion criteria:
If alpaca-py drops the websockets.legacy dependency in a future release,
verify that `pytest tests/ -W error::DeprecationWarning` passes without this
file, then delete it.
"""
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"websockets\..*")
    import alpaca.data.requests  # noqa: F401
    import alpaca.data.timeframe  # noqa: F401
    import alpaca.trading.enums  # noqa: F401
    import alpaca.trading.requests  # noqa: F401
