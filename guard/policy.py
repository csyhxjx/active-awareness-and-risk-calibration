"""Guard decision entry point. Phase 1 deliberately passes native actions through."""

import copy


def decide(obs, native_chunk, ctx: dict) -> dict:
    """Return the native action chunk without changing observations or actions."""
    del ctx
    copy.deepcopy(obs)
    return {"mode": "execute_native", "action_chunk": native_chunk, "audit": {}}
