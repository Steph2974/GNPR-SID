from __future__ import annotations

def pluscode6(latitude: float, longitude: float) -> str:
    """Return the first 6 chars of Open Location Code (Plus Code)."""
    try:
        from openlocationcode import openlocationcode as olc
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency `openlocationcode`. Install it with `pip install openlocationcode`."
        ) from e

    return olc.encode(latitude, longitude)[:6]

