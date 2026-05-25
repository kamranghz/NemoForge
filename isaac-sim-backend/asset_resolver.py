"""
Maps NemoClaw's chosen asset_key to either:
  - A real local USD file path   (for environments)
  - A "procedural" marker dict   (for human, tree, robot, etc.)

Procedural assets are created as geometry directly in Isaac Sim
by extension.py — no USD file needed.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

ASSET_ROOT = "C:/Issac assests/isaac-sim-assets1/Assets/Isaac/4.5/Isaac"

# ── USD file map ──────────────────────────────────────────────────────────────
USD_MAP: dict[str, str] = {
    "warehouse_simple":    "Environments/Simple_Warehouse/warehouse.usd",
    "warehouse_full":      "Environments/Simple_Warehouse/full_warehouse.usd",
    "warehouse_shelves":   "Environments/Simple_Warehouse/warehouse_multiple_shelves.usd",
    "warehouse_forklifts": "Environments/Simple_Warehouse/warehouse_with_forklifts.usd",
    "warehouse_digital":   "Environments/Digital_Twin_Warehouse/small_warehouse_digital_twin.usd",
    "hospital":            "Environments/Hospital/hospital.usd",
    "office":              "Environments/Office/office.usd",
    "simple_room":         "Environments/Simple_Room/simple_room.usd",
    "grid_room":           "Environments/Grid/gridroom_curved.usd",
    "flat_terrain":        "Environments/Terrains/flat_plane.usd",
    "rough_terrain":       "Environments/Terrains/rough_plane.usd",
    "stairs":              "Environments/Terrains/stairs.usd",
    "jetracer_track":      "Environments/Jetracer/jetracer_track_solid.usd",
}

# ── Procedural shape definitions ──────────────────────────────────────────────
# These are created as raw USD geometry in Isaac Sim — no file needed.
PROCEDURAL_MAP: dict[str, dict] = {
    "procedural_human": {
        "type":   "human",
        "color":  [0.85, 0.70, 0.55],   # skin tone
        "height": 1.75,
        "radius": 0.25,
    },
    "procedural_tree": {
        "type":   "tree",
        "color":  [0.20, 0.55, 0.20],   # green
        "height": 3.0,
        "radius": 1.2,
    },
    "procedural_robot": {
        "type":   "robot",
        "color":  [0.60, 0.60, 0.65],   # metallic grey
        "height": 1.5,
        "radius": 0.4,
    },
    "procedural_box": {
        "type":   "box",
        "color":  [0.70, 0.55, 0.35],   # wood brown
        "size":   [0.6, 0.6, 0.6],
    },
    "procedural_sphere": {
        "type":   "sphere",
        "color":  [0.30, 0.50, 0.90],   # blue
        "radius": 0.5,
    },
}

_ANSI = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


# ── JSON extraction ───────────────────────────────────────────────────────────

def _collapse_json(block: str) -> str:
    """
    Remove bare newlines inside a JSON block so that values like:
        "procedural
        _human"
    become valid:
        "procedural_human"
    """
    return re.sub(r'\n\s*', '', block)


def _extract_json(text: str) -> dict:
    """Pull JSON from agent output — handles line-wrapped values, fences, prose."""
    text = _ANSI.sub("", text).strip()

    # Find all {...} blocks (including multi-line), collapse newlines, try to parse
    for m in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        candidate = _collapse_json(m.group())
        try:
            r = json.loads(candidate)
            if isinstance(r, dict) and "asset_key" in r:
                return r
        except json.JSONDecodeError:
            pass

    # Try each line individually (in case agent put JSON on one line)
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and "asset_key" in line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass

    # Strip markdown fences and try full text
    clean = re.sub(r"```(?:json)?", "", text).strip()
    try:
        r = json.loads(_collapse_json(clean))
        if isinstance(r, dict): return r
    except json.JSONDecodeError:
        pass

    logger.warning("Could not parse JSON from agent.\nRaw (last 300 chars):\n%s", text[-300:])
    return {}


def _float3(value, default) -> list[float]:
    try:
        lst = list(value)
        if len(lst) >= 3:
            return [float(v) for v in lst[:3]]
    except (TypeError, ValueError):
        pass
    return list(default)


# ── Main resolver ─────────────────────────────────────────────────────────────

def resolve(agent_output: str, user_input: str = "") -> dict:
    """
    Returns either:
      {"asset_path": "C:/...", "position": [...], "rotation": [...], "scale": [...]}
    or:
      {"procedural": {...}, "position": [...], "rotation": [...], "scale": [...]}
    """
    parsed   = _extract_json(agent_output)
    key      = str(parsed.get("asset_key", "warehouse_simple")).lower().strip()
    position = _float3(parsed.get("position"), [0.0, 0.0, 0.0])
    rotation = _float3(parsed.get("rotation"), [0.0, 0.0, 0.0])
    scale    = _float3(parsed.get("scale"),    [1.0, 1.0, 1.0])

    mode = parsed.get("mode", "add")
    logger.info("NemoClaw chose asset_key: '%s'  mode: '%s'", key, mode)

    # Clear scene command
    if key == "clear_scene":
        return {"command": "clear", "position": position, "rotation": rotation, "scale": scale}

    # Change environment command
    if mode == "change_env":
        relative = USD_MAP.get(key, USD_MAP["warehouse_simple"])
        return {
            "command":    "change_env",
            "asset_path": f"{ASSET_ROOT}/{relative}",
            "position":   position,
            "rotation":   rotation,
            "scale":      scale,
        }

    # Procedural asset
    if key in PROCEDURAL_MAP:
        return {
            "procedural": PROCEDURAL_MAP[key],
            "position":   position,
            "rotation":   rotation,
            "scale":      scale,
        }

    # USD file asset
    relative = USD_MAP.get(key)
    if relative is None:
        # Fuzzy match — find any key that starts with the requested key
        for k, v in USD_MAP.items():
            if k.startswith(key) or key.startswith(k.split("_")[0]):
                relative = v
                logger.warning("Fuzzy matched '%s' → '%s'", key, k)
                break

    if relative is None:
        # Last resort fallback
        relative = USD_MAP["warehouse_simple"]
        logger.error("Unknown key '%s' — fallback to warehouse_simple", key)

    return {
        "asset_path": f"{ASSET_ROOT}/{relative}",
        "position":   position,
        "rotation":   rotation,
        "scale":      scale,
    }
