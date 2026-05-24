import json
import logging
import math
import re
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import gradio as gr
import torch

from modules import script_callbacks, scripts, shared


logger = logging.getLogger("anima_artist_scheduled_mixer")

EXTENSION_DIR = Path(__file__).resolve().parents[1]
TEMPLATE_FILE = EXTENSION_DIR / "artist_mixer_templates.json"
REFERENCE_URL = "https://github.com/An1X3R/Anima-Artist-Mixer"

LANGUAGE_OPTION = "anima_artist_scheduled_mixer_language"
LANGUAGE_CHOICES = ("zh", "en")

FUSION_INTERPOLATE = "interpolate"
FUSION_CONCAT_WITH_BASE = "concat_with_base"
COMBINE_OUTPUT_AVG = "output_avg"
COMBINE_CONCAT = "concat"

CURVE_SMOOTH = "Smooth"
CURVE_HOLD = "Hold"
CURVE_TRIANGLE = "Triangle"
CURVE_FRONT = "Front loaded"
CURVE_BACK = "Back loaded"
CURVE_CHOICES = [CURVE_SMOOTH, CURVE_HOLD, CURVE_TRIANGLE, CURVE_FRONT, CURVE_BACK]

PRESET_CUSTOM = "Custom"
PRESET_COMPOSITION = "Composition"
PRESET_CHARACTER = "Character"
PRESET_STYLE = "Style"
STAGE_PRESETS = [PRESET_CUSTOM, PRESET_COMPOSITION, PRESET_CHARACTER, PRESET_STYLE]

OPT_PERFORMANCE = "Performance"
OPT_BALANCE = "Balance"
OPT_QUALITY = "Quality"
OPT_PRESETS = [OPT_PERFORMANCE, OPT_BALANCE, OPT_QUALITY]

APPLY_BASE = "Base"
APPLY_HIRES = "Hires"
APPLY_BOTH = "Both"

TABLE_HEADERS = [
    "Enabled",
    "Artist",
    "Weight",
    "Blocks",
    "Start",
    "End",
    "Peak",
    "Curve",
    "Stage",
    "Shift",
    "Auto Shift",
]

TABLE_DATATYPES = [
    "bool",
    "str",
    "number",
    "str",
    "number",
    "number",
    "number",
    "str",
    "str",
    "number",
    "bool",
]


LANG = {
    "zh": {
        "title": "Anima 画师串调度混合",
        "accordion": "Anima 画师串调度混合",
        "settings_label": "Anima 画师串调度混合界面语言",
        "enable": "启用画师串混合",
        "base_panel": "底图画师",
        "hires_panel": "高分辨率修复画师",
        "hires_independent": "高分辨率修复使用独立画师串；关闭时继承底图设置",
        "row_count": "画师行数",
        "hr_row_count": "高分画师行数",
        "base_shift": "底图 Shift",
        "hires_shift": "高分 Shift",
        "artist_table": "画师设置",
        "global_strength": "全局画师强度",
        "optimization": "优化预设",
        "combine": "组合模式",
        "fusion": "融合模式",
        "apply_uncond": "同时作用于负条件",
        "cache": "启用文本编码缓存",
        "template": "模板",
        "template_name": "模板名称",
        "save_template": "保存当前设置为模板",
        "rename_to": "重命名为",
        "rename_template": "重命名模板",
        "delete_template": "删除模板",
        "apply_target": "模板应用目标",
        "apply_template": "应用模板",
        "normalize_rows": "应用阶段/Shift 预设",
        "help": "说明",
        "status": "状态",
    },
    "en": {
        "title": "Anima Artist Scheduled Mixer",
        "accordion": "Anima Artist Scheduled Mixer",
        "settings_label": "Anima Artist Scheduled Mixer UI language",
        "enable": "Enable artist mixing",
        "base_panel": "Base artists",
        "hires_panel": "Hires. fix artists",
        "hires_independent": "Use independent Hires. fix artist chain; disabled = inherit base settings",
        "row_count": "Artist rows",
        "hr_row_count": "Hires artist rows",
        "base_shift": "Base Shift",
        "hires_shift": "Hires Shift",
        "artist_table": "Artist settings",
        "global_strength": "Global artist strength",
        "optimization": "Optimization preset",
        "combine": "Combine mode",
        "fusion": "Fusion mode",
        "apply_uncond": "Apply to unconditional rows",
        "cache": "Enable text-encoding cache",
        "template": "Template",
        "template_name": "Template name",
        "save_template": "Save current settings as template",
        "rename_to": "Rename to",
        "rename_template": "Rename template",
        "delete_template": "Delete template",
        "apply_target": "Apply target",
        "apply_template": "Apply template",
        "normalize_rows": "Apply stage/Shift presets",
        "help": "Guide",
        "status": "Status",
    },
}


INTRO_EN = f"""
Independent artist encoding and scheduled cross-attention mixing for Anima.
Each artist row is encoded separately as `artist + base prompt`, then mixed inside Anima cross-attention with `output_avg + interpolate`, close to the approach used by [{REFERENCE_URL}]({REFERENCE_URL}).

Thanks to **An1X3R/Anima-Artist-Mixer** and **汐浮尘/utowo** for the original split-and-encode/cross-attention design.
"""

INTRO_ZH = f"""
面向 Anima 的独立画师编码与按阶段 cross-attention 混合。每个画师会单独按 `画师 + 主提示词` 编码，再在 Anima 的 cross-attention 内用 `output_avg + interpolate` 混合，效果路线接近 [{REFERENCE_URL}]({REFERENCE_URL})。

感谢 **An1X3R/Anima-Artist-Mixer** 与 **汐浮尘/utowo** 的原始独立编码和 cross-attention 混合设计。
"""

HELP_EN = """
**Columns**

`Artist`: one artist tag or weighted tag such as `(wlop:1.2)`. `Weight` controls this row's relative and absolute contribution. `Blocks` accepts `0-27`, `0,3,5-12`, or negative indices such as `-1`.

`Start/End/Peak` are denoise progress values from 0 to 1. `Curve` shapes the row strength inside the window. `Stage + Shift + Auto Shift` can rewrite timing like the Anima LoRA Stage Scheduler: Composition is early, Character is middle, Style is late; larger Shift moves windows slightly later.

**Strength**

Global artist strength blends the normal prompt cross-attention output with the mixed artist output. Row weights choose each artist's share; small row weights also reduce total influence when the active weights sum below 1.

**Optimization**

Performance narrows the default layers/time and caps active work more aggressively. Balance is the default. Quality keeps wider windows. The core mode remains `output_avg + interpolate`, matching the recommended mode in the reference project.
"""

HELP_ZH = """
**列说明**

`Artist` 填单个画师标签，也可填 `(wlop:1.2)` 这类权重写法。`Weight` 控制该画师的相对比例，也会在总权重低于 1 时降低实际介入。`Blocks` 支持 `0-27`、`0,3,5-12`，也支持 `-1` 这种倒数索引。

`Start/End/Peak` 是 0 到 1 的去噪进度。`Curve` 控制窗口内强度变化。`Stage + Shift + Auto Shift` 会像 Anima LoRA 阶段插件一样重写时间：构图靠前，人物居中，画风靠后；Shift 越大窗口会略向后移动。

**强度**

全局画师强度用于把原始提示词 cross-attention 输出与画师混合输出插值。单行权重决定画师占比；当活跃权重总和低于 1 时，也会降低整体介入。

**优化**

性能预设会缩短默认层数和时间窗口，平衡为默认，质量预设保留更宽窗口。核心仍使用参考项目推荐的 `output_avg + interpolate` 路线。
"""


def _language():
    value = getattr(shared.opts, LANGUAGE_OPTION, "zh")
    return value if value in LANGUAGE_CHOICES else "zh"


def _t(key):
    return LANG.get(_language(), LANG["zh"]).get(key, key)


def _intro_text(language):
    return INTRO_ZH if language == "中文" else INTRO_EN


def _register_ui_settings():
    if LANGUAGE_OPTION in shared.opts.data_labels:
        return
    shared.opts.add_option(
        LANGUAGE_OPTION,
        shared.OptionInfo(
            "zh",
            "Anima Artist Scheduled Mixer language / Anima 画师串调度混合语言",
            gr.Radio,
            {"choices": LANGUAGE_CHOICES},
            section=("anima-artist-scheduled-mixer", "Anima Artist Scheduled Mixer"),
            category_id="system",
        ).needs_reload_ui(),
    )


script_callbacks.on_ui_settings(_register_ui_settings)


def _clamp(value, low, high):
    return max(low, min(high, value))


def _to_float(value, fallback=0.0):
    try:
        if value is None or value == "":
            return fallback
        return float(value)
    except Exception:
        return fallback


def _to_bool(value, fallback=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return fallback
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "启用", "是"}:
        return True
    if text in {"0", "false", "no", "n", "off", "禁用", "否"}:
        return False
    return fallback


def _format_float(value):
    text = f"{_to_float(value):.6f}".rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


def _split_artist_chain(chain):
    source = str(chain or "").replace("，", ",").replace("\r", "\n").replace("\n", ",")
    return [part.strip() for part in source.split(",") if part.strip()]


def _parse_inline_weight(text):
    value = str(text or "").strip()
    match = re.fullmatch(r"\((.*):\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\)", value)
    if not match:
        return value, 1.0
    return match.group(1).strip(), _to_float(match.group(2), 1.0)


def _parse_blocks(text, num_blocks):
    source = str(text or "").replace("，", ",").replace(" ", "")
    if not source:
        return set(range(num_blocks))
    result = set()
    for token in source.split(","):
        if not token:
            continue
        if "-" in token[1:]:
            dash = token.index("-", 1)
            try:
                start = int(token[:dash])
                end = int(token[dash + 1 :])
            except Exception:
                continue
            if start < 0:
                start += num_blocks
            if end < 0:
                end += num_blocks
            if start > end:
                start, end = end, start
            for idx in range(max(0, start), min(num_blocks - 1, end) + 1):
                result.add(idx)
        else:
            try:
                idx = int(token)
            except Exception:
                continue
            if idx < 0:
                idx += num_blocks
            if 0 <= idx < num_blocks:
                result.add(idx)
    return result or set(range(num_blocks))


def _auto_stage_values(stage, shift_value):
    shift = _clamp(_to_float(shift_value, 3.0), 1.0, 24.0)
    move = _clamp((shift - 3.0) / 21.0, -0.10, 1.0) * 0.12
    if stage == PRESET_COMPOSITION:
        start = 0.0
        end = _clamp(0.32 + move, 0.22, 0.46)
        peak = _clamp(0.08 + move * 0.25, start, end)
    elif stage == PRESET_STYLE:
        start = _clamp(0.62 + move, 0.48, 0.82)
        end = 1.0
        peak = _clamp(0.86 + move * 0.35, start, end)
    else:
        start = _clamp(0.20 + move * 0.65, 0.12, 0.36)
        end = _clamp(0.74 + move * 0.45, 0.58, 0.88)
        peak = _clamp((start + end) * 0.5, start, end)
    return start, end, peak


def _curve_factor(progress, start, end, peak, curve):
    if progress is None:
        return 1.0
    start = _clamp(start, 0.0, 1.0)
    end = _clamp(end, 0.0, 1.0)
    if end < start:
        start, end = end, start
    if progress < start or progress > end:
        return 0.0
    if end - start <= 1e-6:
        return 1.0
    local = (progress - start) / max(end - start, 1e-6)
    curve = str(curve or CURVE_SMOOTH)
    if curve == CURVE_HOLD:
        return 1.0
    if curve == CURVE_FRONT:
        return 1.0 - 0.75 * local
    if curve == CURVE_BACK:
        return 0.25 + 0.75 * local
    peak = _clamp(peak, start, end)
    peak_local = (peak - start) / max(end - start, 1e-6)
    if curve == CURVE_TRIANGLE:
        if local <= peak_local:
            return local / max(peak_local, 1e-6)
        return (1.0 - local) / max(1.0 - peak_local, 1e-6)
    if local <= peak_local:
        raw = local / max(peak_local, 1e-6)
    else:
        raw = (1.0 - local) / max(1.0 - peak_local, 1e-6)
    raw = _clamp(raw, 0.0, 1.0)
    return raw * raw * (3.0 - 2.0 * raw)


def _default_rows(count=3, shift=3.0, optimization=OPT_BALANCE):
    presets = [PRESET_CHARACTER, PRESET_STYLE, PRESET_COMPOSITION]
    rows = []
    for i in range(max(0, int(count))):
        stage = presets[i % len(presets)]
        start, end, peak = _auto_stage_values(stage, shift)
        blocks = "4-23"
        if optimization == OPT_PERFORMANCE:
            blocks = "8-20"
        elif optimization == OPT_QUALITY:
            blocks = "0-27"
        rows.append([True, "", 1.0, blocks, start, end, peak, CURVE_SMOOTH, stage, shift, True])
    return rows


def _coerce_table(value):
    if value is None:
        return []
    if hasattr(value, "to_dict"):
        try:
            records = value.to_dict("records")
            return [[record.get(h) for h in TABLE_HEADERS] for record in records]
        except Exception:
            pass
    if isinstance(value, dict) and "data" in value:
        value = value.get("data")
    rows = []
    for row in value if isinstance(value, (list, tuple)) else []:
        if isinstance(row, dict):
            rows.append([row.get(h) for h in TABLE_HEADERS])
        elif isinstance(row, (list, tuple)):
            padded = list(row)[: len(TABLE_HEADERS)]
            padded += [None] * (len(TABLE_HEADERS) - len(padded))
            rows.append(padded)
    return rows


def _normalize_rows(value, row_count=None, shift=3.0, optimization=OPT_BALANCE):
    rows = _coerce_table(value)
    if row_count is None:
        row_count = len(rows) or 1
    count = max(0, min(32, int(_to_float(row_count, len(rows) or 1))))
    defaults = _default_rows(count, shift, optimization)
    normalized = []
    for index in range(count):
        raw = rows[index] if index < len(rows) else defaults[index]
        raw += [None] * (len(TABLE_HEADERS) - len(raw))
        enabled = _to_bool(raw[0], True)
        artist = str(raw[1] or "").strip()
        weight = _clamp(_to_float(raw[2], 1.0), -4.0, 4.0)
        blocks = str(raw[3] or defaults[index][3]).strip() or defaults[index][3]
        start = _clamp(_to_float(raw[4], defaults[index][4]), 0.0, 1.0)
        end = _clamp(_to_float(raw[5], defaults[index][5]), 0.0, 1.0)
        peak = _clamp(_to_float(raw[6], defaults[index][6]), 0.0, 1.0)
        curve = str(raw[7] or CURVE_SMOOTH).strip()
        if curve not in CURVE_CHOICES:
            curve = CURVE_SMOOTH
        stage = str(raw[8] or PRESET_CUSTOM).strip()
        if stage not in STAGE_PRESETS:
            stage = PRESET_CUSTOM
        row_shift = _clamp(_to_float(raw[9], shift), 1.0, 24.0)
        auto = _to_bool(raw[10], True)
        if auto and stage != PRESET_CUSTOM:
            start, end, peak = _auto_stage_values(stage, row_shift)
        normalized.append(
            [
                enabled,
                artist,
                weight,
                blocks,
                round(start, 4),
                round(end, 4),
                round(peak, 4),
                curve,
                stage,
                round(row_shift, 4),
                auto,
            ]
        )
    return normalized


def _resize_table(value, count, shift, optimization):
    rows = _normalize_rows(value, count, shift, optimization)
    return gr.update(value=rows, row_count=(max(1, len(rows)), "dynamic"))


def _apply_shift_to_rows(value, shift, optimization):
    rows = _coerce_table(value)
    out = []
    for row in rows:
        row += [None] * (len(TABLE_HEADERS) - len(row))
        row[9] = _clamp(_to_float(shift, 3.0), 1.0, 24.0)
        out.append(row)
    return gr.update(value=_normalize_rows(out, len(out) or 1, shift, optimization))


def _template_data():
    if not TEMPLATE_FILE.exists():
        return {}
    try:
        with TEMPLATE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        logger.exception("Failed to load Anima artist mixer templates")
        return {}


def _save_template_data(data):
    TEMPLATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with TEMPLATE_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _template_choices():
    return sorted(_template_data().keys())


def _template_dropdown_update(value=None):
    choices = _template_choices()
    if value not in choices:
        value = choices[0] if choices else None
    return gr.update(choices=choices, value=value)


def _save_template_ui(name, base_rows, hires_rows, hires_independent, base_shift, hires_shift, global_strength, optimization, combine_mode, fusion_mode, apply_uncond, enable_cache):
    name = str(name or "").strip()
    if not name:
        return _template_dropdown_update(), "Template name is empty."
    data = _template_data()
    data[name] = {
        "base_rows": _normalize_rows(base_rows, None, base_shift, optimization),
        "hires_rows": _normalize_rows(hires_rows, None, hires_shift, optimization),
        "hires_independent": bool(hires_independent),
        "base_shift": _to_float(base_shift, 3.0),
        "hires_shift": _to_float(hires_shift, 3.0),
        "global_strength": _to_float(global_strength, 0.7),
        "optimization": optimization if optimization in OPT_PRESETS else OPT_BALANCE,
        "combine_mode": combine_mode if combine_mode in (COMBINE_OUTPUT_AVG, COMBINE_CONCAT) else COMBINE_OUTPUT_AVG,
        "fusion_mode": fusion_mode if fusion_mode in (FUSION_INTERPOLATE, FUSION_CONCAT_WITH_BASE) else FUSION_INTERPOLATE,
        "apply_uncond": bool(apply_uncond),
        "enable_cache": bool(enable_cache),
    }
    _save_template_data(data)
    return _template_dropdown_update(name), f"Saved template: {name}"


def _rename_template_ui(old_name, new_name):
    old_name = str(old_name or "").strip()
    new_name = str(new_name or "").strip()
    data = _template_data()
    if not old_name or old_name not in data:
        return _template_dropdown_update(), "No template selected."
    if not new_name:
        return _template_dropdown_update(old_name), "New template name is empty."
    data[new_name] = data.pop(old_name)
    _save_template_data(data)
    return _template_dropdown_update(new_name), f"Renamed template to: {new_name}"


def _delete_template_ui(name):
    name = str(name or "").strip()
    data = _template_data()
    if name in data:
        data.pop(name)
        _save_template_data(data)
        return _template_dropdown_update(), f"Deleted template: {name}"
    return _template_dropdown_update(), "No template deleted."


def _apply_template_ui(name, target, base_rows, hires_rows):
    data = _template_data()
    tpl = data.get(str(name or "").strip())
    if not isinstance(tpl, dict):
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "No template selected.",
        )
    base_tpl = tpl.get("base_rows") or []
    hires_tpl = tpl.get("hires_rows") or base_tpl
    target = target if target in {APPLY_BASE, APPLY_HIRES, APPLY_BOTH} else APPLY_BASE
    out_base = gr.update()
    out_hires = gr.update()
    base_count = gr.update()
    hires_count = gr.update()
    if target in {APPLY_BASE, APPLY_BOTH}:
        out_base = gr.update(value=base_tpl, row_count=(max(1, len(base_tpl)), "dynamic"))
        base_count = gr.update(value=max(1, len(base_tpl)))
    if target in {APPLY_HIRES, APPLY_BOTH}:
        out_hires = gr.update(value=hires_tpl, row_count=(max(1, len(hires_tpl)), "dynamic"))
        hires_count = gr.update(value=max(1, len(hires_tpl)))
    return (
        out_base,
        out_hires,
        base_count,
        hires_count,
        gr.update(value=tpl.get("hires_independent", False)),
        gr.update(value=tpl.get("base_shift", 3.0)),
        gr.update(value=tpl.get("hires_shift", tpl.get("base_shift", 3.0))),
        gr.update(value=tpl.get("global_strength", 0.7)),
        gr.update(value=tpl.get("optimization", OPT_BALANCE)),
        gr.update(value=tpl.get("combine_mode", COMBINE_OUTPUT_AVG)),
        gr.update(value=tpl.get("fusion_mode", FUSION_INTERPOLATE)),
        gr.update(value=tpl.get("apply_uncond", False)),
        gr.update(value=tpl.get("enable_cache", True)),
        f"Applied template `{name}` to {target}.",
    )


def _extract_cond_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        for key in ("crossattn", "cross_attn", "c_crossattn"):
            item = value.get(key)
            if torch.is_tensor(item):
                return item
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        if len(value) >= 1 and torch.is_tensor(value[0]):
            return value[0]
        if len(value) >= 1 and isinstance(value[0], (list, tuple)):
            return _extract_cond_tensor(value[0])
    return None


def _ensure_3d(tensor):
    if tensor is None:
        return None
    if tensor.dim() == 2:
        return tensor.unsqueeze(0)
    if tensor.dim() == 4 and tensor.shape[1] == 1:
        return tensor.squeeze(1)
    if tensor.dim() == 3:
        return tensor
    return tensor.reshape(1, tensor.shape[-2], tensor.shape[-1])


_COND_CACHE = OrderedDict()
_COND_CACHE_LIMIT = 64


def _cache_key(p, texts):
    sd_model = getattr(p, "sd_model", None)
    model_key = (
        id(getattr(sd_model, "cond_stage_model", None)),
        id(getattr(sd_model, "forge_objects", None)),
        str(getattr(sd_model, "current_lora_hash", "")),
    )
    return model_key, tuple(texts)


def _encode_text_batch(p, texts, use_cache=True):
    key = _cache_key(p, texts)
    if use_cache and key in _COND_CACHE:
        _COND_CACHE.move_to_end(key)
        return _COND_CACHE[key]
    conds = p.sd_model.get_learned_conditioning(list(texts))
    tensors = []
    if torch.is_tensor(conds):
        conds = list(conds)
    for cond in conds:
        tensor = _ensure_3d(_extract_cond_tensor(cond))
        if tensor is None:
            raise RuntimeError("Failed to extract Anima conditioning tensor.")
        tensors.append(tensor.detach().to("cpu"))
    batch = torch.cat(tensors, dim=0)
    if use_cache:
        _COND_CACHE[key] = batch
        while len(_COND_CACHE) > _COND_CACHE_LIMIT:
            _COND_CACHE.popitem(last=False)
    return batch


@dataclass
class ArtistRuntime:
    name: str
    weight: float
    blocks: set[int]
    start: float
    end: float
    peak: float
    curve: str
    cond: torch.Tensor


@dataclass
class MixerState:
    run_id: str
    enabled: bool
    global_strength: float
    combine_mode: str
    fusion_mode: str
    apply_uncond: bool
    batched: bool
    artists: list[ArtistRuntime] = field(default_factory=list)
    progress_warning: bool = False
    batched_disabled: bool = False

    def target_blocks(self):
        blocks = set()
        for artist in self.artists:
            blocks.update(artist.blocks)
        return blocks


_PATCHED_MODULES = []
_ACTIVE_STATE = None


def _unpatch_cross_attn():
    global _PATCHED_MODULES
    for module, original in reversed(_PATCHED_MODULES):
        try:
            if getattr(module.forward, "_anima_artist_mixer_wrapper", False):
                module.forward = original
        except Exception:
            logger.exception("Failed to restore Anima artist mixer cross-attn wrapper")
    _PATCHED_MODULES = []


def _validate_anima_unet(unet):
    try:
        dm = unet.model.diffusion_model
    except Exception:
        return None, "Cannot find diffusion_model"
    blocks = getattr(dm, "blocks", None)
    if not blocks:
        return None, "diffusion_model.blocks is empty"
    first = blocks[0]
    if not hasattr(first, "cross_attn"):
        return None, "blocks[0] has no cross_attn"
    if not hasattr(first.cross_attn, "context_dim"):
        return None, "cross_attn has no context_dim"
    return dm, "ok"


def _install_cross_attn_patch(dm, state):
    _unpatch_cross_attn()
    target_blocks = state.target_blocks()
    for idx, block in enumerate(getattr(dm, "blocks", [])):
        if idx not in target_blocks or not hasattr(block, "cross_attn"):
            continue
        module = block.cross_attn
        original = module.forward

        def make_wrapper(original_forward, layer_idx):
            def wrapped(x, context=None, rope_emb=None, transformer_options={}):
                return _dispatch_cross_attn(
                    original_forward,
                    layer_idx,
                    state,
                    x,
                    context=context,
                    rope_emb=rope_emb,
                    transformer_options=transformer_options,
                )

            wrapped._anima_artist_mixer_wrapper = True
            return wrapped

        module.forward = make_wrapper(original, idx)
        _PATCHED_MODULES.append((module, original))


def _broadcast_batch(tensor, batch_size):
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch_size, -1, -1)
    if batch_size % tensor.shape[0] == 0:
        return tensor.repeat(batch_size // tensor.shape[0], 1, 1)
    return tensor[:1].expand(batch_size, -1, -1)


def _resolve_row_mask(cond_or_uncond, batch_size, apply_uncond):
    if not cond_or_uncond:
        return [True] * batch_size
    markers = list(cond_or_uncond)
    if len(markers) == batch_size:
        return [apply_uncond or marker == 0 for marker in markers]
    if batch_size % len(markers) == 0:
        each = batch_size // len(markers)
        mask = []
        for marker in markers:
            mask.extend([apply_uncond or marker == 0] * each)
        return mask
    return [True] * batch_size


def _current_progress(transformer_options, state):
    if not isinstance(transformer_options, dict):
        return None
    cur = transformer_options.get("sigmas")
    all_sigmas = transformer_options.get("sampling_sigmas")
    if cur is None or all_sigmas is None:
        return None
    try:
        cur_value = float(cur.flatten()[0].detach().float().cpu().item()) if torch.is_tensor(cur) else float(cur[0])
        sigmas = all_sigmas.detach().float().cpu() if torch.is_tensor(all_sigmas) else torch.tensor(list(all_sigmas), dtype=torch.float32)
        if sigmas.numel() <= 1:
            return None
        idx = int(torch.argmin(torch.abs(sigmas - cur_value)).item())
        return _clamp(idx / max(1, sigmas.numel() - 1), 0.0, 1.0)
    except Exception:
        if not state.progress_warning:
            logger.warning("Anima artist mixer could not resolve sampling progress; time windows will act as full strength.")
            state.progress_warning = True
        return None


def _active_artists(state, layer_idx, progress):
    active = []
    for artist in state.artists:
        if layer_idx not in artist.blocks:
            continue
        factor = _curve_factor(progress, artist.start, artist.end, artist.peak, artist.curve)
        if factor <= 1e-6:
            continue
        active.append((artist, artist.weight * factor))
    return active


def _to_context(tensor, context):
    tensor = _ensure_3d(tensor)
    return tensor.to(device=context.device, dtype=context.dtype, non_blocking=True)


def _artist_forward_batched(original_forward, x, context, rope_emb, transformer_options, artists, weights, fusion_mode):
    batch_size = context.shape[0]
    contexts = []
    for artist, _ in artists:
        artist_context = _broadcast_batch(_to_context(artist.cond, context), batch_size)
        if fusion_mode == FUSION_CONCAT_WITH_BASE:
            contexts.append(torch.cat([context, artist_context], dim=1))
        else:
            contexts.append(artist_context)
    lengths = {item.shape[1] for item in contexts}
    if len(lengths) > 1:
        raise RuntimeError(f"Cannot batch artist contexts with different token lengths: {lengths}")
    count = len(contexts)
    x_rep = x.repeat(count, *([1] * (x.dim() - 1)))
    context_rep = torch.cat(contexts, dim=0)
    rope_rep = rope_emb
    if torch.is_tensor(rope_emb) and rope_emb.dim() > 0 and rope_emb.shape[0] == batch_size:
        rope_rep = rope_emb.repeat(count, *([1] * (rope_emb.dim() - 1)))
    opts = dict(transformer_options) if isinstance(transformer_options, dict) else {}
    cou = opts.get("cond_or_uncond")
    if cou is not None:
        opts["cond_or_uncond"] = list(cou) * count
    out = original_forward(x_rep, context=context_rep, rope_emb=rope_rep, transformer_options=opts)
    out = out.view(count, batch_size, *out.shape[1:])
    w = torch.tensor(weights, device=out.device, dtype=out.dtype).view(count, *([1] * (out.dim() - 1)))
    return (out * w).sum(dim=0)


def _dispatch_cross_attn(original_forward, layer_idx, state, x, context=None, rope_emb=None, transformer_options={}):
    if not state.enabled or context is None or not state.artists:
        return original_forward(x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)
    progress = _current_progress(transformer_options, state)
    active = _active_artists(state, layer_idx, progress)
    if not active:
        return original_forward(x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)
    try:
        if state.combine_mode == COMBINE_CONCAT:
            return _dispatch_concat(original_forward, state, x, context, rope_emb, transformer_options, active)
        return _dispatch_output_avg(original_forward, state, x, context, rope_emb, transformer_options, active)
    except Exception as exc:
        logger.exception("Anima artist mixer failed at block %s, falling back to original cross-attn: %s", layer_idx, exc)
        return original_forward(x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)


def _dispatch_output_avg(original_forward, state, x, context, rope_emb, transformer_options, active):
    batch_size = context.shape[0]
    mask = _resolve_row_mask(
        transformer_options.get("cond_or_uncond") if isinstance(transformer_options, dict) else None,
        batch_size,
        state.apply_uncond,
    )
    raw_weights = [float(weight) for _, weight in active]
    total_abs = sum(abs(w) for w in raw_weights)
    if total_abs <= 1e-8:
        return original_forward(x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)
    denom = max(1.0, total_abs)
    weights = [w / denom for w in raw_weights]
    artist_total = None
    if state.batched and len(active) > 1 and not state.batched_disabled:
        try:
            artist_total = _artist_forward_batched(
                original_forward,
                x,
                context,
                rope_emb,
                transformer_options,
                active,
                weights,
                state.fusion_mode,
            )
        except Exception as exc:
            logger.warning("Anima artist mixer batched path failed, using serial artist forwards: %s", exc)
            state.batched_disabled = True
            artist_total = None
    if artist_total is None:
        for (artist, _), weight in zip(active, weights):
            artist_context = _broadcast_batch(_to_context(artist.cond, context), batch_size)
            kv = torch.cat([context, artist_context], dim=1) if state.fusion_mode == FUSION_CONCAT_WITH_BASE else artist_context
            out_i = original_forward(x, context=kv, rope_emb=rope_emb, transformer_options=transformer_options)
            artist_total = out_i * weight if artist_total is None else artist_total + out_i * weight
    strength = _clamp(float(state.global_strength), 0.0, 2.0)
    base_out = original_forward(x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)
    out = base_out.clone()
    for idx, hit in enumerate(mask):
        if hit:
            out[idx] = base_out[idx] * (1.0 - strength) + artist_total[idx] * strength
    return out


def _dispatch_concat(original_forward, state, x, context, rope_emb, transformer_options, active):
    batch_size = context.shape[0]
    mask = _resolve_row_mask(
        transformer_options.get("cond_or_uncond") if isinstance(transformer_options, dict) else None,
        batch_size,
        state.apply_uncond,
    )
    raw_weights = [float(weight) for _, weight in active]
    total_abs = sum(abs(w) for w in raw_weights)
    if total_abs <= 1e-8:
        return original_forward(x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)
    denom = max(1.0, total_abs)
    parts = []
    for artist, weight in active:
        artist_context = _broadcast_batch(_to_context(artist.cond, context), batch_size)
        parts.append(artist_context * (weight / denom))
    combined = torch.cat(parts, dim=1)
    if state.fusion_mode == FUSION_CONCAT_WITH_BASE:
        merged = torch.cat([context, combined], dim=1)
        artist_out = original_forward(x, context=merged, rope_emb=rope_emb, transformer_options=transformer_options)
    else:
        artist_out = original_forward(x, context=combined, rope_emb=rope_emb, transformer_options=transformer_options)
    strength = _clamp(float(state.global_strength), 0.0, 2.0)
    base_out = original_forward(x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)
    out = base_out.clone()
    for idx, hit in enumerate(mask):
        if hit:
            out[idx] = base_out[idx] * (1.0 - strength) + artist_out[idx] * strength
    return out


def _current_prompts(p):
    if getattr(p, "is_hr_pass", False):
        prompts = getattr(p, "hr_prompts", None) or getattr(p, "all_hr_prompts", None)
    else:
        prompts = getattr(p, "prompts", None) or getattr(p, "all_prompts", None)
    if isinstance(prompts, list) and prompts:
        return [str(item) for item in prompts]
    prompt = getattr(p, "prompt", "")
    return [str(prompt or "")]


def _build_artists(p, rows, num_blocks, shift, optimization, use_cache):
    rows = _normalize_rows(rows, None, shift, optimization)
    prompts = _current_prompts(p)
    artists = []
    for row in rows:
        enabled, artist_text, weight, blocks_text, start, end, peak, curve, _stage, _shift, _auto = row
        if not enabled or not artist_text:
            continue
        names = _split_artist_chain(artist_text)
        if not names:
            continue
        for name in names:
            clean_name, inline_weight = _parse_inline_weight(name)
            if not clean_name:
                continue
            texts = [f"{clean_name}\n{prompt}" if prompt.strip() else clean_name for prompt in prompts]
            cond = _encode_text_batch(p, texts, use_cache=use_cache)
            artists.append(
                ArtistRuntime(
                    name=clean_name,
                    weight=float(weight) * float(inline_weight),
                    blocks=_parse_blocks(blocks_text, num_blocks),
                    start=_clamp(float(start), 0.0, 1.0),
                    end=_clamp(float(end), 0.0, 1.0),
                    peak=_clamp(float(peak), 0.0, 1.0),
                    curve=curve if curve in CURVE_CHOICES else CURVE_SMOOTH,
                    cond=cond,
                )
            )
    return artists


def _optimization_defaults(optimization):
    if optimization == OPT_PERFORMANCE:
        return {"batched": True, "max_artists": 6}
    if optimization == OPT_QUALITY:
        return {"batched": True, "max_artists": 16}
    return {"batched": True, "max_artists": 10}


class Script(scripts.Script):
    def title(self):
        return _t("title")

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(_t("accordion"), open=False, elem_id=self.elem_id("accordion")):
            with gr.Row():
                intro_language = gr.Radio(
                    choices=["English", "中文"],
                    value="English",
                    label="Description language / 说明语言",
                    elem_id=self.elem_id("intro_language"),
                )
            intro = gr.Markdown(value=INTRO_EN, elem_id=self.elem_id("intro"))
            intro_language.change(fn=_intro_text, inputs=[intro_language], outputs=[intro], queue=False, show_progress=False)

            enable = gr.Checkbox(label=_t("enable"), value=False, elem_id=self.elem_id("enable"))
            with gr.Row():
                optimization = gr.Dropdown(label=_t("optimization"), choices=OPT_PRESETS, value=OPT_BALANCE, elem_id=self.elem_id("optimization"))
                global_strength = gr.Slider(label=_t("global_strength"), minimum=0.0, maximum=2.0, step=0.01, value=0.7, elem_id=self.elem_id("global_strength"))
                enable_cache = gr.Checkbox(label=_t("cache"), value=True, elem_id=self.elem_id("enable_cache"))
            with gr.Row():
                combine_mode = gr.Dropdown(label=_t("combine"), choices=[COMBINE_OUTPUT_AVG, COMBINE_CONCAT], value=COMBINE_OUTPUT_AVG, elem_id=self.elem_id("combine_mode"))
                fusion_mode = gr.Dropdown(label=_t("fusion"), choices=[FUSION_INTERPOLATE, FUSION_CONCAT_WITH_BASE], value=FUSION_INTERPOLATE, elem_id=self.elem_id("fusion_mode"))
                apply_uncond = gr.Checkbox(label=_t("apply_uncond"), value=False, elem_id=self.elem_id("apply_uncond"))

            with gr.Tab("Base"):
                with gr.Row():
                    base_row_count = gr.Number(label=_t("row_count"), value=3, precision=0, elem_id=self.elem_id("base_row_count"))
                    base_shift = gr.Number(label=_t("base_shift"), value=3.0, precision=4, elem_id=self.elem_id("base_shift"))
                    base_apply_presets = gr.Button(_t("normalize_rows"), elem_id=self.elem_id("base_apply_presets"))
                base_rows = gr.Dataframe(
                    label=_t("artist_table"),
                    headers=TABLE_HEADERS,
                    datatype=TABLE_DATATYPES,
                    value=_default_rows(3, 3.0, OPT_BALANCE),
                    row_count=(3, "dynamic"),
                    col_count=(len(TABLE_HEADERS), "fixed"),
                    interactive=True,
                    elem_id=self.elem_id("base_rows"),
                )

            with gr.Tab("Hires. fix"):
                hires_independent = gr.Checkbox(label=_t("hires_independent"), value=False, elem_id=self.elem_id("hires_independent"))
                with gr.Row():
                    hires_row_count = gr.Number(label=_t("hr_row_count"), value=3, precision=0, elem_id=self.elem_id("hires_row_count"))
                    hires_shift = gr.Number(label=_t("hires_shift"), value=3.0, precision=4, elem_id=self.elem_id("hires_shift"))
                    hires_apply_presets = gr.Button(_t("normalize_rows"), elem_id=self.elem_id("hires_apply_presets"))
                hires_rows = gr.Dataframe(
                    label=_t("artist_table"),
                    headers=TABLE_HEADERS,
                    datatype=TABLE_DATATYPES,
                    value=_default_rows(3, 3.0, OPT_BALANCE),
                    row_count=(3, "dynamic"),
                    col_count=(len(TABLE_HEADERS), "fixed"),
                    interactive=True,
                    elem_id=self.elem_id("hires_rows"),
                )

            with gr.Tab(_t("template")):
                with gr.Row():
                    template_dropdown = gr.Dropdown(label=_t("template"), choices=_template_choices(), value=None, allow_custom_value=False, elem_id=self.elem_id("template_dropdown"))
                    template_apply_target = gr.Dropdown(label=_t("apply_target"), choices=[APPLY_BASE, APPLY_HIRES, APPLY_BOTH], value=APPLY_BASE, elem_id=self.elem_id("template_apply_target"))
                    template_apply = gr.Button(_t("apply_template"), variant="primary", elem_id=self.elem_id("template_apply"))
                with gr.Row():
                    template_name = gr.Textbox(label=_t("template_name"), value="", elem_id=self.elem_id("template_name"))
                    template_save = gr.Button(_t("save_template"), elem_id=self.elem_id("template_save"))
                with gr.Row():
                    rename_to = gr.Textbox(label=_t("rename_to"), value="", elem_id=self.elem_id("rename_to"))
                    rename_button = gr.Button(_t("rename_template"), elem_id=self.elem_id("rename_button"))
                    delete_button = gr.Button(_t("delete_template"), elem_id=self.elem_id("delete_button"))
                template_status = gr.Markdown(value="", elem_id=self.elem_id("template_status"))

            with gr.Tab(_t("help")):
                gr.Markdown(value=HELP_ZH if _language() == "zh" else HELP_EN)

        base_row_count.change(
            fn=_resize_table,
            inputs=[base_rows, base_row_count, base_shift, optimization],
            outputs=[base_rows],
            queue=False,
            show_progress=False,
        )
        hires_row_count.change(
            fn=_resize_table,
            inputs=[hires_rows, hires_row_count, hires_shift, optimization],
            outputs=[hires_rows],
            queue=False,
            show_progress=False,
        )
        base_shift.change(
            fn=_apply_shift_to_rows,
            inputs=[base_rows, base_shift, optimization],
            outputs=[base_rows],
            queue=False,
            show_progress=False,
        )
        hires_shift.change(
            fn=_apply_shift_to_rows,
            inputs=[hires_rows, hires_shift, optimization],
            outputs=[hires_rows],
            queue=False,
            show_progress=False,
        )
        base_apply_presets.click(
            fn=_resize_table,
            inputs=[base_rows, base_row_count, base_shift, optimization],
            outputs=[base_rows],
            queue=False,
            show_progress=False,
        )
        hires_apply_presets.click(
            fn=_resize_table,
            inputs=[hires_rows, hires_row_count, hires_shift, optimization],
            outputs=[hires_rows],
            queue=False,
            show_progress=False,
        )
        template_save.click(
            fn=_save_template_ui,
            inputs=[
                template_name,
                base_rows,
                hires_rows,
                hires_independent,
                base_shift,
                hires_shift,
                global_strength,
                optimization,
                combine_mode,
                fusion_mode,
                apply_uncond,
                enable_cache,
            ],
            outputs=[template_dropdown, template_status],
            queue=False,
            show_progress=False,
        )
        rename_button.click(fn=_rename_template_ui, inputs=[template_dropdown, rename_to], outputs=[template_dropdown, template_status], queue=False, show_progress=False)
        delete_button.click(fn=_delete_template_ui, inputs=[template_dropdown], outputs=[template_dropdown, template_status], queue=False, show_progress=False)
        template_apply.click(
            fn=_apply_template_ui,
            inputs=[template_dropdown, template_apply_target, base_rows, hires_rows],
            outputs=[
                base_rows,
                hires_rows,
                base_row_count,
                hires_row_count,
                hires_independent,
                base_shift,
                hires_shift,
                global_strength,
                optimization,
                combine_mode,
                fusion_mode,
                apply_uncond,
                enable_cache,
                template_status,
            ],
            queue=False,
            show_progress=False,
        )

        return [
            enable,
            base_rows,
            hires_rows,
            hires_independent,
            base_shift,
            hires_shift,
            global_strength,
            optimization,
            combine_mode,
            fusion_mode,
            apply_uncond,
            enable_cache,
        ]

    def process_before_every_sampling(self, p, *args, **kwargs):
        global _ACTIVE_STATE
        if len(args) < 12:
            return
        (
            enable,
            base_rows,
            hires_rows,
            hires_independent,
            base_shift,
            hires_shift,
            global_strength,
            optimization,
            combine_mode,
            fusion_mode,
            apply_uncond,
            enable_cache,
        ) = args[:12]

        _unpatch_cross_attn()
        _ACTIVE_STATE = None
        if not enable:
            return
        if getattr(p, "_ad_inner", False):
            return
        unet = getattr(getattr(p, "sd_model", None), "forge_objects", None)
        unet = getattr(unet, "unet", None)
        if unet is None:
            return
        dm, msg = _validate_anima_unet(unet)
        if dm is None:
            logger.warning("Anima artist mixer disabled: %s", msg)
            return
        rows = hires_rows if getattr(p, "is_hr_pass", False) and hires_independent else base_rows
        shift = hires_shift if getattr(p, "is_hr_pass", False) and hires_independent else base_shift
        optimization = optimization if optimization in OPT_PRESETS else OPT_BALANCE
        try:
            artists = _build_artists(p, rows, len(dm.blocks), shift, optimization, bool(enable_cache))
        except Exception as exc:
            logger.exception("Failed to encode Anima artist mixer artists: %s", exc)
            return
        defaults = _optimization_defaults(optimization)
        max_artists = defaults["max_artists"]
        if len(artists) > max_artists:
            logger.warning("Anima artist mixer active artists %d exceed %d for %s preset; truncating.", len(artists), max_artists, optimization)
            artists = artists[:max_artists]
        if not artists:
            return
        state = MixerState(
            run_id=uuid.uuid4().hex,
            enabled=True,
            global_strength=_clamp(_to_float(global_strength, 0.7), 0.0, 2.0),
            combine_mode=combine_mode if combine_mode in {COMBINE_OUTPUT_AVG, COMBINE_CONCAT} else COMBINE_OUTPUT_AVG,
            fusion_mode=fusion_mode if fusion_mode in {FUSION_INTERPOLATE, FUSION_CONCAT_WITH_BASE} else FUSION_INTERPOLATE,
            apply_uncond=bool(apply_uncond),
            batched=bool(defaults["batched"]),
            artists=artists,
        )
        _install_cross_attn_patch(dm, state)
        _ACTIVE_STATE = state
        p.extra_generation_params["Anima Artist Mixer"] = (
            f"{len(artists)} artists, strength={state.global_strength:.2f}, "
            f"{state.combine_mode}/{state.fusion_mode}, preset={optimization}"
        )

    def postprocess(self, p, processed, *args, **kwargs):
        global _ACTIVE_STATE
        _unpatch_cross_attn()
        _ACTIVE_STATE = None
