import json
import logging
import re
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import gradio as gr
import torch

from modules import script_callbacks, scripts, shared


logger = logging.getLogger("anima_artist_scheduled_mixer")

EXTENSION_DIR = Path(__file__).resolve().parents[1]
TEMPLATE_FILE = EXTENSION_DIR / "artist_mixer_templates.json"
CURRENT_SETTINGS_FILE = EXTENSION_DIR / "artist_mixer_current_settings.json"
REFERENCE_URL = "https://github.com/An1X3R/Anima-Artist-Mixer"

LANGUAGE_OPTION = "anima_artist_scheduled_mixer_language"
LAST_TEMPLATE_OPTION = "anima_artist_scheduled_mixer_last_template"
LAST_TEMPLATE_TARGET_OPTION = "anima_artist_scheduled_mixer_last_template_target"
LANGUAGE_CHOICES = ("zh", "en")

FUSION_INTERPOLATE = "interpolate"
FUSION_CONCAT_WITH_BASE = "concat_with_base"
FUSION_QUALITY_DELTA = "quality_delta"
# quality_delta is already anchored by comparing artist_out against base_out.
# Concatenating the base context again dilutes the artist token influence.
BASE_CONTEXT_FUSIONS = {FUSION_CONCAT_WITH_BASE}
COMBINE_OUTPUT_AVG = "output_avg"
COMBINE_CONCAT = "concat"
COMBINE_LOWRANK_AVG = "lowrank_avg"

ANCHOR_SEEDS_POOL = (42, 100, 200, 300)
ANCHOR_SEEDS_MAX = len(ANCHOR_SEEDS_POOL)
STATIC_CAPTURE_K_DEFAULT = 6
STATIC_CAPTURE_K_MAX = 12

ADVANCED_SETTING_KEYS = (
    "artist_ema_alpha",
    "lowrank_k",
    "artist_static_capture",
    "static_capture_k",
    "artist_anchor_q",
    "anchor_seed_list",
    "anchor_seeds_count",
    "anchor_user_blend",
    "anchor_deep_layer_threshold",
    "stabilizer_end_percent",
    "style_balance",
    "structure_preserve",
    "delta_norm_cap",
)

ADVANCED_DEFAULTS = {
    "artist_ema_alpha": 0.0,
    "lowrank_k": 1,
    "artist_static_capture": False,
    "static_capture_k": STATIC_CAPTURE_K_DEFAULT,
    "artist_anchor_q": False,
    "anchor_seed_list": "",
    "anchor_seeds_count": 1,
    "anchor_user_blend": 0.0,
    "anchor_deep_layer_threshold": -1,
    "stabilizer_end_percent": 1.0,
    "style_balance": 0.0,
    "structure_preserve": 0.0,
    "delta_norm_cap": 0.0,
}

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

MAX_ARTIST_ROWS = 32
DEFAULT_ARTIST_ROWS = 4

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
    "Auto Shift",
]

TABLE_DATATYPES = [
    "str",
    "str",
    "number",
    "str",
    "number",
    "number",
    "number",
    "str",
    "str",
    "str",
]

OPTION_LABELS = {
    "optimization": OrderedDict(
        [
            (OPT_PERFORMANCE, {"en": "Performance", "zh": "性能", "aliases": ("Perf", "Speed", "速度")}),
            (OPT_BALANCE, {"en": "Balance", "zh": "平衡", "aliases": ("Balanced", "均衡")}),
            (OPT_QUALITY, {"en": "Quality", "zh": "质量", "aliases": ("High quality", "高质量")}),
        ]
    ),
    "combine": OrderedDict(
        [
            (COMBINE_OUTPUT_AVG, {"en": "Output average", "zh": "输出平均", "aliases": ("output average", "输出平均值")}),
            (COMBINE_CONCAT, {"en": "Token concatenation", "zh": "token拼接", "aliases": ("Token concat", "concat", "拼接", "令牌拼接")}),
            (COMBINE_LOWRANK_AVG, {"en": "Low-rank average", "zh": "低秩平均", "aliases": ("lowrank", "low-rank", "低秩约束")}),
        ]
    ),
    "fusion": OrderedDict(
        [
            (FUSION_INTERPOLATE, {"en": "Interpolate", "zh": "插值融合", "aliases": ("interp", "插值")}),
            (
                FUSION_CONCAT_WITH_BASE,
                {"en": "Concat with base", "zh": "拼接底图条件", "aliases": ("concat base", "拼接原始条件")},
            ),
            (
                FUSION_QUALITY_DELTA,
                {
                    "en": "Quality-safe delta",
                    "zh": "保真增量",
                    "aliases": ("delta", "safe delta", "quality delta", "保真差值", "质量安全增量"),
                },
            ),
        ]
    ),
    "curve": OrderedDict(
        [
            (CURVE_SMOOTH, {"en": "Smooth", "zh": "平滑", "aliases": ("平滑曲线",)}),
            (CURVE_HOLD, {"en": "Hold", "zh": "保持", "aliases": ("恒定", "全程保持")}),
            (CURVE_TRIANGLE, {"en": "Triangle", "zh": "三角峰", "aliases": ("Triangle peak", "三角")}),
            (CURVE_FRONT, {"en": "Front loaded", "zh": "前段强化", "aliases": ("Early", "前段")}),
            (CURVE_BACK, {"en": "Back loaded", "zh": "后段强化", "aliases": ("Late", "后段")}),
        ]
    ),
    "stage": OrderedDict(
        [
            (PRESET_CUSTOM, {"en": "Custom", "zh": "自定义", "aliases": ("手动",)}),
            (PRESET_COMPOSITION, {"en": "Composition", "zh": "构图", "aliases": ("Compose", "布局")}),
            (PRESET_CHARACTER, {"en": "Character", "zh": "人物", "aliases": ("Subject", "角色")}),
            (PRESET_STYLE, {"en": "Style", "zh": "画风", "aliases": ("Artist style", "风格")}),
        ]
    ),
    "apply_target": OrderedDict(
        [
            (APPLY_BASE, {"en": "Base", "zh": "底图", "aliases": ("Base image", "底图设置")}),
            (APPLY_HIRES, {"en": "Hires. fix", "zh": "高分修复", "aliases": ("Hires", "Highres", "高分")}),
            (APPLY_BOTH, {"en": "Both", "zh": "两者", "aliases": ("All", "全部")}),
        ]
    ),
}

TABLE_HEADER_LABELS = OrderedDict(
    [
        ("Enabled", {"en": "Enabled", "zh": "启用"}),
        ("Artist", {"en": "Artist", "zh": "画师"}),
        ("Weight", {"en": "Weight", "zh": "权重"}),
        ("Blocks", {"en": "Blocks", "zh": "层数"}),
        ("Start", {"en": "Start", "zh": "开始"}),
        ("End", {"en": "End", "zh": "结束"}),
        ("Peak", {"en": "Peak", "zh": "峰值"}),
        ("Curve", {"en": "Curve", "zh": "曲线"}),
        ("Stage", {"en": "Stage", "zh": "阶段"}),
        ("Auto Shift", {"en": "Auto Shift", "zh": "自动偏移"}),
    ]
)


LANG = {
    "zh": {
        "title": "Anima 画师串调度混合",
        "accordion": "Anima 画师串调度混合",
        "settings_label": "Anima 画师串调度混合界面语言",
        "intro_language": "界面语言",
        "language_saved": "界面语言已保存，插件面板已切换。",
        "language_save_failed": "界面语言保存失败，请检查控制台日志。",
        "enable": "启用画师串混合",
        "base_tab": "底图",
        "hires_tab": "高分修复",
        "base_panel": "底图画师",
        "hires_panel": "高分辨率修复画师",
        "hires_independent": "高分辨率修复使用独立画师串；关闭时继承底图设置",
        "disable_hires_mixing": "高分辨率修复不启用画师串混合",
        "row_count": "画师行数",
        "hr_row_count": "高分画师行数",
        "shift_runtime_hint": "阶段/自动偏移读取生成参数区当前 Shift；生成时会再次读取实际底图/高分 Shift。",
        "artist_table": "画师设置",
        "global_strength": "全局画师强度",
        "optimization": "优化预设",
        "combine": "组合模式",
        "fusion": "融合模式",
        "apply_uncond": "同时作用于负条件",
        "cache": "启用文本编码缓存",
        "advanced_tab": "高级稳定",
        "advanced_hint": "高级稳定器默认关闭；同时启用过多功能会增加显存和计算开销。",
        "artist_ema_alpha": "EMA 平滑强度",
        "lowrank_k": "低秩约束阶数",
        "artist_static_capture": "启用 Static Capture",
        "static_capture_k": "Static Capture 预热步数",
        "artist_anchor_q": "启用 Anchor Q",
        "anchor_seed_list": "Anchor 固定种子（逗号分隔）",
        "anchor_seeds_count": "内置 Anchor 种子数量",
        "anchor_user_blend": "Anchor 与当前 Q 混合比例",
        "anchor_deep_layer_threshold": "Anchor 深层截止（-1 为全部层）",
        "stabilizer_end_percent": "稳定器结束进度",
        "style_balance": "Style Balance 强度",
        "structure_preserve": "Structure Guard 结构保持",
        "delta_norm_cap": "Structure Guard 增量上限",
        "template": "模板",
        "template_name": "模板名称",
        "save_base_template": "保存底图画师串设置",
        "save_hires_template": "保存高分辨率修复模板",
        "hires_template_disabled": "请先开启“高分辨率修复使用独立画师串”，再保存高分辨率修复模板。",
        "rename_to": "重命名为",
        "rename_template": "重命名模板",
        "delete_template": "删除模板",
        "reset_defaults": "全部恢复默认",
        "apply_target": "模板应用目标",
        "apply_template": "应用模板",
        "normalize_rows": "刷新阶段预设",
        "help": "说明",
        "status": "状态",
        "empty_template_name": "模板名称为空。",
        "saved_base_template": "已保存底图画师串模板：{name}",
        "saved_hires_template": "已保存高分辨率修复模板：{name}",
        "no_template_selected": "未选择模板。",
        "new_template_name_empty": "新模板名称为空。",
        "renamed_template": "已重命名模板为：{name}",
        "deleted_template": "已删除模板：{name}",
        "no_template_deleted": "没有删除任何模板。",
        "applied_template": "已将模板 `{name}` 应用到{target}。",
        "defaults_restored": "已恢复全部默认设置。",
    },
    "en": {
        "title": "Anima Artist Scheduled Mixer",
        "accordion": "Anima Artist Scheduled Mixer",
        "settings_label": "Anima Artist Scheduled Mixer UI language",
        "intro_language": "UI language",
        "language_saved": "UI language saved. The plugin panel has been updated.",
        "language_save_failed": "Failed to save UI language. Check the console log.",
        "enable": "Enable artist mixing",
        "base_tab": "Base",
        "hires_tab": "Hires. fix",
        "base_panel": "Base artists",
        "hires_panel": "Hires. fix artists",
        "hires_independent": "Use independent Hires. fix artist chain; disabled = inherit base settings",
        "disable_hires_mixing": "Disable artist mixing during Hires. fix",
        "row_count": "Artist rows",
        "hr_row_count": "Hires artist rows",
        "shift_runtime_hint": "Stage/Auto Shift reads the current Shift from the generation panel; sampling reads the actual Base/Hires Shift again.",
        "artist_table": "Artist settings",
        "global_strength": "Global artist strength",
        "optimization": "Optimization preset",
        "combine": "Combine mode",
        "fusion": "Fusion mode",
        "apply_uncond": "Apply to unconditional rows",
        "cache": "Enable text-encoding cache",
        "advanced_tab": "Advanced stability",
        "advanced_hint": "Advanced stabilizers are disabled by default. Combining several of them increases VRAM and compute cost.",
        "artist_ema_alpha": "EMA smoothing strength",
        "lowrank_k": "Low-rank constraint rank",
        "artist_static_capture": "Enable Static Capture",
        "static_capture_k": "Static Capture warmup steps",
        "artist_anchor_q": "Enable Anchor Q",
        "anchor_seed_list": "Fixed anchor seeds (comma separated)",
        "anchor_seeds_count": "Built-in anchor seed count",
        "anchor_user_blend": "Anchor/current Q blend",
        "anchor_deep_layer_threshold": "Anchor deep-layer cutoff (-1 = all)",
        "stabilizer_end_percent": "Stabilizer end progress",
        "style_balance": "Style Balance strength",
        "structure_preserve": "Structure Guard preservation",
        "delta_norm_cap": "Structure Guard delta cap",
        "template": "Template",
        "template_name": "Template name",
        "save_base_template": "Save base artist-chain settings",
        "save_hires_template": "Save Hires. fix template",
        "hires_template_disabled": "Enable independent Hires. fix artist chain before saving a Hires. fix template.",
        "rename_to": "Rename to",
        "rename_template": "Rename template",
        "delete_template": "Delete template",
        "reset_defaults": "Reset all to defaults",
        "apply_target": "Apply target",
        "apply_template": "Apply template",
        "normalize_rows": "Refresh stage presets",
        "help": "Guide",
        "status": "Status",
        "empty_template_name": "Template name is empty.",
        "saved_base_template": "Saved base artist-chain template: {name}",
        "saved_hires_template": "Saved Hires. fix template: {name}",
        "no_template_selected": "No template selected.",
        "new_template_name_empty": "New template name is empty.",
        "renamed_template": "Renamed template to: {name}",
        "deleted_template": "Deleted template: {name}",
        "no_template_deleted": "No template deleted.",
        "applied_template": "Applied template `{name}` to {target}.",
        "defaults_restored": "Restored all settings to defaults.",
    },
}


INTRO_EN = f"""
Independent artist encoding and scheduled cross-attention mixing for Anima.
Each artist row is encoded separately as `artist + base prompt`, then mixed inside Anima cross-attention. The default `Output average + Interpolate` path follows the stronger reference behavior from [{REFERENCE_URL}]({REFERENCE_URL}); `Quality-safe delta` is still available when you want a more conservative Forge-friendly blend.

The option labels follow the selected UI language, while templates keep stable internal values. Old templates that stored English, Chinese, or raw values should still load correctly after switching languages.

Thanks to **An1X3R/Anima-Artist-Mixer** and **汐浮尘/utowo** for the original split-and-encode/cross-attention design.
"""

INTRO_ZH = f"""
面向 Anima 的独立画师编码与按阶段 cross-attention 混合。每个画师会单独按 `画师 + 主提示词` 编码，再在 Anima 的 cross-attention 内混合。默认的“输出平均 + 插值融合”沿用 [{REFERENCE_URL}]({REFERENCE_URL}) 的强混合路线；需要更保守的 Forge 友好混合时，也可以手动切到“保真增量”。

面板选项会跟随界面语言显示，模板内部使用稳定值保存。旧模板即使保存过英文、中文或原始值，切换语言后也会尽量自动识别并回显。

感谢 **An1X3R/Anima-Artist-Mixer** 与 **汐浮尘/utowo** 的原始独立编码和 cross-attention 混合设计。
"""

HELP_EN = """
**Columns**

`Artist`: one artist prompt per row. It may include an inline row multiplier such as `(wlop:1.2)`, `[wlop:0.8]`, or `wlop:1.2`; the final row strength is `Weight x inline multiplier`. Keep comma-separated chains for quick migration if needed, but one artist per row gives the cleanest control.

`Weight` controls this artist's relative and absolute contribution. `Blocks` accepts `0-27`, `0,3,5-12`, or negative indices such as `-1`. `Start` and `End` are denoise progress values from 0 to 1. `Peak` is the point inside that window where the row reaches full strength; it matters most for Smooth and Triangle curves.

`Curve` shapes the row strength inside the Start/End window. Hold stays at full strength for the whole window. Smooth fades in and out around the peak. Triangle rises and falls linearly. Front loaded starts strong and fades later; Back loaded does the opposite.

`Stage` is a timing preset. Custom keeps your manual values. Composition starts early, Character focuses on the middle, and Style runs late. Choosing a non-Custom stage refreshes Start/End/Peak immediately and enables Auto Shift for that row. Auto Shift keeps the row aligned with the current Forge Shift value; turn it off afterward if you want to fine-tune the numbers by hand.

**Strength**

Global artist strength controls how much of the artist branch is injected into the normal prompt cross-attention output. Row weights choose each artist's share; small row weights also reduce total influence when the active weights sum below 1. A row value like `Artist=(wlop:1.2), Weight=0.5` acts like a final artist row weight of `0.6`.

**Optimization Preset**

Performance limits the default block range and caps the active artist count lower, so it is cheaper. Balance keeps the reference-friendly default while allowing more artists. Quality keeps the broad block range and raises the artist cap for heavier mixes.

**Combine Mode**

Output average runs each artist branch separately, then averages the artist outputs by weight. Low-rank average constrains the artist deltas to a rank-K subspace for better cross-seed consistency. Token concatenation joins the artist contexts first and sends them through attention together; it is useful for compact mixes, but the artists can interact more strongly.

**Fusion Mode**

Interpolate blends between the base prompt output and the artist branch output. It is the strongest and most reference-compatible default. Concat with base lets attention see base tokens and artist tokens together before blending. Quality-safe delta keeps the base output and adds a norm-limited artist difference, which is steadier but can look much weaker.

**Advanced Stability**

EMA smooths artist outputs across sampling steps. Static Capture averages the first K distinct steps and freezes the result. Anchor Q performs a fixed-seed pre-pass and uses its hidden states for artist attention queries. Style Balance equalizes artist-delta volume before row weights are applied. Structure Guard projects and caps the final artist delta to protect composition. `Stabilizer end progress` limits EMA, Static Capture, and Anchor Q to the early portion of sampling.

The UI localizes option labels. Templates are saved with stable internal values, so English/Chinese/raw values can be applied across language modes.
"""

HELP_ZH = """
**列说明**

`画师` 每行建议填一个画师提示词。可以直接写行内倍率，例如 `(wlop:1.2)`、`[wlop:0.8]` 或 `wlop:1.2`；最终强度是 `权重 x 行内倍率`。为了迁移旧画师串，逗号分隔仍可用，但每行一个画师最方便单独控制。

`权重` 控制该画师的相对比例，也会在总权重低于 1 时降低实际介入。`层数` 支持 `0-27`、`0,3,5-12`，也支持 `-1` 这种倒数索引。`开始` 和 `结束` 是 0 到 1 的去噪进度。`峰值` 表示这一行在窗口内达到满强度的位置，对平滑和三角峰曲线最明显。

`曲线` 决定窗口内强度怎样变化。保持会在整个窗口内满强度。平滑会围绕峰值淡入淡出。三角峰会线性升到峰值再线性降下。前段强化是一开始更强、后面变弱；后段强化则相反。

`阶段` 是时间预设。自定义会保留你手动填写的值。构图偏前段，人物偏中段，画风偏后段。选择自定义之外的阶段时，界面会立即刷新开始、结束和峰值，并为这一行打开自动偏移。自动偏移会跟随 Forge 生成参数区的 Shift；如果之后想微调数字，再把自动偏移关掉即可。

**强度**

全局画师强度用于控制画师分支往原始提示词 cross-attention 输出里注入多少。单行权重决定画师占比；当活跃权重总和低于 1 时，也会降低整体介入。例如 `画师=(wlop:1.2)，权重=0.5`，最终该行画师权重相当于 `0.6`。

**优化预设**

性能会收窄默认层数范围，并降低可同时参与的画师上限，生成更省。平衡保留更接近参考插件的默认范围，同时允许较多画师。质量保留宽层数范围，并提高画师上限，适合更重的混合。

**组合模式**

输出平均会让每个画师分支分别跑 attention，再按权重平均输出，是最稳定、也最接近参考推荐的方式。低秩平均会把多个画师的增量约束到 K 阶子空间，用于提高跨 seed 一致性。token拼接会先把多个画师上下文拼在一起再送入 attention，适合更紧凑的混合，但画师之间会更容易互相影响。

**融合模式**

插值融合会在主提示词输出和画师分支输出之间做插值，是最强、也最接近参考插件默认效果的方式。拼接底图条件会先让 attention 同时看到底图 token 和画师 token，再进行混合。保真增量会保留底图输出，只加入有范数限制的画师差值，更稳但也可能明显偏弱。

**高级稳定**

EMA 会跨采样步平滑画师输出；Static Capture 会平均前 K 个不同采样步并冻结结果；Anchor Q 会用固定种子预跑并将捕获的隐藏状态用于画师 attention 查询；Style Balance 会在应用画师权重前平衡各画师增量幅度；Structure Guard 会投影并限制最终画师增量以保护构图。“稳定器结束进度”用于限制 EMA、Static Capture 和 Anchor Q 只作用于采样前段。

面板会按语言汉化选项。模板保存稳定内部值，因此中文、英文或旧 raw 值模板都可以跨语言应用。
"""


def _language():
    value = getattr(shared.opts, LANGUAGE_OPTION, "zh")
    return value if value in LANGUAGE_CHOICES else "zh"


def _t(key):
    return LANG.get(_language(), LANG["zh"]).get(key, key)


def _choice_to_language(language):
    if language in LANGUAGE_CHOICES:
        return language
    return "zh" if language == "中文" else "en"


def _language_choice(language=None):
    return "中文" if _choice_to_language(language or _language()) == "zh" else "English"


def _intro_text(language=None):
    return INTRO_ZH if _choice_to_language(language or _language()) == "zh" else INTRO_EN


def _intro_default(language=None):
    return _intro_text(language or _language())


def _html_attr(value):
    return str(value).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def _status_html(message, refresh_language=None):
    refresh_attr = f' data-anima-artist-language-refresh="{_html_attr(refresh_language)}"' if refresh_language else ""
    return f"<span{refresh_attr}>{_html_attr(message)}</span>"


def _save_ui_language(language):
    language_code = _choice_to_language(language)
    try:
        if LANGUAGE_OPTION in shared.opts.data_labels:
            shared.opts.set(LANGUAGE_OPTION, language_code, run_callbacks=False)
        else:
            shared.opts.data[LANGUAGE_OPTION] = language_code
        shared.opts.save(shared.config_filename)
        return _intro_text(language_code), _status_html(LANG[language_code]["language_saved"], refresh_language=language_code)
    except Exception:
        logger.exception("Failed to save Anima artist mixer UI language")
        return _intro_text(language_code), _status_html(LANG[language_code]["language_save_failed"])


def _set_persistent_option(name, value):
    if name in shared.opts.data_labels:
        shared.opts.set(name, value, run_callbacks=False)
    else:
        shared.opts.data[name] = value


def _last_template_name():
    name = str(getattr(shared.opts, LAST_TEMPLATE_OPTION, "") or "").strip()
    return name if name in _template_data() else None


def _last_template_target():
    return _option_key("apply_target", getattr(shared.opts, LAST_TEMPLATE_TARGET_OPTION, APPLY_BASE), APPLY_BASE)


def _remember_template(name, target):
    name = str(name or "").strip()
    if not name:
        return
    target_key = _option_key("apply_target", target, APPLY_BASE)
    try:
        _set_persistent_option(LAST_TEMPLATE_OPTION, name)
        _set_persistent_option(LAST_TEMPLATE_TARGET_OPTION, target_key)
        shared.opts.save(shared.config_filename)
    except Exception:
        logger.exception("Failed to remember last Anima artist mixer template")


def _bool_label(value, language=None):
    language = language or _language()
    return ("是" if _to_bool(value, True) else "否") if language == "zh" else ("Yes" if _to_bool(value, True) else "No")


def _option_key(group, value, fallback=None):
    mapping = OPTION_LABELS.get(group, {})
    if fallback is None and mapping:
        fallback = next(iter(mapping))
    if value is None:
        return fallback
    text = str(value).strip()
    if text in mapping:
        return text
    folded = text.casefold()
    for key, labels in mapping.items():
        candidates = [key, labels.get("en", ""), labels.get("zh", "")]
        candidates.extend(labels.get("aliases", ()))
        if folded in {str(item).strip().casefold() for item in candidates if item is not None}:
            return key
    return fallback


def _option_label(group, value, language=None):
    language = language or _language()
    key = _option_key(group, value)
    labels = OPTION_LABELS.get(group, {}).get(key, {})
    return labels.get(language) or labels.get("en") or str(value or "")


def _option_choices(group, language=None):
    language = language or _language()
    return [_option_label(group, key, language) for key in OPTION_LABELS.get(group, {})]


def _table_headers(language=None):
    language = language or _language()
    return [TABLE_HEADER_LABELS[h].get(language, h) for h in TABLE_HEADERS]


def _row_component_headers(language=None):
    language = language or _language()
    headers = TABLE_HEADER_LABELS
    return {
        "enabled": headers["Enabled"].get(language, "Enabled"),
        "artist": headers["Artist"].get(language, "Artist"),
        "weight": headers["Weight"].get(language, "Weight"),
        "blocks": headers["Blocks"].get(language, "Blocks"),
        "start": headers["Start"].get(language, "Start"),
        "end": headers["End"].get(language, "End"),
        "peak": headers["Peak"].get(language, "Peak"),
        "curve": headers["Curve"].get(language, "Curve"),
        "stage": headers["Stage"].get(language, "Stage"),
        "auto_shift": headers["Auto Shift"].get(language, "Auto Shift"),
    }


def _header_key(value):
    text = str(value or "").strip()
    if text in TABLE_HEADERS:
        return text
    folded = text.casefold()
    for key, labels in TABLE_HEADER_LABELS.items():
        candidates = [key, labels.get("en", ""), labels.get("zh", "")]
        if folded in {str(item).strip().casefold() for item in candidates if item is not None}:
            return key
    return text


def _normalize_bool_display(value, fallback=True):
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in {"yes", "no", "是", "否"}:
            return value.strip()
    return _bool_label(_to_bool(value, fallback))


def _rows_for_display(rows, language=None):
    out = []
    for row in _coerce_table(rows):
        row += [None] * (len(TABLE_HEADERS) - len(row))
        new_row = list(row[: len(TABLE_HEADERS)])
        new_row[0] = _bool_label(new_row[0], language)
        new_row[7] = _option_label("curve", new_row[7], language)
        new_row[8] = _option_label("stage", new_row[8], language)
        new_row[9] = _bool_label(new_row[9], language)
        out.append(new_row)
    return out


def _rows_for_storage(rows, shift=3.0, optimization=OPT_BALANCE):
    return _normalize_rows(rows, None, shift, optimization, display=False)


def _rows_from_components(*values):
    rows = []
    per_row = 10
    for index in range(0, len(values), per_row):
        chunk = list(values[index : index + per_row])
        if len(chunk) < per_row:
            break
        rows.append(chunk)
    return rows


def _component_values_from_rows(rows, shift=3.0, optimization=OPT_BALANCE, count=None):
    normalized = _normalize_rows(rows, MAX_ARTIST_ROWS, shift, optimization, display=True)
    if count is None:
        count = len(_coerce_table(rows)) or 1
    count = max(1, min(MAX_ARTIST_ROWS, int(_to_float(count, 1))))
    updates = []
    for index, row in enumerate(normalized):
        visible = index < count
        updates.extend(
            [
                gr.update(value=row[0], visible=visible, interactive=True),
                gr.update(value=row[1], visible=visible, interactive=True),
                gr.update(value=row[2], visible=visible, interactive=True),
                gr.update(value=row[3], visible=visible, interactive=True),
                gr.update(value=row[4], visible=visible, interactive=True),
                gr.update(value=row[5], visible=visible, interactive=True),
                gr.update(value=row[6], visible=visible, interactive=True),
                gr.update(value=row[7], visible=visible, interactive=True),
                gr.update(value=row[8], visible=visible, interactive=True),
                gr.update(value=row[9], visible=visible, interactive=True),
            ]
        )
    return updates


def _timing_values_from_rows(rows, shift=3.0, optimization=OPT_BALANCE, count=None):
    normalized = _normalize_rows(rows, MAX_ARTIST_ROWS, shift, optimization, display=True)
    if count is None:
        count = len(_coerce_table(rows)) or 1
    count = max(1, min(MAX_ARTIST_ROWS, int(_to_float(count, 1))))
    updates = []
    for index, row in enumerate(normalized):
        if index < count:
            updates.extend(
                [
                    gr.update(value=row[4], interactive=True),
                    gr.update(value=row[5], interactive=True),
                    gr.update(value=row[6], interactive=True),
                ]
            )
        else:
            updates.extend([gr.update(), gr.update(), gr.update()])
    return updates


def _block_timing_values_from_rows(rows, shift=3.0, optimization=OPT_BALANCE, count=None):
    normalized = _normalize_rows(rows, MAX_ARTIST_ROWS, shift, optimization, display=True)
    if count is None:
        count = len(_coerce_table(rows)) or 1
    count = max(1, min(MAX_ARTIST_ROWS, int(_to_float(count, 1))))
    updates = []
    for index, row in enumerate(normalized):
        if index < count:
            updates.extend(
                [
                    gr.update(value=row[3], interactive=True),
                    gr.update(value=row[4], interactive=True),
                    gr.update(value=row[5], interactive=True),
                    gr.update(value=row[6], interactive=True),
                ]
            )
        else:
            updates.extend([gr.update(), gr.update(), gr.update(), gr.update()])
    return updates


def _resize_row_components(count, shift, optimization, *values):
    rows = _rows_from_components(*values)
    count = max(1, min(MAX_ARTIST_ROWS, int(_to_float(count, len(rows) or 1))))
    return _component_values_from_rows(rows, shift, optimization, count)


def _apply_shift_to_timing_components(count, shift, optimization, *values):
    rows = _rows_from_components(*values)
    count = max(1, min(MAX_ARTIST_ROWS, int(_to_float(count, len(rows) or 1))))
    return _timing_values_from_rows(rows, shift, optimization, count)


def _default_blocks_for_optimization(optimization):
    optimization = _option_key("optimization", optimization, OPT_BALANCE)
    if optimization == OPT_PERFORMANCE:
        return "10-18"
    return "0-27"


def _apply_optimization_to_block_timing_components(count, shift, optimization, *values):
    rows = _rows_from_components(*values)
    count = max(1, min(MAX_ARTIST_ROWS, int(_to_float(count, len(rows) or 1))))
    blocks = _default_blocks_for_optimization(optimization)
    for index in range(min(count, len(rows))):
        row = list(rows[index])
        row += [None] * (len(TABLE_HEADERS) - len(row))
        row[3] = blocks
        if _to_bool(row[9], True) and _option_key("stage", row[8], PRESET_CUSTOM) != PRESET_CUSTOM:
            row[4], row[5], row[6] = _auto_stage_values(row[8], shift)
        rows[index] = row
    return _block_timing_values_from_rows(rows, shift, optimization, count)


def _apply_stage_to_row_components(row_index, count, shift, optimization, *values):
    rows = _rows_from_components(*values)
    count = max(1, min(MAX_ARTIST_ROWS, int(_to_float(count, len(rows) or 1))))
    if 0 <= row_index < len(rows):
        row = list(rows[row_index])
        row += [None] * (len(TABLE_HEADERS) - len(row))
        stage = _option_key("stage", row[8], PRESET_CUSTOM)
        if stage == PRESET_CUSTOM:
            row[9] = False
        else:
            start, end, peak = _auto_stage_values(stage, shift)
            row[4], row[5], row[6] = start, end, peak
            row[9] = True
        normalized = _normalize_rows([row], 1, shift, optimization, display=True)[0]
        return [
            gr.update(value=normalized[4], interactive=True),
            gr.update(value=normalized[5], interactive=True),
            gr.update(value=normalized[6], interactive=True),
            gr.update(value=normalized[9], interactive=True),
        ]
    return [gr.update(), gr.update(), gr.update(), gr.update()]


def _apply_auto_shift_to_row_components(row_index, count, shift, optimization, *values):
    rows = _rows_from_components(*values)
    count = max(1, min(MAX_ARTIST_ROWS, int(_to_float(count, len(rows) or 1))))
    if not (0 <= row_index < min(count, len(rows))):
        return [gr.update(), gr.update(), gr.update()]
    row = list(rows[row_index])
    row += [None] * (len(TABLE_HEADERS) - len(row))
    normalized = _normalize_rows([row], 1, shift, optimization, display=True)[0]
    return [
        gr.update(value=normalized[4], interactive=True),
        gr.update(value=normalized[5], interactive=True),
        gr.update(value=normalized[6], interactive=True),
    ]


def _active_rows_from_components(count, shift, optimization, *values, display=False):
    rows = _rows_from_components(*values)
    count = max(1, min(MAX_ARTIST_ROWS, int(_to_float(count, len(rows) or 1))))
    return _normalize_rows(rows, count, shift, optimization, display=display)


def _runtime_shift_from_processing(p, fallback=3.0):
    if getattr(p, "is_hr_pass", False):
        return _clamp(_to_float(getattr(p, "hr_distilled_cfg", None), fallback), 1.0, 24.0)
    return _clamp(_to_float(getattr(p, "distilled_cfg_scale", None), fallback), 1.0, 24.0)


def _shift_sync_script(base_shift_id, hires_shift_id, is_img2img=False):
    source_ids = (
        [["img2img_distilled_cfg_scale", base_shift_id], ["img2img_distilled_cfg_scale", hires_shift_id]]
        if is_img2img
        else [["txt2img_distilled_cfg_scale", base_shift_id], ["txt2img_hr_distilled_cfg", hires_shift_id]]
    )
    return f"""
<script>
(function() {{
    const sourceIds = {json.dumps(source_ids)};

    function inputOf(id) {{
        const root = document.getElementById(id);
        return root ? root.querySelector("input, textarea") : null;
    }}

    function setHidden(hiddenId, value) {{
        const input = inputOf(hiddenId);
        if (!input || value === undefined || value === null || input.value === String(value)) {{
            return;
        }}
        input.value = value;
        input.dispatchEvent(new Event("input", {{bubbles: true}}));
        input.dispatchEvent(new Event("change", {{bubbles: true}}));
    }}

    function syncOne(sourceId, hiddenId) {{
        const source = inputOf(sourceId);
        if (!source) {{
            return false;
        }}
        setHidden(hiddenId, source.value);
        if (source.dataset.animaArtistMixerShiftSync === "1") {{
            return true;
        }}
        source.dataset.animaArtistMixerShiftSync = "1";
        const sync = function() {{ setHidden(hiddenId, source.value); }};
        source.addEventListener("input", sync);
        source.addEventListener("change", sync);
        return true;
    }}

    function syncAll() {{
        let found = false;
        for (const pair of sourceIds) {{
            found = syncOne(pair[0], pair[1]) || found;
        }}
        return found;
    }}

    if (!syncAll()) {{
        let tries = 0;
        const timer = window.setInterval(function() {{
            tries += 1;
            if (syncAll() || tries > 40) {{
                window.clearInterval(timer);
            }}
        }}, 500);
    }}
}})();
</script>
"""


def _current_settings_autosave_script(root_id, trigger_id):
    return f"""
<script>
(function() {{
    const rootId = {json.dumps(root_id)};
    const triggerId = {json.dumps(trigger_id)};
    let timer = null;

    function rootOf(id) {{
        return document.getElementById(id);
    }}

    function buttonOf(id) {{
        const root = rootOf(id);
        return root ? root.querySelector("button") : null;
    }}

    function isInside(id, element) {{
        const root = rootOf(id);
        return root && element && root.contains(element);
    }}

    function triggerSave() {{
        const button = buttonOf(triggerId);
        if (button) {{
            button.click();
        }}
    }}

    function scheduleSave(event) {{
        if (event && isInside(triggerId, event.target)) {{
            return;
        }}
        window.clearTimeout(timer);
        timer = window.setTimeout(triggerSave, 700);
    }}

    function attach() {{
        const root = rootOf(rootId);
        const button = buttonOf(triggerId);
        if (!root || !button) {{
            return false;
        }}
        if (root.dataset.animaArtistMixerAutosave === "1") {{
            return true;
        }}
        root.dataset.animaArtistMixerAutosave = "1";
        root.addEventListener("input", scheduleSave, true);
        root.addEventListener("change", scheduleSave, true);
        root.addEventListener("click", scheduleSave, true);
        return true;
    }}

    if (!attach()) {{
        let tries = 0;
        const timerId = window.setInterval(function() {{
            tries += 1;
            if (attach() || tries > 40) {{
                window.clearInterval(timerId);
            }}
        }}, 500);
    }}
}})();
</script>
"""


def _create_artist_row_controls(prefix, lang, defaults, elem_id_func):
    labels = _row_component_headers(lang)
    components = []
    bool_choices = [_bool_label(True, lang), _bool_label(False, lang)]
    curve_choices = _option_choices("curve", lang)
    stage_choices = _option_choices("stage", lang)
    for index in range(MAX_ARTIST_ROWS):
        row = defaults[index] if index < len(defaults) else _normalize_rows([_default_rows(1, 3.0, OPT_BALANCE)[0]], 1, 3.0, OPT_BALANCE)[0]
        visible = index < len(defaults)
        with gr.Row(elem_id=elem_id_func(f"{prefix}_artist_row_{index}")):
            enabled = gr.Dropdown(label=labels["enabled"], choices=bool_choices, value=row[0], allow_custom_value=False, min_width=82, visible=visible, elem_id=elem_id_func(f"{prefix}_enabled_{index}"))
            artist = gr.Textbox(label=labels["artist"], value=row[1], lines=1, min_width=190, visible=visible, elem_id=elem_id_func(f"{prefix}_artist_{index}"))
            weight = gr.Number(label=labels["weight"], value=row[2], precision=4, min_width=92, visible=visible, elem_id=elem_id_func(f"{prefix}_weight_{index}"))
            blocks = gr.Textbox(label=labels["blocks"], value=row[3], lines=1, min_width=116, visible=visible, elem_id=elem_id_func(f"{prefix}_blocks_{index}"))
            start = gr.Number(label=labels["start"], value=row[4], precision=4, min_width=86, visible=visible, elem_id=elem_id_func(f"{prefix}_start_{index}"))
            end = gr.Number(label=labels["end"], value=row[5], precision=4, min_width=86, visible=visible, elem_id=elem_id_func(f"{prefix}_end_{index}"))
            peak = gr.Number(label=labels["peak"], value=row[6], precision=4, min_width=86, visible=visible, elem_id=elem_id_func(f"{prefix}_peak_{index}"))
            curve = gr.Dropdown(label=labels["curve"], choices=curve_choices, value=row[7], allow_custom_value=False, min_width=130, visible=visible, elem_id=elem_id_func(f"{prefix}_curve_{index}"))
            stage = gr.Dropdown(label=labels["stage"], choices=stage_choices, value=row[8], allow_custom_value=False, min_width=130, visible=visible, elem_id=elem_id_func(f"{prefix}_stage_{index}"))
            auto_shift = gr.Dropdown(label=labels["auto_shift"], choices=bool_choices, value=row[9], allow_custom_value=False, min_width=112, visible=visible, elem_id=elem_id_func(f"{prefix}_auto_shift_{index}"))
        components.extend([enabled, artist, weight, blocks, start, end, peak, curve, stage, auto_shift])
    return components


def _row_outputs(row_components, offsets):
    return [row_components[index + offset] for index in range(0, len(row_components), 10) for offset in offsets]


def _register_ui_settings():
    if LANGUAGE_OPTION not in shared.opts.data_labels:
        shared.opts.add_option(
            LANGUAGE_OPTION,
            shared.OptionInfo(
                "zh",
                "Anima Artist Scheduled Mixer language / Anima 画师串调度混合语言",
                gr.Radio,
                {"choices": LANGUAGE_CHOICES},
                section=("anima-artist-scheduled-mixer", "Anima Artist Scheduled Mixer"),
                category_id="system",
            ),
        )
    if LAST_TEMPLATE_OPTION not in shared.opts.data_labels:
        shared.opts.add_option(
            LAST_TEMPLATE_OPTION,
            shared.OptionInfo(
                "",
                "Last applied Anima Artist Scheduled Mixer template / 上次应用的 Anima 画师串模板",
                gr.Textbox,
                {"visible": False},
                section=("anima-artist-scheduled-mixer", "Anima Artist Scheduled Mixer"),
                category_id="system",
            ),
        )
    if LAST_TEMPLATE_TARGET_OPTION not in shared.opts.data_labels:
        shared.opts.add_option(
            LAST_TEMPLATE_TARGET_OPTION,
            shared.OptionInfo(
                APPLY_BASE,
                "Last Anima Artist Scheduled Mixer template target / 上次模板应用目标",
                gr.Textbox,
                {"visible": False},
                section=("anima-artist-scheduled-mixer", "Anima Artist Scheduled Mixer"),
                category_id="system",
            ),
        )


script_callbacks.on_ui_settings(_register_ui_settings)


def _root_block_load(*args, **kwargs):
    try:
        from modules_forge.main_entry import Context

        if getattr(Context, "root_block", None) is not None:
            Context.root_block.load(*args, **kwargs)
    except Exception:
        logger.exception("Failed to register Anima artist mixer UI load callback")


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
    if text in {"1", "true", "yes", "y", "on", "启用", "是", "开"}:
        return True
    if text in {"0", "false", "no", "n", "off", "禁用", "否", "关"}:
        return False
    return fallback


def _parse_anchor_seed_list(value):
    seeds = []
    seen = set()
    for part in re.split(r"[\s,，;]+", str(value or "")):
        if not part:
            continue
        try:
            seed = int(part)
        except (TypeError, ValueError):
            continue
        if seed < 0:
            continue
        seed %= 2 ** 63
        if seed in seen:
            continue
        seen.add(seed)
        seeds.append(seed)
        if len(seeds) >= ANCHOR_SEEDS_MAX:
            break
    return seeds


def _normalize_advanced_settings(data=None):
    source = data.get("advanced", data) if isinstance(data, dict) else {}
    return {
        "artist_ema_alpha": _clamp(_to_float(source.get("artist_ema_alpha"), 0.0), 0.0, 0.95),
        "lowrank_k": max(1, min(MAX_ARTIST_ROWS, int(_to_float(source.get("lowrank_k"), 1)))),
        "artist_static_capture": _to_bool(source.get("artist_static_capture"), False),
        "static_capture_k": max(1, min(STATIC_CAPTURE_K_MAX, int(_to_float(source.get("static_capture_k"), STATIC_CAPTURE_K_DEFAULT)))),
        "artist_anchor_q": _to_bool(source.get("artist_anchor_q"), False),
        "anchor_seed_list": str(source.get("anchor_seed_list") or "").strip(),
        "anchor_seeds_count": max(1, min(ANCHOR_SEEDS_MAX, int(_to_float(source.get("anchor_seeds_count"), 1)))),
        "anchor_user_blend": _clamp(_to_float(source.get("anchor_user_blend"), 0.0), 0.0, 1.0),
        "anchor_deep_layer_threshold": max(-1, min(64, int(_to_float(source.get("anchor_deep_layer_threshold"), -1)))),
        "stabilizer_end_percent": _clamp(_to_float(source.get("stabilizer_end_percent"), 1.0), 0.0, 1.0),
        "style_balance": _clamp(_to_float(source.get("style_balance"), 0.0), 0.0, 1.0),
        "structure_preserve": _clamp(_to_float(source.get("structure_preserve"), 0.0), 0.0, 1.0),
        "delta_norm_cap": _clamp(_to_float(source.get("delta_norm_cap"), 0.0), 0.0, 4.0),
    }


def _advanced_settings_from_values(values):
    raw = dict(zip(ADVANCED_SETTING_KEYS, list(values)[: len(ADVANCED_SETTING_KEYS)]))
    return _normalize_advanced_settings(raw)


def _advanced_values(settings=None):
    normalized = _normalize_advanced_settings(settings)
    return [normalized[key] for key in ADVANCED_SETTING_KEYS]


def _advanced_updates(settings=None):
    return [gr.update(value=value, interactive=True) for value in _advanced_values(settings)]


def _format_float(value):
    text = f"{_to_float(value):.6f}".rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


def _split_artist_chain(chain):
    source = str(chain or "").replace("，", ",").replace("\r", "\n").replace("\n", ",")
    return [part.strip() for part in source.split(",") if part.strip()]


def _parse_inline_weight(text):
    value = str(text or "").strip()
    if not value:
        return "", 1.0
    number = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))"
    wrapped = re.fullmatch(rf"[\(\[](.+?)[：:]\s*{number}[\)\]]", value)
    if wrapped:
        return wrapped.group(1).strip(), _to_float(wrapped.group(2), 1.0)
    if not (value.startswith("<") and value.endswith(">")):
        trailing = re.fullmatch(rf"(.+?)[：:]\s*{number}", value)
        if trailing:
            return trailing.group(1).strip(), _to_float(trailing.group(2), 1.0)
    return value, 1.0


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
    stage = _option_key("stage", stage, PRESET_CUSTOM)
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
    curve = _option_key("curve", curve, CURVE_SMOOTH)
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


def _default_rows(count=DEFAULT_ARTIST_ROWS, shift=3.0, optimization=OPT_BALANCE):
    optimization = _option_key("optimization", optimization, OPT_BALANCE)
    rows = []
    for i in range(max(0, int(count))):
        blocks = _default_blocks_for_optimization(optimization)
        rows.append([True, "", 1.0, blocks, 0.0, 1.0, 0.5, CURVE_HOLD, PRESET_CUSTOM, False])
    return rows


def _coerce_table(value):
    if value is None:
        return []
    if hasattr(value, "to_dict"):
        try:
            records = value.to_dict("records")
            rows = []
            for record in records:
                keyed = {_header_key(k): v for k, v in record.items()}
                rows.append([keyed.get(h) for h in TABLE_HEADERS])
            return rows
        except Exception:
            pass
    if isinstance(value, dict) and "data" in value:
        value = value.get("data")
    rows = []
    for row in value if isinstance(value, (list, tuple)) else []:
        if isinstance(row, dict):
            keyed = {_header_key(k): v for k, v in row.items()}
            rows.append([keyed.get(h) for h in TABLE_HEADERS])
        elif isinstance(row, (list, tuple)):
            padded = list(row)
            if len(padded) == len(TABLE_HEADERS) + 1:
                # Older templates had a per-row Shift column before Auto Shift.
                padded.pop(9)
            padded = padded[: len(TABLE_HEADERS)]
            padded += [None] * (len(TABLE_HEADERS) - len(padded))
            rows.append(padded)
    return rows


def _normalize_rows(value, row_count=None, shift=3.0, optimization=OPT_BALANCE, display=True):
    optimization = _option_key("optimization", optimization, OPT_BALANCE)
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
        curve = _option_key("curve", raw[7], CURVE_SMOOTH)
        stage = _option_key("stage", raw[8], PRESET_CUSTOM)
        auto = _to_bool(raw[9], True)
        if auto and stage != PRESET_CUSTOM:
            start, end, peak = _auto_stage_values(stage, shift)
        enabled_value = _bool_label(enabled) if display else enabled
        curve_value = _option_label("curve", curve) if display else curve
        stage_value = _option_label("stage", stage) if display else stage
        auto_value = _bool_label(auto) if display else auto
        normalized.append(
            [
                enabled_value,
                artist,
                weight,
                blocks,
                round(start, 4),
                round(end, 4),
                round(peak, 4),
                curve_value,
                stage_value,
                auto_value,
            ]
        )
    return normalized


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


def _resolve_template_name(value=None, choices=None):
    choices = choices if choices is not None else _template_choices()
    value = str(value or "").strip()
    if value in choices:
        return value
    return choices[0] if choices else None


def _template_dropdown_update(value=None):
    choices = _template_choices()
    value = _resolve_template_name(value, choices)
    return gr.update(choices=choices, value=value)


def _save_template_record(
    name,
    row_key,
    rows,
    row_count,
    shift,
    disable_hires_mixing,
    global_strength,
    optimization,
    combine_mode,
    fusion_mode,
    apply_uncond,
    enable_cache,
    advanced_settings,
    status_key,
):
    name = str(name or "").strip()
    if not name:
        return _template_dropdown_update(), _t("empty_template_name")
    optimization_key = _option_key("optimization", optimization, OPT_BALANCE)
    combine_key = _option_key("combine", combine_mode, COMBINE_OUTPUT_AVG)
    fusion_key = _option_key("fusion", fusion_mode, FUSION_INTERPOLATE)
    row_count = max(1, min(MAX_ARTIST_ROWS, int(_to_float(row_count, len(_coerce_table(rows)) or 1))))
    shift = _to_float(shift, 3.0)
    data = _template_data()
    data[name] = {
        row_key: _normalize_rows(rows, row_count, shift, optimization_key, display=False),
        "hires_independent": row_key == "hires_rows",
        "disable_hires_mixing": bool(disable_hires_mixing),
        "global_strength": _to_float(global_strength, 0.7),
        "optimization": optimization_key,
        "combine_mode": combine_key,
        "fusion_mode": fusion_key,
        "apply_uncond": bool(apply_uncond),
        "enable_cache": bool(enable_cache),
        "advanced": _normalize_advanced_settings(advanced_settings),
    }
    _save_template_data(data)
    return _template_dropdown_update(name), _t(status_key).format(name=name)


def _save_base_template_ui(name, base_row_count, base_shift, disable_hires_mixing, global_strength, optimization, combine_mode, fusion_mode, apply_uncond, enable_cache, *base_values):
    advanced_settings = _advanced_settings_from_values(base_values)
    base_values = base_values[len(ADVANCED_SETTING_KEYS) :]
    base_rows = _rows_from_components(*base_values)
    return _save_template_record(
        name,
        "base_rows",
        base_rows,
        base_row_count,
        base_shift,
        disable_hires_mixing,
        global_strength,
        optimization,
        combine_mode,
        fusion_mode,
        apply_uncond,
        enable_cache,
        advanced_settings,
        "saved_base_template",
    )


def _save_hires_template_ui(name, hires_independent, hires_row_count, hires_shift, disable_hires_mixing, global_strength, optimization, combine_mode, fusion_mode, apply_uncond, enable_cache, *hires_values):
    if not _to_bool(hires_independent, False):
        return _template_dropdown_update(), _t("hires_template_disabled")
    advanced_settings = _advanced_settings_from_values(hires_values)
    hires_values = hires_values[len(ADVANCED_SETTING_KEYS) :]
    hires_rows = _rows_from_components(*hires_values)
    return _save_template_record(
        name,
        "hires_rows",
        hires_rows,
        hires_row_count,
        hires_shift,
        disable_hires_mixing,
        global_strength,
        optimization,
        combine_mode,
        fusion_mode,
        apply_uncond,
        enable_cache,
        advanced_settings,
        "saved_hires_template",
    )


def _rename_template_ui(old_name, new_name):
    old_name = str(old_name or "").strip()
    new_name = str(new_name or "").strip()
    data = _template_data()
    if not old_name or old_name not in data:
        return _template_dropdown_update(), _t("no_template_selected")
    if not new_name:
        return _template_dropdown_update(old_name), _t("new_template_name_empty")
    data[new_name] = data.pop(old_name)
    _save_template_data(data)
    return _template_dropdown_update(new_name), _t("renamed_template").format(name=new_name)


def _delete_template_ui(name):
    name = str(name or "").strip()
    data = _template_data()
    if name in data:
        data.pop(name)
        _save_template_data(data)
        return _template_dropdown_update(), _t("deleted_template").format(name=name)
    return _template_dropdown_update(), _t("no_template_deleted")


def _builtin_default_ui_state(lang):
    enable = False
    base_shift = 3.0
    hires_shift = 3.0
    base_count = DEFAULT_ARTIST_ROWS
    hires_count = DEFAULT_ARTIST_ROWS
    optimization_key = OPT_BALANCE
    combine_key = COMBINE_OUTPUT_AVG
    fusion_key = FUSION_INTERPOLATE
    apply_target_key = APPLY_BASE
    global_strength = 0.7
    hires_independent = False
    disable_hires_mixing = False
    apply_uncond = False
    enable_cache = True
    advanced = _normalize_advanced_settings()
    template_name = None

    base_rows = _normalize_rows(_default_rows(base_count, base_shift, optimization_key), base_count, base_shift, optimization_key)
    hires_rows = _normalize_rows(_default_rows(hires_count, hires_shift, optimization_key), hires_count, hires_shift, optimization_key)

    return {
        "enable": enable,
        "template_name": template_name,
        "template_target": _option_label("apply_target", apply_target_key, lang),
        "base_count": base_count,
        "hires_count": hires_count,
        "base_rows": base_rows,
        "hires_rows": hires_rows,
        "hires_independent": hires_independent,
        "disable_hires_mixing": disable_hires_mixing,
        "global_strength": global_strength,
        "optimization": _option_label("optimization", optimization_key, lang),
        "combine": _option_label("combine", combine_key, lang),
        "fusion": _option_label("fusion", fusion_key, lang),
        "apply_uncond": apply_uncond,
        "enable_cache": enable_cache,
        "advanced": advanced,
    }


def _current_settings_data():
    if not CURRENT_SETTINGS_FILE.exists():
        return None
    try:
        with CURRENT_SETTINGS_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        logger.exception("Failed to load Anima artist mixer current settings")
        return None


def _save_current_settings_data(data):
    try:
        CURRENT_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with CURRENT_SETTINGS_FILE.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("Failed to save Anima artist mixer current settings")


def _delete_current_settings_data():
    try:
        if CURRENT_SETTINGS_FILE.exists():
            CURRENT_SETTINGS_FILE.unlink()
    except Exception:
        logger.exception("Failed to delete Anima artist mixer current settings")


def _state_from_current_settings(data, lang):
    if not isinstance(data, dict):
        return None
    optimization_key = _option_key("optimization", data.get("optimization"), OPT_BALANCE)
    combine_key = _option_key("combine", data.get("combine_mode"), COMBINE_OUTPUT_AVG)
    fusion_key = _option_key("fusion", data.get("fusion_mode"), FUSION_INTERPOLATE)
    apply_target_key = _option_key("apply_target", data.get("template_target"), APPLY_BASE)
    base_shift = _to_float(data.get("base_shift"), 3.0)
    hires_shift = _to_float(data.get("hires_shift"), 3.0)
    base_count = max(1, min(MAX_ARTIST_ROWS, int(_to_float(data.get("base_count"), DEFAULT_ARTIST_ROWS))))
    hires_count = max(1, min(MAX_ARTIST_ROWS, int(_to_float(data.get("hires_count"), DEFAULT_ARTIST_ROWS))))
    base_rows = _normalize_rows(data.get("base_rows"), base_count, base_shift, optimization_key)
    hires_rows = _normalize_rows(data.get("hires_rows"), hires_count, hires_shift, optimization_key)
    template_name = str(data.get("template_name") or "").strip()
    if template_name and template_name not in _template_data():
        template_name = None
    return {
        "enable": bool(data.get("enable", False)),
        "template_name": template_name,
        "template_target": _option_label("apply_target", apply_target_key, lang),
        "base_count": base_count,
        "hires_count": hires_count,
        "base_rows": base_rows,
        "hires_rows": hires_rows,
        "hires_independent": bool(data.get("hires_independent", False)),
        "disable_hires_mixing": bool(data.get("disable_hires_mixing", False)),
        "global_strength": _to_float(data.get("global_strength"), 0.7),
        "optimization": _option_label("optimization", optimization_key, lang),
        "combine": _option_label("combine", combine_key, lang),
        "fusion": _option_label("fusion", fusion_key, lang),
        "apply_uncond": bool(data.get("apply_uncond", False)),
        "enable_cache": bool(data.get("enable_cache", True)),
        "advanced": _normalize_advanced_settings(data),
    }


def _initial_ui_defaults(lang):
    current = _state_from_current_settings(_current_settings_data(), lang)
    return current if current is not None else _builtin_default_ui_state(lang)


def _current_settings_payload(
    enable,
    template_name,
    template_target,
    base_row_count,
    hires_row_count,
    hires_independent,
    disable_hires_mixing,
    base_shift,
    hires_shift,
    global_strength,
    optimization,
    combine_mode,
    fusion_mode,
    apply_uncond,
    enable_cache,
    *values,
):
    advanced_values = values[: len(ADVANCED_SETTING_KEYS)]
    component_values = values[len(ADVANCED_SETTING_KEYS) :]
    advanced_settings = _advanced_settings_from_values(advanced_values)
    optimization_key = _option_key("optimization", optimization, OPT_BALANCE)
    base_values = component_values[: MAX_ARTIST_ROWS * 10]
    hires_values = component_values[MAX_ARTIST_ROWS * 10 : MAX_ARTIST_ROWS * 20]
    base_shift = _to_float(base_shift, 3.0)
    hires_shift = _to_float(hires_shift, 3.0)
    base_count = max(1, min(MAX_ARTIST_ROWS, int(_to_float(base_row_count, DEFAULT_ARTIST_ROWS))))
    hires_count = max(1, min(MAX_ARTIST_ROWS, int(_to_float(hires_row_count, DEFAULT_ARTIST_ROWS))))
    return {
        "version": 2,
        "enable": bool(enable),
        "template_name": str(template_name or "").strip(),
        "template_target": _option_key("apply_target", template_target, APPLY_BASE),
        "base_count": base_count,
        "hires_count": hires_count,
        "base_shift": base_shift,
        "hires_shift": hires_shift,
        "base_rows": _normalize_rows(_rows_from_components(*base_values), base_count, base_shift, optimization_key, display=False),
        "hires_rows": _normalize_rows(_rows_from_components(*hires_values), hires_count, hires_shift, optimization_key, display=False),
        "hires_independent": bool(hires_independent),
        "disable_hires_mixing": bool(disable_hires_mixing),
        "global_strength": _to_float(global_strength, 0.7),
        "optimization": optimization_key,
        "combine_mode": _option_key("combine", combine_mode, COMBINE_OUTPUT_AVG),
        "fusion_mode": _option_key("fusion", fusion_mode, FUSION_INTERPOLATE),
        "apply_uncond": bool(apply_uncond),
        "enable_cache": bool(enable_cache),
        "advanced": advanced_settings,
    }


def _save_current_settings_ui(*values):
    _save_current_settings_data(_current_settings_payload(*values))
    return ""


def _save_runtime_current_settings(
    enable,
    base_row_count,
    hires_row_count,
    hires_independent,
    disable_hires_mixing,
    runtime_base_shift,
    runtime_hires_shift,
    global_strength,
    optimization,
    combine_mode,
    fusion_mode,
    apply_uncond,
    enable_cache,
    *values,
):
    try:
        _save_current_settings_data(
            _current_settings_payload(
                enable,
                "",
                APPLY_BASE,
                base_row_count,
                hires_row_count,
                hires_independent,
                disable_hires_mixing,
                runtime_base_shift,
                runtime_hires_shift,
                global_strength,
                optimization,
                combine_mode,
                fusion_mode,
                apply_uncond,
                enable_cache,
                *values,
            )
        )
    except Exception:
        logger.exception("Failed to save Anima artist mixer runtime settings")


def _do_not_save_to_ui_config(*components):
    for component in components:
        setattr(component, "do_not_save_to_config", True)


def _internal_event_kwargs():
    return {"api_name": False, "show_api": False}


def _default_state_updates(lang=None, status_text=None):
    lang = lang or _language()
    state = _builtin_default_ui_state(lang)
    base_updates = _component_values_from_rows(state["base_rows"], 3.0, state["optimization"], state["base_count"])
    hires_updates = _component_values_from_rows(state["hires_rows"], 3.0, state["optimization"], state["hires_count"])
    return (
        gr.update(value=state["enable"], interactive=True),
        _template_dropdown_update(state["template_name"]),
        gr.update(value=state["template_target"]),
        gr.update(value=state["base_count"], interactive=True),
        gr.update(value=state["hires_count"], interactive=True),
        gr.update(value=state["hires_independent"], interactive=True),
        gr.update(value=state["disable_hires_mixing"], interactive=True),
        gr.update(value=3.0),
        gr.update(value=3.0),
        gr.update(value=state["global_strength"], interactive=True),
        gr.update(value=state["optimization"], interactive=True),
        gr.update(value=state["combine"], interactive=True),
        gr.update(value=state["fusion"], interactive=True),
        gr.update(value=state["apply_uncond"], interactive=True),
        gr.update(value=state["enable_cache"], interactive=True),
        *_advanced_updates(state["advanced"]),
        _t("defaults_restored") if status_text is None else status_text,
        *base_updates,
        *hires_updates,
    )


def _reset_all_defaults_ui():
    _delete_current_settings_data()
    return _default_state_updates()


def _apply_template_updates(name, target, base_row_count, hires_row_count, current_base_shift, current_hires_shift, current_optimization, component_values, remember=False, status_text=None):
    row_values = component_values[len(ADVANCED_SETTING_KEYS) :]
    base_values = row_values[: MAX_ARTIST_ROWS * 10]
    hires_values = row_values[MAX_ARTIST_ROWS * 10 : MAX_ARTIST_ROWS * 20]
    base_current = _rows_from_components(*base_values)
    hires_current = _rows_from_components(*hires_values)
    no_selection = (
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
        gr.update(),
        *[gr.update() for _ in ADVANCED_SETTING_KEYS],
        _t("no_template_selected"),
        *[gr.update() for _ in range(MAX_ARTIST_ROWS * 20)],
    )
    data = _template_data()
    name = _resolve_template_name(name, sorted(data.keys()))
    tpl = data.get(name)
    if not isinstance(tpl, dict):
        return no_selection
    optimization_key = _option_key("optimization", tpl.get("optimization"), OPT_BALANCE)
    base_shift = _to_float(current_base_shift, _to_float(tpl.get("base_shift", 3.0), 3.0))
    hires_shift = _to_float(current_hires_shift, _to_float(tpl.get("hires_shift", base_shift), base_shift))
    has_base_rows = bool(tpl.get("base_rows"))
    has_hires_rows = bool(tpl.get("hires_rows"))
    fallback_optimization = _option_key("optimization", current_optimization, OPT_BALANCE)
    base_tpl = _normalize_rows(
        tpl.get("base_rows") or tpl.get("hires_rows") or base_current,
        None,
        base_shift,
        optimization_key if has_base_rows or has_hires_rows else fallback_optimization,
    )
    hires_tpl = _normalize_rows(
        tpl.get("hires_rows") or tpl.get("base_rows") or hires_current,
        None,
        hires_shift,
        optimization_key if has_hires_rows or has_base_rows else fallback_optimization,
    )
    target = _option_key("apply_target", target, APPLY_BASE)
    if remember:
        _remember_template(name, target)
    base_count_value = max(1, len(base_tpl)) if target in {APPLY_BASE, APPLY_BOTH} else max(1, int(_to_float(base_row_count, 1)))
    hires_count_value = max(1, len(hires_tpl)) if target in {APPLY_HIRES, APPLY_BOTH} else max(1, int(_to_float(hires_row_count, 1)))
    base_updates = [gr.update() for _ in range(MAX_ARTIST_ROWS * 10)]
    hires_updates = [gr.update() for _ in range(MAX_ARTIST_ROWS * 10)]
    if target in {APPLY_BASE, APPLY_BOTH}:
        base_updates = _component_values_from_rows(base_tpl, base_shift, optimization_key, base_count_value)
    if target in {APPLY_HIRES, APPLY_BOTH}:
        hires_updates = _component_values_from_rows(hires_tpl, hires_shift, optimization_key, hires_count_value)
    return (
        gr.update(value=base_count_value, interactive=True),
        gr.update(value=hires_count_value, interactive=True),
        gr.update(value=tpl.get("hires_independent", False), interactive=True),
        gr.update(value=tpl.get("disable_hires_mixing", False), interactive=True),
        gr.update(),
        gr.update(),
        gr.update(value=tpl.get("global_strength", 0.7), interactive=True),
        gr.update(value=_option_label("optimization", optimization_key), interactive=True),
        gr.update(value=_option_label("combine", tpl.get("combine_mode", COMBINE_OUTPUT_AVG)), interactive=True),
        gr.update(value=_option_label("fusion", tpl.get("fusion_mode", FUSION_INTERPOLATE)), interactive=True),
        gr.update(value=tpl.get("apply_uncond", False), interactive=True),
        gr.update(value=tpl.get("enable_cache", True), interactive=True),
        *_advanced_updates(tpl),
        status_text if status_text is not None else _t("applied_template").format(name=name, target=_option_label("apply_target", target)),
        *base_updates,
        *hires_updates,
    )


def _apply_template_ui(name, target, base_row_count, hires_row_count, current_base_shift, current_hires_shift, current_optimization, *component_values):
    return _apply_template_updates(
        name,
        target,
        base_row_count,
        hires_row_count,
        current_base_shift,
        current_hires_shift,
        current_optimization,
        component_values,
        remember=True,
    )


def _blank_template_updates(status_text=""):
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
        gr.update(),
        *[gr.update() for _ in ADVANCED_SETTING_KEYS],
        status_text,
        *[gr.update() for _ in range(MAX_ARTIST_ROWS * 20)],
    )


def _apply_last_template_on_load_ui(base_row_count, hires_row_count, current_base_shift, current_hires_shift, current_optimization, *component_values):
    name = _last_template_name()
    target = _last_template_target()
    if not name:
        return (
            gr.update(),
            gr.update(),
            *_blank_template_updates(""),
        )
    return (
        gr.update(value=name),
        gr.update(value=_option_label("apply_target", target)),
        *_apply_template_updates(
            name,
            target,
            base_row_count,
            hires_row_count,
            current_base_shift,
            current_hires_shift,
            current_optimization,
            component_values,
            status_text="",
        ),
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
    artist_ema_alpha: float = 0.0
    lowrank_k: int = 1
    artist_static_capture: bool = False
    static_capture_k: int = STATIC_CAPTURE_K_DEFAULT
    artist_anchor_q: bool = False
    anchor_seed_list: list[int] = field(default_factory=list)
    anchor_seeds_count: int = 1
    anchor_user_blend: float = 0.0
    anchor_deep_layer_threshold: int = -1
    stabilizer_end_percent: float = 1.0
    style_balance: float = 0.0
    structure_preserve: float = 0.0
    delta_norm_cap: float = 0.0
    dm_ref: object | None = None
    artists: list[ArtistRuntime] = field(default_factory=list)
    progress_warning: bool = False
    batched_disabled: bool = False
    patched_blocks: list[int] = field(default_factory=list)
    dispatch_calls: int = 0
    active_calls: int = 0
    fallback_calls: int = 0
    wrapper_checks: int = 0
    diff_probe: tuple[float, float] | None = None
    mask_probe: tuple[int, int] | None = None
    superseded: bool = False
    current_progress: float | None = None
    current_sigma: float | None = None
    current_layer: int = -1
    ema_cache: dict = field(default_factory=dict)
    static_cache: dict = field(default_factory=dict)
    anchor_cache: dict = field(default_factory=dict)
    anchor_capture: dict = field(default_factory=dict)
    anchor_cache_key: tuple | None = None
    anchor_failed: bool = False
    in_anchor_run: bool = False
    warned_static_no_progress: bool = False
    warned_ema_no_progress: bool = False
    warned_lowrank: bool = False
    warned_anchor_ok: bool = False

    def target_blocks(self):
        blocks = set()
        for artist in self.artists:
            blocks.update(artist.blocks)
        return blocks


_PATCHED_MODULES = []
_PATCHED_MODEL_WRAPPERS = []
_ACTIVE_STATE = None
_RUN_STATES = []


def _unpatch_cross_attn():
    global _PATCHED_MODULES, _PATCHED_MODEL_WRAPPERS
    for module, original in reversed(_PATCHED_MODULES):
        try:
            if getattr(module.forward, "_anima_artist_mixer_wrapper", False):
                module.forward = original
        except Exception:
            logger.exception("Failed to restore Anima artist mixer cross-attn wrapper")
    _PATCHED_MODULES = []
    for unet, had_wrapper, original_wrapper in reversed(_PATCHED_MODEL_WRAPPERS):
        try:
            current = unet.model_options.get("model_function_wrapper")
            if getattr(current, "_anima_artist_mixer_model_wrapper", False):
                if had_wrapper:
                    unet.model_options["model_function_wrapper"] = original_wrapper
                else:
                    unet.model_options.pop("model_function_wrapper", None)
        except Exception:
            logger.exception("Failed to restore Anima artist mixer model wrapper")
    _PATCHED_MODEL_WRAPPERS = []


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
    patched = []
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
        patched.append(idx)
    state.patched_blocks = patched
    if patched:
        logger.info("Anima artist mixer patched %d cross-attn blocks: %s", len(patched), _summarize_blocks(patched))
    else:
        logger.warning("Anima artist mixer found no target cross-attn blocks to patch.")


def _summarize_blocks(blocks):
    values = sorted(set(int(x) for x in blocks))
    if not values:
        return "none"
    ranges = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = value
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def _condition_row_index(row_count, cond_or_uncond, condition_index):
    if cond_or_uncond and row_count % len(cond_or_uncond) == 0:
        return condition_index * (row_count // len(cond_or_uncond))
    return condition_index if condition_index < row_count else 0


def _run_anchor_q(state, apply_model, args):
    c_dict = dict(args.get("c", {}) or {})
    base_context = c_dict.get("c_crossattn")
    user_x = args.get("input")
    timestep = args.get("timestep")
    if not torch.is_tensor(base_context) or user_x is None or timestep is None:
        return

    transformer_options = dict(c_dict.get("transformer_options", {}) or {})
    cond_or_uncond = args.get("cond_or_uncond") or transformer_options.get("cond_or_uncond")
    cond_index = 0
    if cond_or_uncond is not None:
        markers = list(cond_or_uncond)
        if 0 not in markers:
            return
        cond_index = markers.index(0)
    if base_context.shape[0] > 1:
        row = _condition_row_index(base_context.shape[0], cond_or_uncond, cond_index)
        base_context = base_context[row : row + 1]
    batch_size = user_x.shape[0]
    if base_context.shape[0] != batch_size:
        base_context = base_context[:1].expand(batch_size, *base_context.shape[1:])
    base_context = base_context.contiguous().to(device=user_x.device, dtype=user_x.dtype)

    seeds = state.anchor_seed_list[:ANCHOR_SEEDS_MAX]
    if not seeds:
        seeds = list(ANCHOR_SEEDS_POOL[: state.anchor_seeds_count])
    try:
        sigma_key = round(float(timestep.flatten()[0].item()), 4)
    except Exception:
        sigma_key = None
    cache_key = (tuple(user_x.shape), _tensor_fingerprint(base_context), sigma_key, tuple(seeds))
    if state.anchor_cache_key == cache_key and state.anchor_cache:
        return

    safe_options = dict(transformer_options)
    for key in ("cond_or_uncond", "cond_mark", "cond_indices", "uncond_indices", "patches"):
        safe_options.pop(key, None)
    anchor_kwargs = dict(c_dict)
    anchor_kwargs["c_crossattn"] = base_context
    anchor_kwargs["transformer_options"] = safe_options
    accumulator = {}
    state.in_anchor_run = True
    try:
        with torch.no_grad():
            for seed in seeds:
                generator = torch.Generator(device=user_x.device)
                generator.manual_seed(seed)
                anchor_x = torch.randn(user_x.shape, generator=generator, device=user_x.device, dtype=user_x.dtype)
                state.anchor_capture = {}
                apply_model(anchor_x, timestep, **anchor_kwargs)
                for layer_idx, hidden in state.anchor_capture.items():
                    value = hidden.to(torch.float32)
                    accumulator[layer_idx] = value if layer_idx not in accumulator else accumulator[layer_idx] + value
        inv = 1.0 / max(1, len(seeds))
        state.anchor_cache = {layer_idx: (value * inv).to(user_x.dtype).detach() for layer_idx, value in accumulator.items()}
    except Exception as exc:
        if _should_reraise(exc):
            raise
        logger.warning("Anima artist mixer Anchor Q pre-run failed and was disabled: %s", exc)
        state.anchor_cache = {}
        state.anchor_failed = True
    finally:
        state.anchor_capture = {}
        state.in_anchor_run = False

    if state.anchor_cache:
        state.anchor_cache_key = cache_key
        if not state.warned_anchor_ok:
            logger.info("Anima artist mixer Anchor Q captured %d cross-attn layers.", len(state.anchor_cache))
            state.warned_anchor_ok = True
    elif not state.anchor_failed:
        state.anchor_failed = True
        logger.warning("Anima artist mixer Anchor Q captured no layers and was disabled for this run.")


def _install_model_wrapper(unet, dm, state):
    global _PATCHED_MODEL_WRAPPERS
    options = getattr(unet, "model_options", None)
    if not isinstance(options, dict):
        return
    existing = options.get("model_function_wrapper")
    if getattr(existing, "_anima_artist_mixer_model_wrapper", False):
        return
    had_wrapper = "model_function_wrapper" in options

    def model_wrapper(apply_model, args):
        state.wrapper_checks += 1
        if not _PATCHED_MODULES:
            _install_cross_attn_patch_no_unpatch(dm, state)
        c_dict = args.get("c", {}) or {}
        transformer_options = c_dict.get("transformer_options", {}) or {}
        progress = _current_progress(transformer_options, state)
        timestep = args.get("timestep")
        if timestep is not None:
            try:
                state.current_sigma = float(timestep.flatten()[0].item())
            except Exception:
                pass
        if state.artist_anchor_q and not state.anchor_failed and not state.anchor_cache and _stabilizer_active(state, progress):
            _run_anchor_q(state, apply_model, args)
        if existing is not None:
            return existing(apply_model, args)
        return apply_model(args.get("input"), args.get("timestep"), **args.get("c", {}))

    model_wrapper._anima_artist_mixer_model_wrapper = True
    options["model_function_wrapper"] = model_wrapper
    _PATCHED_MODEL_WRAPPERS.append((unet, had_wrapper, existing))


def _install_cross_attn_patch_no_unpatch(dm, state):
    if _PATCHED_MODULES:
        return
    target_blocks = state.target_blocks()
    patched = []
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
        patched.append(idx)
    state.patched_blocks = patched
    if patched:
        logger.info("Anima artist mixer re-patched %d cross-attn blocks at model call: %s", len(patched), _summarize_blocks(patched))


def _broadcast_batch(tensor, batch_size):
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch_size, -1, -1)
    if batch_size % tensor.shape[0] == 0:
        return tensor.repeat(batch_size // tensor.shape[0], 1, 1)
    return tensor[:1].expand(batch_size, -1, -1)


def _should_reraise(error):
    if isinstance(error, MemoryError):
        return True
    for owner in (torch, getattr(torch, "cuda", None)):
        error_type = getattr(owner, "OutOfMemoryError", None) if owner is not None else None
        if error_type is not None and isinstance(error, error_type):
            return True
    return False


def _tensor_fingerprint(tensor):
    if tensor is None or not torch.is_tensor(tensor):
        return None
    try:
        flat = tensor.detach().reshape(-1)
        step = max(1, flat.numel() // 1024)
        digest = flat[::step].to(torch.float32).sum().item()
        return tuple(tensor.shape), str(tensor.dtype), round(digest, 3)
    except Exception:
        return tuple(tensor.shape), str(tensor.dtype), None


def _active_signature(active):
    return tuple((artist.name, round(float(weight), 6)) for artist, weight in active)


def _stabilizer_active(state, progress):
    if progress is None:
        return False
    return progress <= state.stabilizer_end_percent + 1e-6


def _project_perpendicular(delta, base):
    delta_f32 = delta.to(torch.float32)
    base_f32 = base.to(torch.float32)
    base_norm_sq = (base_f32 * base_f32).sum(dim=-1, keepdim=True).clamp(min=1e-8)
    projection = (delta_f32 * base_f32).sum(dim=-1, keepdim=True) / base_norm_sq
    return (delta_f32 - projection * base_f32).to(delta.dtype)


def _limit_delta_norm(delta, base, cap_ratio):
    cap_ratio = float(cap_ratio)
    if cap_ratio <= 0.0:
        return delta
    base_norm = torch.linalg.vector_norm(base.to(torch.float32), dim=-1, keepdim=True)
    delta_norm = torch.linalg.vector_norm(delta.to(torch.float32), dim=-1, keepdim=True)
    limit = (base_norm * cap_ratio).clamp(min=1e-8)
    scale = (limit / delta_norm.clamp(min=1e-8)).clamp(max=1.0).to(delta.dtype)
    return delta * scale


def _lowrank_rows(rows, rank):
    if rank >= rows.shape[0]:
        return rows
    gram = rows @ rows.transpose(0, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    indices = torch.argsort(eigenvalues, descending=True)[:rank]
    basis = eigenvectors[:, indices]
    return basis @ (basis.transpose(0, 1) @ rows)


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
        progress = _clamp(idx / max(1, sigmas.numel() - 1), 0.0, 1.0)
        state.current_progress = progress
        state.current_sigma = cur_value
        return progress
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


def _to_context_like(tensor, context):
    artist = _broadcast_batch(_to_context(tensor, context), context.shape[0])
    if artist.shape[-1] != context.shape[-1]:
        raise RuntimeError(f"Artist context dim {artist.shape[-1]} does not match base context dim {context.shape[-1]}")
    while artist.dim() < context.dim():
        artist = artist.unsqueeze(1)
    if artist.dim() != context.dim():
        raise RuntimeError(f"Artist context rank {artist.dim()} does not match base context rank {context.dim()}")
    expand_shape = list(artist.shape)
    for dim in range(1, context.dim() - 2):
        if artist.shape[dim] == context.shape[dim]:
            continue
        if artist.shape[dim] != 1:
            raise RuntimeError(f"Cannot align artist context shape {tuple(artist.shape)} to base context shape {tuple(context.shape)}")
        expand_shape[dim] = context.shape[dim]
    return artist.expand(*expand_shape)


def _context_token_dim(context):
    return max(1, context.dim() - 2)


def _concat_contexts(base_context, artist_context):
    return torch.cat([base_context, artist_context], dim=_context_token_dim(base_context))


def _artist_outputs_batched(original_forward, x, context, rope_emb, transformer_options, artists, fusion_mode):
    batch_size = context.shape[0]
    contexts = []
    for artist, _ in artists:
        artist_context = _to_context_like(artist.cond, context)
        contexts.append(_concat_contexts(context, artist_context) if fusion_mode in BASE_CONTEXT_FUSIONS else artist_context)
    lengths = {item.shape[_context_token_dim(item)] for item in contexts}
    if len(lengths) > 1:
        raise RuntimeError(f"Cannot batch artist contexts with different token lengths: {lengths}")
    count = len(contexts)
    x_rep = x.repeat(count, *([1] * (x.dim() - 1)))
    context_rep = torch.cat(contexts, dim=0)
    rope_rep = rope_emb
    if torch.is_tensor(rope_emb) and rope_emb.dim() > 0 and rope_emb.shape[0] == batch_size:
        rope_rep = rope_emb.repeat(count, *([1] * (rope_emb.dim() - 1)))
    options = dict(transformer_options) if isinstance(transformer_options, dict) else {}
    cond_or_uncond = options.get("cond_or_uncond")
    if cond_or_uncond is not None:
        options["cond_or_uncond"] = list(cond_or_uncond) * count
    output = original_forward(x_rep, context=context_rep, rope_emb=rope_rep, transformer_options=options)
    output = output.view(count, batch_size, *output.shape[1:])
    return [output[index] for index in range(count)]


def _anchor_query_x(state, layer_idx, x, progress):
    if not state.artist_anchor_q or state.anchor_failed or not _stabilizer_active(state, progress):
        return x
    if state.anchor_deep_layer_threshold >= 0 and layer_idx >= state.anchor_deep_layer_threshold:
        return x
    anchor = state.anchor_cache.get(layer_idx)
    if anchor is None or anchor.shape[1:] != x.shape[1:]:
        return x
    if anchor.shape[0] != x.shape[0]:
        if x.shape[0] % anchor.shape[0] == 0:
            anchor = anchor.repeat(x.shape[0] // anchor.shape[0], *([1] * (anchor.dim() - 1)))
        elif anchor.shape[0] % x.shape[0] == 0:
            anchor = anchor[: x.shape[0]]
        else:
            return x
    anchor = anchor.to(device=x.device, dtype=x.dtype)
    blend = state.anchor_user_blend
    return x * blend + anchor * (1.0 - blend) if blend > 0.0 else anchor


def _collect_artist_outputs(original_forward, layer_idx, state, x, context, rope_emb, transformer_options, active, progress):
    query_x = _anchor_query_x(state, layer_idx, x, progress)
    if state.batched and len(active) > 1 and not state.batched_disabled:
        try:
            return _artist_outputs_batched(
                original_forward,
                query_x,
                context,
                rope_emb,
                transformer_options,
                active,
                state.fusion_mode,
            )
        except Exception as exc:
            if _should_reraise(exc):
                raise
            logger.warning("Anima artist mixer batched path failed, using serial artist forwards: %s", exc)
            state.batched_disabled = True

    outputs = []
    for artist, _ in active:
        artist_context = _to_context_like(artist.cond, context)
        kv = _concat_contexts(context, artist_context) if state.fusion_mode in BASE_CONTEXT_FUSIONS else artist_context
        outputs.append(original_forward(query_x, context=kv, rope_emb=rope_emb, transformer_options=transformer_options))
    return outputs


def _static_capture_outputs(state, layer_idx, progress, signature, x, context, collect):
    if not state.artist_static_capture or state.fusion_mode == FUSION_CONCAT_WITH_BASE:
        return collect()
    if progress is None:
        if not state.warned_static_no_progress:
            logger.warning("Anima artist mixer cannot resolve sampling progress; Static Capture is disabled for this run.")
            state.warned_static_no_progress = True
        return collect()
    if not _stabilizer_active(state, progress):
        return collect()

    key = (layer_idx, signature, _tensor_fingerprint(context), tuple(x.shape))
    entry = state.static_cache.get(key)
    if entry is None:
        entry = {"steps": set(), "sum": None, "count": 0, "frozen": None}
        state.static_cache[key] = entry
    if entry["frozen"] is not None:
        return [value.to(device=context.device, dtype=context.dtype) for value in entry["frozen"]]

    step_key = round(float(progress), 6)
    if step_key not in entry["steps"]:
        outputs = collect()
        if entry["sum"] is None:
            entry["sum"] = [value.detach().to(torch.float32) for value in outputs]
        else:
            for index, value in enumerate(outputs):
                entry["sum"][index] = entry["sum"][index] + value.detach().to(torch.float32)
        entry["steps"].add(step_key)
        entry["count"] += 1
        if entry["count"] >= state.static_capture_k:
            inv = 1.0 / entry["count"]
            entry["frozen"] = [(value * inv).to(context.dtype).detach() for value in entry["sum"]]
            entry["sum"] = None
            entry["steps"] = None
            return [value.to(device=context.device, dtype=context.dtype) for value in entry["frozen"]]

    if entry["sum"] is None or entry["count"] <= 0:
        return collect()
    inv = 1.0 / entry["count"]
    return [(value * inv).to(device=context.device, dtype=context.dtype) for value in entry["sum"]]


def _balance_artist_outputs(outputs, base_out, strength):
    if strength <= 0.0 or len(outputs) < 2:
        return outputs
    base_f32 = base_out.to(torch.float32)
    deltas = torch.stack([(value - base_out).to(torch.float32) for value in outputs], dim=0)
    flat = deltas.reshape(deltas.shape[0], deltas.shape[1], -1)
    norms = torch.linalg.vector_norm(flat, dim=-1).clamp(min=1e-8)
    target = norms.mean(dim=0, keepdim=True)
    scale = (target / norms).clamp(min=0.5, max=2.0)
    scale = scale.view(deltas.shape[0], deltas.shape[1], *([1] * (deltas.dim() - 2)))
    balanced = deltas * (1.0 - strength) + deltas * scale * strength
    return [(base_f32 + balanced[index]).to(outputs[index]) for index in range(len(outputs))]


def _combine_lowrank_outputs(outputs, weights, base_out, rank, state):
    count = len(outputs)
    deltas = torch.stack(outputs, dim=0).to(torch.float32) - base_out.to(torch.float32).unsqueeze(0)
    shape = deltas.shape
    rows = deltas.reshape(count, -1)
    if rank < count:
        try:
            rows = _lowrank_rows(rows, rank)
        except Exception as exc:
            if _should_reraise(exc):
                raise
            if not state.warned_lowrank:
                logger.warning("Anima artist mixer low-rank constraint failed; using output average: %s", exc)
                state.warned_lowrank = True
    weight_tensor = torch.tensor(weights, device=rows.device, dtype=rows.dtype).view(count, 1)
    delta = (rows * weight_tensor).sum(dim=0).reshape(shape[1:]).to(base_out.dtype)
    return base_out + delta


def _apply_ema(state, layer_idx, progress, signature, artist_out):
    alpha = state.artist_ema_alpha
    if alpha <= 0.0 or state.artist_static_capture:
        return artist_out
    if progress is None:
        if not state.warned_ema_no_progress:
            logger.warning("Anima artist mixer cannot resolve sampling progress; EMA is disabled for this run.")
            state.warned_ema_no_progress = True
        return artist_out
    if not _stabilizer_active(state, progress):
        return artist_out
    key = (layer_idx, signature, tuple(artist_out.shape))
    previous = state.ema_cache.get(key)
    if previous is not None and previous.shape == artist_out.shape:
        artist_out = previous.to(artist_out) * alpha + artist_out * (1.0 - alpha)
    state.ema_cache[key] = artist_out.detach()
    return artist_out


def _blend_artist_output(state, base_out, artist_out, mask, strength):
    if state.fusion_mode == FUSION_QUALITY_DELTA:
        guarded = _norm_limited_delta(base_out, artist_out, 1.0)
        delta = guarded - base_out
    else:
        delta = artist_out - base_out
    if state.structure_preserve > 0.0:
        perpendicular = _project_perpendicular(delta, base_out)
        delta = delta * (1.0 - state.structure_preserve) + perpendicular * state.structure_preserve
    cap = state.delta_norm_cap
    if cap > 0.0:
        delta = _limit_delta_norm(delta, base_out, cap)
    blended = base_out + delta * float(strength)
    row_mask = torch.tensor(mask, device=base_out.device, dtype=torch.bool).view(len(mask), *([1] * (base_out.dim() - 1)))
    return torch.where(row_mask, blended, base_out)


def _norm_limited_delta(base_out, artist_out, strength, max_ratio=0.6):
    delta = artist_out - base_out
    if delta.numel() == 0:
        return base_out
    with torch.no_grad():
        flat_delta = delta.detach().float().flatten(1)
        flat_base = base_out.detach().float().flatten(1)
        delta_norm = flat_delta.norm(dim=1).clamp_min(1e-6)
        base_norm = flat_base.norm(dim=1).clamp_min(1e-6)
        limit = base_norm * float(max_ratio)
        scale = torch.minimum(torch.ones_like(delta_norm), limit / delta_norm)
        shape = [scale.shape[0]] + [1] * (delta.dim() - 1)
    return base_out + delta * scale.to(device=delta.device, dtype=delta.dtype).view(*shape) * float(strength)


def _dispatch_cross_attn(original_forward, layer_idx, state, x, context=None, rope_emb=None, transformer_options=None):
    transformer_options = transformer_options or {}
    if state.in_anchor_run:
        state.anchor_capture[layer_idx] = x.detach().clone()
        return original_forward(x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)
    state.dispatch_calls += 1
    if not state.enabled or context is None or not state.artists:
        state.fallback_calls += 1
        return original_forward(x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)
    progress = _current_progress(transformer_options, state)
    state.current_layer = layer_idx
    active = _active_artists(state, layer_idx, progress)
    if not active:
        state.fallback_calls += 1
        return original_forward(x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)
    state.active_calls += 1
    try:
        if state.combine_mode == COMBINE_CONCAT:
            return _dispatch_concat(original_forward, state, x, context, rope_emb, transformer_options, active)
        return _dispatch_output_avg(original_forward, state, x, context, rope_emb, transformer_options, active)
    except Exception as exc:
        if _should_reraise(exc):
            raise
        state.fallback_calls += 1
        logger.exception("Anima artist mixer failed at block %s, falling back to original cross-attn: %s", layer_idx, exc)
        return original_forward(x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)


def _dispatch_output_avg(original_forward, state, x, context, rope_emb, transformer_options, active):
    batch_size = context.shape[0]
    cou = transformer_options.get("cond_or_uncond") if isinstance(transformer_options, dict) else None
    if state.mask_probe is None:
        state.mask_probe = (len(cou) if cou is not None else -1, int(batch_size))
    mask = _resolve_row_mask(cou, batch_size, state.apply_uncond)
    raw_weights = [float(weight) for _, weight in active]
    total_abs = sum(abs(w) for w in raw_weights)
    if total_abs <= 1e-8:
        return original_forward(x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)
    total_influence = _clamp(total_abs, 0.0, 1.0)
    weights = [weight / total_abs for weight in raw_weights]
    progress = state.current_progress
    signature = _active_signature(active)
    outputs = _static_capture_outputs(
        state,
        getattr(state, "current_layer", -1),
        progress,
        signature,
        x,
        context,
        lambda: _collect_artist_outputs(
            original_forward,
            getattr(state, "current_layer", -1),
            state,
            x,
            context,
            rope_emb,
            transformer_options,
            active,
            progress,
        ),
    )
    base_out = original_forward(x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)
    outputs = _balance_artist_outputs(outputs, base_out, state.style_balance)
    if state.combine_mode == COMBINE_LOWRANK_AVG and len(outputs) > 1:
        artist_total = _combine_lowrank_outputs(outputs, weights, base_out, state.lowrank_k, state)
    else:
        artist_total = None
        for output, weight in zip(outputs, weights):
            artist_total = output * weight if artist_total is None else artist_total + output * weight
    artist_total = _apply_ema(state, getattr(state, "current_layer", -1), progress, signature, artist_total)
    strength_limit = 1.0 if state.fusion_mode == FUSION_QUALITY_DELTA else 2.0
    strength = _clamp(float(state.global_strength), 0.0, strength_limit) * total_influence
    if state.diff_probe is None:
        with torch.no_grad():
            delta_norm = (artist_total - base_out).detach().float().flatten(1).norm(dim=1).mean().item()
            base_norm = base_out.detach().float().flatten(1).norm(dim=1).mean().clamp_min(1e-6).item()
            state.diff_probe = (float(delta_norm), float(delta_norm / base_norm))
    return _blend_artist_output(state, base_out, artist_total, mask, strength)


def _dispatch_concat(original_forward, state, x, context, rope_emb, transformer_options, active):
    batch_size = context.shape[0]
    cou = transformer_options.get("cond_or_uncond") if isinstance(transformer_options, dict) else None
    if state.mask_probe is None:
        state.mask_probe = (len(cou) if cou is not None else -1, int(batch_size))
    mask = _resolve_row_mask(cou, batch_size, state.apply_uncond)
    raw_weights = [float(weight) for _, weight in active]
    total_abs = sum(abs(w) for w in raw_weights)
    if total_abs <= 1e-8:
        return original_forward(x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)
    total_influence = _clamp(total_abs, 0.0, 1.0)
    parts = []
    for artist, weight in active:
        artist_context = _to_context_like(artist.cond, context)
        parts.append(artist_context * (weight / total_abs))
    combined = torch.cat(parts, dim=_context_token_dim(context))
    artist_context = _concat_contexts(context, combined) if state.fusion_mode in BASE_CONTEXT_FUSIONS else combined
    progress = state.current_progress
    signature = ("concat", _active_signature(active))
    query_x = _anchor_query_x(state, state.current_layer, x, progress)
    artist_out = _static_capture_outputs(
        state,
        state.current_layer,
        progress,
        signature,
        x,
        context,
        lambda: [original_forward(query_x, context=artist_context, rope_emb=rope_emb, transformer_options=transformer_options)],
    )[0]
    artist_out = _apply_ema(state, state.current_layer, progress, signature, artist_out)
    strength_limit = 1.0 if state.fusion_mode == FUSION_QUALITY_DELTA else 2.0
    strength = _clamp(float(state.global_strength), 0.0, strength_limit) * total_influence
    base_out = original_forward(x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)
    if state.diff_probe is None:
        with torch.no_grad():
            delta_norm = (artist_out - base_out).detach().float().flatten(1).norm(dim=1).mean().item()
            base_norm = base_out.detach().float().flatten(1).norm(dim=1).mean().clamp_min(1e-6).item()
            state.diff_probe = (float(delta_norm), float(delta_norm / base_norm))
    return _blend_artist_output(state, base_out, artist_out, mask, strength)


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
    optimization = _option_key("optimization", optimization, OPT_BALANCE)
    rows = _normalize_rows(rows, None, shift, optimization, display=False)
    prompts = _current_prompts(p)
    artists = []
    for row in rows:
        enabled, artist_text, weight, blocks_text, start, end, peak, curve, _stage, _auto = row
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
                    curve=_option_key("curve", curve, CURVE_SMOOTH),
                    cond=cond,
                )
            )
    return artists


def _optimization_defaults(optimization):
    optimization = _option_key("optimization", optimization, OPT_BALANCE)
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
        lang = _language()
        defaults = _initial_ui_defaults(lang)
        intro_default_choice = _language_choice(lang)
        table_css = f"""
        <style>
        #{self.elem_id("accordion")} [data-testid="sort-button"],
        #{self.elem_id("accordion")} .sort-button,
        #{self.elem_id("accordion")} button[aria-label*="Sort"],
        #{self.elem_id("accordion")} button[title*="Sort"] {{
            display: none !important;
        }}
        #{self.elem_id("save_current_settings_row")} {{
            display: none !important;
        }}
        </style>
        """
        with gr.Accordion(_t("accordion"), open=False, elem_id=self.elem_id("accordion")):
            gr.HTML(table_css)
            gr.HTML(_shift_sync_script(self.elem_id("runtime_base_shift"), self.elem_id("runtime_hires_shift"), is_img2img=is_img2img))
            gr.HTML(_current_settings_autosave_script(self.elem_id("accordion"), self.elem_id("save_current_settings")))
            with gr.Row():
                intro_language = gr.Radio(
                    choices=["English", "中文"],
                    value=intro_default_choice,
                    label=_t("intro_language"),
                    elem_id=self.elem_id("intro_language"),
                )
            intro = gr.Markdown(value=_intro_default(lang), elem_id=self.elem_id("intro"))
            intro_status = gr.HTML(value="", elem_id=self.elem_id("intro_status"))

            runtime_base_shift = gr.Number(value=3.0, visible=False, elem_id=self.elem_id("runtime_base_shift"))
            runtime_hires_shift = gr.Number(value=3.0, visible=False, elem_id=self.elem_id("runtime_hires_shift"))
            with gr.Row(elem_id=self.elem_id("save_current_settings_row")):
                save_current_settings = gr.Button(value="Save current settings", elem_id=self.elem_id("save_current_settings"))
                save_current_settings_status = gr.Markdown(value="", elem_id=self.elem_id("save_current_settings_status"))
            enable = gr.Checkbox(label=_t("enable"), value=defaults["enable"], elem_id=self.elem_id("enable"))
            with gr.Row():
                optimization = gr.Dropdown(label=_t("optimization"), choices=_option_choices("optimization", lang), value=defaults["optimization"], elem_id=self.elem_id("optimization"))
                global_strength = gr.Slider(label=_t("global_strength"), minimum=0.0, maximum=2.0, step=0.01, value=defaults["global_strength"], elem_id=self.elem_id("global_strength"))
                enable_cache = gr.Checkbox(label=_t("cache"), value=defaults["enable_cache"], elem_id=self.elem_id("enable_cache"))
            with gr.Row():
                combine_mode = gr.Dropdown(label=_t("combine"), choices=_option_choices("combine", lang), value=defaults["combine"], elem_id=self.elem_id("combine_mode"))
                fusion_mode = gr.Dropdown(label=_t("fusion"), choices=_option_choices("fusion", lang), value=defaults["fusion"], elem_id=self.elem_id("fusion_mode"))
                apply_uncond = gr.Checkbox(label=_t("apply_uncond"), value=defaults["apply_uncond"], elem_id=self.elem_id("apply_uncond"))

            with gr.Tab(_t("base_tab")):
                with gr.Row():
                    base_row_count = gr.Number(label=_t("row_count"), value=defaults["base_count"], precision=0, elem_id=self.elem_id("base_row_count"))
                    base_apply_presets = gr.Button(_t("normalize_rows"), elem_id=self.elem_id("base_apply_presets"))
                base_shift_hint = gr.Markdown(value=_t("shift_runtime_hint"))
                base_artist_table_label = gr.Markdown(value=f"**{_t('artist_table')}**")
                base_row_components = _create_artist_row_controls("base", lang, defaults["base_rows"], self.elem_id)

            with gr.Tab(_t("hires_tab")):
                hires_independent = gr.Checkbox(label=_t("hires_independent"), value=defaults["hires_independent"], elem_id=self.elem_id("hires_independent"))
                disable_hires_mixing = gr.Checkbox(label=_t("disable_hires_mixing"), value=defaults["disable_hires_mixing"], elem_id=self.elem_id("disable_hires_mixing"))
                with gr.Row():
                    hires_row_count = gr.Number(label=_t("hr_row_count"), value=defaults["hires_count"], precision=0, elem_id=self.elem_id("hires_row_count"))
                    hires_apply_presets = gr.Button(_t("normalize_rows"), elem_id=self.elem_id("hires_apply_presets"))
                hires_shift_hint = gr.Markdown(value=_t("shift_runtime_hint"))
                hires_artist_table_label = gr.Markdown(value=f"**{_t('artist_table')}**")
                hires_row_components = _create_artist_row_controls("hires", lang, defaults["hires_rows"], self.elem_id)

            advanced_defaults = defaults["advanced"]
            with gr.Tab(_t("advanced_tab")):
                advanced_hint = gr.Markdown(value=_t("advanced_hint"))
                with gr.Row():
                    artist_ema_alpha = gr.Slider(label=_t("artist_ema_alpha"), minimum=0.0, maximum=0.95, step=0.05, value=advanced_defaults["artist_ema_alpha"], elem_id=self.elem_id("artist_ema_alpha"))
                    lowrank_k = gr.Slider(label=_t("lowrank_k"), minimum=1, maximum=MAX_ARTIST_ROWS, step=1, value=advanced_defaults["lowrank_k"], elem_id=self.elem_id("lowrank_k"))
                    stabilizer_end_percent = gr.Slider(label=_t("stabilizer_end_percent"), minimum=0.0, maximum=1.0, step=0.05, value=advanced_defaults["stabilizer_end_percent"], elem_id=self.elem_id("stabilizer_end_percent"))
                with gr.Row():
                    artist_static_capture = gr.Checkbox(label=_t("artist_static_capture"), value=advanced_defaults["artist_static_capture"], elem_id=self.elem_id("artist_static_capture"))
                    static_capture_k = gr.Slider(label=_t("static_capture_k"), minimum=1, maximum=STATIC_CAPTURE_K_MAX, step=1, value=advanced_defaults["static_capture_k"], elem_id=self.elem_id("static_capture_k"))
                with gr.Row():
                    artist_anchor_q = gr.Checkbox(label=_t("artist_anchor_q"), value=advanced_defaults["artist_anchor_q"], elem_id=self.elem_id("artist_anchor_q"))
                    anchor_seed_list = gr.Textbox(label=_t("anchor_seed_list"), value=advanced_defaults["anchor_seed_list"], lines=1, elem_id=self.elem_id("anchor_seed_list"))
                    anchor_seeds_count = gr.Slider(label=_t("anchor_seeds_count"), minimum=1, maximum=ANCHOR_SEEDS_MAX, step=1, value=advanced_defaults["anchor_seeds_count"], elem_id=self.elem_id("anchor_seeds_count"))
                with gr.Row():
                    anchor_user_blend = gr.Slider(label=_t("anchor_user_blend"), minimum=0.0, maximum=1.0, step=0.05, value=advanced_defaults["anchor_user_blend"], elem_id=self.elem_id("anchor_user_blend"))
                    anchor_deep_layer_threshold = gr.Number(label=_t("anchor_deep_layer_threshold"), value=advanced_defaults["anchor_deep_layer_threshold"], precision=0, elem_id=self.elem_id("anchor_deep_layer_threshold"))
                with gr.Row():
                    style_balance = gr.Slider(label=_t("style_balance"), minimum=0.0, maximum=1.0, step=0.05, value=advanced_defaults["style_balance"], elem_id=self.elem_id("style_balance"))
                    structure_preserve = gr.Slider(label=_t("structure_preserve"), minimum=0.0, maximum=1.0, step=0.05, value=advanced_defaults["structure_preserve"], elem_id=self.elem_id("structure_preserve"))
                    delta_norm_cap = gr.Slider(label=_t("delta_norm_cap"), minimum=0.0, maximum=4.0, step=0.05, value=advanced_defaults["delta_norm_cap"], elem_id=self.elem_id("delta_norm_cap"))

            advanced_components = [
                artist_ema_alpha,
                lowrank_k,
                artist_static_capture,
                static_capture_k,
                artist_anchor_q,
                anchor_seed_list,
                anchor_seeds_count,
                anchor_user_blend,
                anchor_deep_layer_threshold,
                stabilizer_end_percent,
                style_balance,
                structure_preserve,
                delta_norm_cap,
            ]

            with gr.Tab(_t("template")):
                with gr.Row():
                    template_choices = _template_choices()
                    template_dropdown = gr.Dropdown(label=_t("template"), choices=template_choices, value=_resolve_template_name(defaults["template_name"], template_choices), allow_custom_value=False, elem_id=self.elem_id("template_dropdown"))
                    template_apply_target = gr.Dropdown(label=_t("apply_target"), choices=_option_choices("apply_target", lang), value=defaults["template_target"], elem_id=self.elem_id("template_apply_target"))
                    template_apply = gr.Button(_t("apply_template"), variant="primary", elem_id=self.elem_id("template_apply"))
                with gr.Row():
                    template_name = gr.Textbox(label=_t("template_name"), value="", elem_id=self.elem_id("template_name"))
                    template_save_base = gr.Button(_t("save_base_template"), elem_id=self.elem_id("template_save_base"))
                    template_save_hires = gr.Button(_t("save_hires_template"), elem_id=self.elem_id("template_save_hires"))
                with gr.Row():
                    rename_to = gr.Textbox(label=_t("rename_to"), value="", elem_id=self.elem_id("rename_to"))
                    rename_button = gr.Button(_t("rename_template"), elem_id=self.elem_id("rename_button"))
                    delete_button = gr.Button(_t("delete_template"), elem_id=self.elem_id("delete_button"))
                    reset_defaults_button = gr.Button(_t("reset_defaults"), elem_id=self.elem_id("reset_defaults_button"))
                template_status = gr.Markdown(value="", elem_id=self.elem_id("template_status"))

            with gr.Tab(_t("help")):
                help_text = gr.Markdown(value=HELP_ZH if _language() == "zh" else HELP_EN)

        def refresh_language(
            language,
            optimization_value,
            combine_value,
            fusion_value,
            template_target_value,
            *row_values,
        ):
            intro_update, status_update = _save_ui_language(language)
            language_code = _choice_to_language(language)
            labels = LANG[language_code]
            row_labels = _row_component_headers(language_code)
            bool_choices = [_bool_label(True, language_code), _bool_label(False, language_code)]
            curve_choices = _option_choices("curve", language_code)
            stage_choices = _option_choices("stage", language_code)
            base_values = row_values[: MAX_ARTIST_ROWS * 10]
            hires_values = row_values[MAX_ARTIST_ROWS * 10 : MAX_ARTIST_ROWS * 20]

            outputs = [
                intro_update,
                status_update,
                gr.update(label=labels["intro_language"]),
                gr.update(label=labels["enable"]),
                gr.update(label=labels["optimization"], choices=_option_choices("optimization", language_code), value=_option_label("optimization", optimization_value, language_code)),
                gr.update(label=labels["global_strength"]),
                gr.update(label=labels["cache"]),
                gr.update(label=labels["combine"], choices=_option_choices("combine", language_code), value=_option_label("combine", combine_value, language_code)),
                gr.update(label=labels["fusion"], choices=_option_choices("fusion", language_code), value=_option_label("fusion", fusion_value, language_code)),
                gr.update(label=labels["apply_uncond"]),
                gr.update(value=labels["advanced_hint"]),
                gr.update(label=labels["artist_ema_alpha"]),
                gr.update(label=labels["lowrank_k"]),
                gr.update(label=labels["artist_static_capture"]),
                gr.update(label=labels["static_capture_k"]),
                gr.update(label=labels["artist_anchor_q"]),
                gr.update(label=labels["anchor_seed_list"]),
                gr.update(label=labels["anchor_seeds_count"]),
                gr.update(label=labels["anchor_user_blend"]),
                gr.update(label=labels["anchor_deep_layer_threshold"]),
                gr.update(label=labels["stabilizer_end_percent"]),
                gr.update(label=labels["style_balance"]),
                gr.update(label=labels["structure_preserve"]),
                gr.update(label=labels["delta_norm_cap"]),
                gr.update(label=labels["row_count"]),
                gr.update(value=labels["normalize_rows"]),
                gr.update(value=labels["shift_runtime_hint"]),
                gr.update(value=f"**{labels['artist_table']}**"),
                gr.update(label=labels["hires_independent"]),
                gr.update(label=labels["disable_hires_mixing"]),
                gr.update(label=labels["hr_row_count"]),
                gr.update(value=labels["normalize_rows"]),
                gr.update(value=labels["shift_runtime_hint"]),
                gr.update(value=f"**{labels['artist_table']}**"),
                gr.update(label=labels["template"]),
                gr.update(label=labels["apply_target"], choices=_option_choices("apply_target", language_code), value=_option_label("apply_target", template_target_value, language_code)),
                gr.update(value=labels["apply_template"]),
                gr.update(label=labels["template_name"]),
                gr.update(value=labels["save_base_template"]),
                gr.update(value=labels["save_hires_template"]),
                gr.update(label=labels["rename_to"]),
                gr.update(value=labels["rename_template"]),
                gr.update(value=labels["delete_template"]),
                gr.update(value=labels["reset_defaults"]),
                gr.update(value=HELP_EN if language_code == "en" else HELP_ZH),
            ]

            for values in (base_values, hires_values):
                for offset in range(0, MAX_ARTIST_ROWS * 10, 10):
                    enabled_value = values[offset] if offset < len(values) else True
                    curve_value = values[offset + 7] if offset + 7 < len(values) else CURVE_SMOOTH
                    stage_value = values[offset + 8] if offset + 8 < len(values) else PRESET_CUSTOM
                    auto_value = values[offset + 9] if offset + 9 < len(values) else True
                    outputs.extend(
                        [
                            gr.update(label=row_labels["enabled"], choices=bool_choices, value=_bool_label(enabled_value, language_code)),
                            gr.update(label=row_labels["artist"]),
                            gr.update(label=row_labels["weight"]),
                            gr.update(label=row_labels["blocks"]),
                            gr.update(label=row_labels["start"]),
                            gr.update(label=row_labels["end"]),
                            gr.update(label=row_labels["peak"]),
                            gr.update(label=row_labels["curve"], choices=curve_choices, value=_option_label("curve", curve_value, language_code)),
                            gr.update(label=row_labels["stage"], choices=stage_choices, value=_option_label("stage", stage_value, language_code)),
                            gr.update(label=row_labels["auto_shift"], choices=bool_choices, value=_bool_label(auto_value, language_code)),
                        ]
                    )
            return outputs

        intro_language.change(
            fn=refresh_language,
            inputs=[intro_language, optimization, combine_mode, fusion_mode, template_apply_target, *base_row_components, *hires_row_components],
            outputs=[
                intro,
                intro_status,
                intro_language,
                enable,
                optimization,
                global_strength,
                enable_cache,
                combine_mode,
                fusion_mode,
                apply_uncond,
                advanced_hint,
                *advanced_components,
                base_row_count,
                base_apply_presets,
                base_shift_hint,
                base_artist_table_label,
                hires_independent,
                disable_hires_mixing,
                hires_row_count,
                hires_apply_presets,
                hires_shift_hint,
                hires_artist_table_label,
                template_dropdown,
                template_apply_target,
                template_apply,
                template_name,
                template_save_base,
                template_save_hires,
                rename_to,
                rename_button,
                delete_button,
                reset_defaults_button,
                help_text,
                *base_row_components,
                *hires_row_components,
            ],
            queue=False,
            show_progress=False,
            **_internal_event_kwargs(),
        )

        base_row_count.change(
            fn=_resize_row_components,
            inputs=[base_row_count, runtime_base_shift, optimization, *base_row_components],
            outputs=base_row_components,
            queue=False,
            show_progress=False,
            **_internal_event_kwargs(),
        )
        hires_row_count.change(
            fn=_resize_row_components,
            inputs=[hires_row_count, runtime_hires_shift, optimization, *hires_row_components],
            outputs=hires_row_components,
            queue=False,
            show_progress=False,
            **_internal_event_kwargs(),
        )
        runtime_base_shift.change(
            fn=_apply_shift_to_timing_components,
            inputs=[base_row_count, runtime_base_shift, optimization, *base_row_components],
            outputs=_row_outputs(base_row_components, (4, 5, 6)),
            queue=False,
            show_progress=False,
            **_internal_event_kwargs(),
        )
        runtime_hires_shift.change(
            fn=_apply_shift_to_timing_components,
            inputs=[hires_row_count, runtime_hires_shift, optimization, *hires_row_components],
            outputs=_row_outputs(hires_row_components, (4, 5, 6)),
            queue=False,
            show_progress=False,
            **_internal_event_kwargs(),
        )
        base_apply_presets.click(
            fn=_resize_row_components,
            inputs=[base_row_count, runtime_base_shift, optimization, *base_row_components],
            outputs=base_row_components,
            queue=False,
            show_progress=False,
            **_internal_event_kwargs(),
        )
        hires_apply_presets.click(
            fn=_resize_row_components,
            inputs=[hires_row_count, runtime_hires_shift, optimization, *hires_row_components],
            outputs=hires_row_components,
            queue=False,
            show_progress=False,
            **_internal_event_kwargs(),
        )
        optimization.change(
            fn=_apply_optimization_to_block_timing_components,
            inputs=[base_row_count, runtime_base_shift, optimization, *base_row_components],
            outputs=_row_outputs(base_row_components, (3, 4, 5, 6)),
            queue=False,
            show_progress=False,
            **_internal_event_kwargs(),
        )
        optimization.change(
            fn=_apply_optimization_to_block_timing_components,
            inputs=[hires_row_count, runtime_hires_shift, optimization, *hires_row_components],
            outputs=_row_outputs(hires_row_components, (3, 4, 5, 6)),
            queue=False,
            show_progress=False,
            **_internal_event_kwargs(),
        )
        for row_components, row_count, runtime_shift in (
            (base_row_components, base_row_count, runtime_base_shift),
            (hires_row_components, hires_row_count, runtime_hires_shift),
        ):
            for offset in range(0, len(row_components), 10):
                row_components[offset + 8].change(
                    fn=partial(_apply_stage_to_row_components, offset // 10),
                    inputs=[row_count, runtime_shift, optimization, *row_components],
                    outputs=[
                        row_components[offset + 4],
                        row_components[offset + 5],
                        row_components[offset + 6],
                        row_components[offset + 9],
                    ],
                    queue=False,
                    show_progress=False,
                    **_internal_event_kwargs(),
                )
                row_components[offset + 9].change(
                    fn=partial(_apply_auto_shift_to_row_components, offset // 10),
                    inputs=[row_count, runtime_shift, optimization, *row_components],
                    outputs=[
                        row_components[offset + 4],
                        row_components[offset + 5],
                        row_components[offset + 6],
                    ],
                    queue=False,
                    show_progress=False,
                    **_internal_event_kwargs(),
                )
        template_save_base.click(
            fn=_save_base_template_ui,
            inputs=[
                template_name,
                base_row_count,
                runtime_base_shift,
                disable_hires_mixing,
                global_strength,
                optimization,
                combine_mode,
                fusion_mode,
                apply_uncond,
                enable_cache,
                *advanced_components,
                *base_row_components,
            ],
            outputs=[template_dropdown, template_status],
            queue=False,
            show_progress=False,
            **_internal_event_kwargs(),
        )
        template_save_hires.click(
            fn=_save_hires_template_ui,
            inputs=[
                template_name,
                hires_independent,
                hires_row_count,
                runtime_hires_shift,
                disable_hires_mixing,
                global_strength,
                optimization,
                combine_mode,
                fusion_mode,
                apply_uncond,
                enable_cache,
                *advanced_components,
                *hires_row_components,
            ],
            outputs=[template_dropdown, template_status],
            queue=False,
            show_progress=False,
            **_internal_event_kwargs(),
        )
        rename_button.click(fn=_rename_template_ui, inputs=[template_dropdown, rename_to], outputs=[template_dropdown, template_status], queue=False, show_progress=False, **_internal_event_kwargs())
        delete_button.click(fn=_delete_template_ui, inputs=[template_dropdown], outputs=[template_dropdown, template_status], queue=False, show_progress=False, **_internal_event_kwargs())
        template_apply.click(
            fn=_apply_template_ui,
            inputs=[template_dropdown, template_apply_target, base_row_count, hires_row_count, runtime_base_shift, runtime_hires_shift, optimization, *advanced_components, *base_row_components, *hires_row_components],
            outputs=[
                base_row_count,
                hires_row_count,
                hires_independent,
                disable_hires_mixing,
                runtime_base_shift,
                runtime_hires_shift,
                global_strength,
                optimization,
                combine_mode,
                fusion_mode,
                apply_uncond,
                enable_cache,
                *advanced_components,
                template_status,
                *base_row_components,
                *hires_row_components,
            ],
            queue=False,
            show_progress=False,
            **_internal_event_kwargs(),
        )
        reset_defaults_button.click(
            fn=_reset_all_defaults_ui,
            inputs=[],
            outputs=[
                enable,
                template_dropdown,
                template_apply_target,
                base_row_count,
                hires_row_count,
                hires_independent,
                disable_hires_mixing,
                runtime_base_shift,
                runtime_hires_shift,
                global_strength,
                optimization,
                combine_mode,
                fusion_mode,
                apply_uncond,
                enable_cache,
                *advanced_components,
                template_status,
                *base_row_components,
                *hires_row_components,
            ],
            queue=False,
            show_progress=False,
            **_internal_event_kwargs(),
        )
        current_settings_inputs = [
            enable,
            template_dropdown,
            template_apply_target,
            base_row_count,
            hires_row_count,
            hires_independent,
            disable_hires_mixing,
            runtime_base_shift,
            runtime_hires_shift,
            global_strength,
            optimization,
            combine_mode,
            fusion_mode,
            apply_uncond,
            enable_cache,
            *advanced_components,
            *base_row_components,
            *hires_row_components,
        ]
        _do_not_save_to_ui_config(save_current_settings, *current_settings_inputs)
        save_current_settings.click(
            fn=_save_current_settings_ui,
            inputs=current_settings_inputs,
            outputs=[save_current_settings_status],
            queue=False,
            show_progress=False,
            **_internal_event_kwargs(),
        )

        return [
            enable,
            base_row_count,
            hires_row_count,
            hires_independent,
            disable_hires_mixing,
            runtime_base_shift,
            runtime_hires_shift,
            global_strength,
            optimization,
            combine_mode,
            fusion_mode,
            apply_uncond,
            enable_cache,
            *advanced_components,
            *base_row_components,
            *hires_row_components,
        ]

    def process_before_every_sampling(self, p, *args, **kwargs):
        global _ACTIVE_STATE
        base_arg_count = 13
        advanced_arg_count = len(ADVANCED_SETTING_KEYS)
        if len(args) < base_arg_count + advanced_arg_count:
            return
        (
            enable,
            base_row_count,
            hires_row_count,
            hires_independent,
            disable_hires_mixing,
            runtime_base_shift,
            runtime_hires_shift,
            global_strength,
            optimization,
            combine_mode,
            fusion_mode,
            apply_uncond,
            enable_cache,
        ) = args[:base_arg_count]
        advanced_values = args[base_arg_count : base_arg_count + advanced_arg_count]
        advanced = _advanced_settings_from_values(advanced_values)
        component_values = args[base_arg_count + advanced_arg_count :]
        base_values = component_values[: MAX_ARTIST_ROWS * 10]
        hires_values = component_values[MAX_ARTIST_ROWS * 10 : MAX_ARTIST_ROWS * 20]
        _save_runtime_current_settings(
            enable,
            base_row_count,
            hires_row_count,
            hires_independent,
            disable_hires_mixing,
            runtime_base_shift,
            runtime_hires_shift,
            global_strength,
            optimization,
            combine_mode,
            fusion_mode,
            apply_uncond,
            enable_cache,
            *advanced_values,
            *component_values,
        )

        _unpatch_cross_attn()
        _ACTIVE_STATE = None
        if not enable:
            return
        if getattr(p, "is_hr_pass", False) and bool(disable_hires_mixing):
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
        fallback_shift = runtime_hires_shift if getattr(p, "is_hr_pass", False) and hires_independent else runtime_base_shift
        shift = _runtime_shift_from_processing(p, fallback_shift)
        base_rows = _active_rows_from_components(base_row_count, shift, optimization, *base_values, display=False)
        hires_rows = _active_rows_from_components(hires_row_count, shift, optimization, *hires_values, display=False)
        rows = hires_rows if getattr(p, "is_hr_pass", False) and hires_independent else base_rows
        optimization = _option_key("optimization", optimization, OPT_BALANCE)
        combine_mode = _option_key("combine", combine_mode, COMBINE_OUTPUT_AVG)
        fusion_mode = _option_key("fusion", fusion_mode, FUSION_INTERPOLATE)
        if advanced["artist_anchor_q"] and advanced["artist_static_capture"]:
            logger.warning("Anima artist mixer Anchor Q and Static Capture were both enabled; Static Capture is disabled for this run.")
            advanced["artist_static_capture"] = False
        if advanced["artist_anchor_q"] and fusion_mode == FUSION_CONCAT_WITH_BASE:
            logger.warning("Anima artist mixer Anchor Q is incompatible with concat-with-base fusion and was disabled for this run.")
            advanced["artist_anchor_q"] = False
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
        strength_limit = 1.0 if fusion_mode == FUSION_QUALITY_DELTA else 2.0
        state = MixerState(
            run_id=uuid.uuid4().hex,
            enabled=True,
            global_strength=_clamp(_to_float(global_strength, 0.7), 0.0, strength_limit),
            combine_mode=combine_mode,
            fusion_mode=fusion_mode,
            apply_uncond=bool(apply_uncond),
            batched=bool(defaults["batched"]),
            artist_ema_alpha=advanced["artist_ema_alpha"],
            lowrank_k=advanced["lowrank_k"],
            artist_static_capture=advanced["artist_static_capture"],
            static_capture_k=advanced["static_capture_k"],
            artist_anchor_q=advanced["artist_anchor_q"],
            anchor_seed_list=_parse_anchor_seed_list(advanced["anchor_seed_list"]),
            anchor_seeds_count=advanced["anchor_seeds_count"],
            anchor_user_blend=advanced["anchor_user_blend"],
            anchor_deep_layer_threshold=advanced["anchor_deep_layer_threshold"],
            stabilizer_end_percent=advanced["stabilizer_end_percent"],
            style_balance=advanced["style_balance"],
            structure_preserve=advanced["structure_preserve"],
            delta_norm_cap=advanced["delta_norm_cap"],
            dm_ref=dm,
            artists=artists,
        )
        previous_states = list(_RUN_STATES)
        for previous in previous_states:
            previous.superseded = True
        _install_cross_attn_patch(dm, state)
        _install_model_wrapper(unet, dm, state)
        _ACTIVE_STATE = state
        _RUN_STATES.append(state)
        p.extra_generation_params["Anima Artist Mixer"] = (
            f"{len(artists)} artists, strength={state.global_strength:.2f}, "
            f"{state.combine_mode}/{state.fusion_mode}, preset={optimization}, shift={shift:.2f}"
        )

    def postprocess(self, p, processed, *args, **kwargs):
        global _ACTIVE_STATE
        states = list(_RUN_STATES)
        _unpatch_cross_attn()
        _ACTIVE_STATE = None
        _RUN_STATES.clear()
        for state in states:
            if state.superseded:
                continue
            if not state.dispatch_calls:
                logger.warning(
                    "Anima artist mixer did not receive any cross-attn calls for run %s. "
                    "Patched blocks=%s, wrapper checks=%d. If the image is unchanged, reload UI and test again.",
                    state.run_id,
                    _summarize_blocks(state.patched_blocks),
                    state.wrapper_checks,
                )
            else:
                logger.info(
                    "Anima artist mixer run %s finished: dispatch=%d active=%d fallback=%d patched=%s wrapper_checks=%d strength=%.2f mode=%s/%s",
                    state.run_id,
                    state.dispatch_calls,
                    state.active_calls,
                    state.fallback_calls,
                    _summarize_blocks(state.patched_blocks),
                    state.wrapper_checks,
                    state.global_strength,
                    state.combine_mode,
                    state.fusion_mode,
                )
                logger.info(
                    "Anima artist mixer context path: fusion=%s uses_base_concat=%s",
                    state.fusion_mode,
                    state.fusion_mode in BASE_CONTEXT_FUSIONS,
                )
                if state.diff_probe is not None:
                    logger.info(
                        "Anima artist mixer diff probe: artist_delta_norm=%.4f artist_delta_ratio=%.4f",
                        state.diff_probe[0],
                        state.diff_probe[1],
                    )
                if state.mask_probe is not None:
                    logger.info(
                        "Anima artist mixer mask probe: cond_or_uncond_len=%d context_batch=%d apply_uncond=%s",
                        state.mask_probe[0],
                        state.mask_probe[1],
                        state.apply_uncond,
                    )
