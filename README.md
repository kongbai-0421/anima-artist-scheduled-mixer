# Anima Artist Scheduled Mixer

Anima Artist Scheduled Mixer is a Forge Neo extension for mixing Anima artist tags in a way closer to SDXL-style artist chains.

Instead of asking Anima's LLM text encoder to understand several artist names in one contextualized prompt, the extension encodes every artist row independently and mixes those artist conditionings inside Anima cross-attention during sampling. This follows the same core idea as [An1X3R/Anima-Artist-Mixer](https://github.com/An1X3R/Anima-Artist-Mixer): split artists, encode separately, then blend at cross-attention instead of modifying text-encoder layers.

Special thanks to **An1X3R/Anima-Artist-Mixer** and **汐浮尘/utowo** for the original split-and-encode and cross-attention mixing design.

## Features

- Per-artist rows with artist prompt, weight, block range, denoise time range, peak, curve, and stage preset.
- Inline row weights in the artist cell, for example `(wlop:1.2)`, `[wlop:0.8]`, or `wlop:1.2`; this multiplier is combined with the row `Weight`.
- Separate Base and Hires. fix artist chains. Hires. fix is disabled by default and inherits Base settings.
- Current settings are saved automatically when controls change, then restored after refreshing the UI or restarting Forge.
- Template save, apply, rename, and delete. Templates remain editable after applying and can be reused across sessions.
- Reset all settings to the built-in defaults when you want a clean state.
- Optional switch to disable artist mixing during Hires. fix while keeping it active for the base pass.
- Templates store stable internal values and can be applied across English/Chinese UI modes.
- Localized option labels for optimization presets, combine mode, fusion mode, curve, stage, and template target.
- Performance, Balance, and Quality optimization presets. Balance is the default.
- English/Chinese UI setting at the top of the panel; switching it refreshes only this plugin panel.
- Text-encoding cache for repeated prompts and artist rows.
- Reference-style `Output average + Interpolate` by default, with `Quality-safe delta` still available for a steadier, more conservative blend.
- Advanced stability tab with EMA, low-rank constraint, Static Capture, Anchor Q, Style Balance, and Structure Guard.

## Advanced Stability

These controls are disabled by default, so existing configurations keep their previous behavior. They operate on artist cross-attention branches during sampling and do not modify model files.

- `EMA` smooths artist outputs across sampling steps to reduce step-to-step jitter. Start with `EMA smoothing strength` at `0.25-0.50`.
- `Low-rank constraint` is exposed as the `Low-rank average` combine mode. It projects multiple artist deltas into a rank-K subspace to reduce cross-seed drift. Start with rank `1-2`; raise it only for larger artist sets.
- `Static Capture` averages artist outputs from the first K distinct sampling steps and freezes the result. Start with `4-6` warmup steps; it adds early compute and VRAM use.
- `Anchor Q` runs a fixed-seed pre-pass, captures cross-attention query hidden states, and replaces queries only on artist branches. Up to four fixed seeds can be entered. With `Anchor/current Q blend` at `0`, the artist branch uses the Anchor fully; start around `0.15-0.35`. Anchor Q adds a pre-pass cost.
- `Style Balance` gently equalizes artist-delta norms before row weights are applied, preventing a high-amplitude artist from dominating. Start with `0.25-0.50`.
- `Structure Guard` projects the final artist delta away from the base-output direction and can cap its magnitude relative to the base output. Start with `Structure preservation` at `0.20-0.40`, then optionally use a `Delta cap` around `0.20-0.50` for stronger protection.

`Stabilizer end progress` limits EMA, Static Capture, and Anchor Q to the early part of sampling. Anchor Q takes priority over Static Capture when both are enabled; Anchor Q is disabled automatically with `Concat with base` fusion because that path changes the query/context contract. Combining more stabilizers increases compute and VRAM use, so add them one at a time while comparing results.

## Row Controls

Each row is intended to hold one artist prompt. This gives separate control over weight, blocks, timing, curve, and stage. Comma-separated artist chains still work for quick migration, but one artist per row is recommended for predictable scheduling.

`Stage + Shift + Auto Shift` rewrites the row time window in a way similar to Anima LoRA Stage Scheduler: Composition is early, Character is middle, and Style is late. Selecting a non-Custom stage refreshes Start, End, and Peak immediately. Turn off Auto Shift afterward when you want to fine-tune those values manually.

## Settings And Templates

Changing any main control updates `artist_mixer_current_settings.json` locally. That file is ignored by git and is used only to restore your last UI state after a refresh or restart.

Use `Reset all to defaults` to clear the local current-settings file and restore the built-in defaults. Template files are not deleted by reset.

The Hires. fix tab includes `Disable artist mixing during Hires. fix`. When enabled, the plugin unpatches before the Hires pass and leaves Hires. fix generation untouched.

## Recommended Defaults

- Optimization: `Balance`
- Combine mode: `Output average`
- Fusion mode: `Interpolate`
- Global artist strength: `0.70`
- Default blocks: `0-27` for Balance and Quality, `10-18` for Performance.
- Use `Quality-safe delta` when the reference-style interpolation is too strong for a prompt.

## Notes

This cannot be perfectly lossless like SDXL artist mixing because Anima uses a non-linear LLM text encoder. The goal is controlled, usable style blending with lower interference between artist tags.

The extension patches Anima cross-attention only during sampling and restores the original forwards after generation. It does not edit text-encoder layers or model files.

## License

See [LICENSE](LICENSE) and [LICENSE_zh.md](LICENSE_zh.md).
