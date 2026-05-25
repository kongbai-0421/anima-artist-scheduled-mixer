# Anima Artist Scheduled Mixer

Anima Artist Scheduled Mixer is a Forge Neo extension for mixing Anima artist tags in a way closer to SDXL-style artist chains.

Instead of asking Anima's LLM text encoder to understand several artist names in one contextualized prompt, the extension encodes every artist row independently and mixes those artist conditionings inside Anima cross-attention during sampling. This follows the same core idea as [An1X3R/Anima-Artist-Mixer](https://github.com/An1X3R/Anima-Artist-Mixer): split artists, encode separately, then blend at cross-attention instead of modifying text-encoder layers.

Special thanks to **An1X3R/Anima-Artist-Mixer** and **汐浮尘/utowo** for the original split-and-encode and cross-attention mixing design.

## Features

- Per-artist rows with artist prompt, weight, block range, denoise time range, peak, curve, and stage preset.
- Inline row weights in the artist cell, for example `(wlop:1.2)`, `[wlop:0.8]`, or `wlop:1.2`; this multiplier is combined with the row `Weight`.
- Separate Base and Hires. fix artist chains. Hires. fix is disabled by default and inherits Base settings.
- Template save, apply, rename, and delete. Templates store stable internal values and can be applied across English/Chinese UI modes.
- Localized option labels for optimization presets, combine mode, fusion mode, curve, stage, and template target.
- Performance, Balance, and Quality optimization presets. Balance is the default.
- English/Chinese UI setting, plus a description switch at the top of the panel.
- Text-encoding cache for repeated prompts and artist rows.
- Quality-safe delta fusion by default, with the stronger reference-style `Output average + Interpolate` path still available.

## Row Controls

Each row is intended to hold one artist prompt. This gives separate control over weight, blocks, timing, curve, and stage. Comma-separated artist chains still work for quick migration, but one artist per row is recommended for predictable scheduling.

`Stage + Shift + Auto Shift` rewrites the row time window in a way similar to Anima LoRA Stage Scheduler: Composition is early, Character is middle, and Style is late. Turn off Auto Shift when you want to edit Start, End, and Peak manually.

## Recommended Defaults

- Optimization: `Balance`
- Combine mode: `Output average`
- Fusion mode: `Quality-safe delta`
- Global artist strength: `0.55`
- Default blocks: `6-21` for Balance, `10-18` for Performance, `4-23` for Quality.
- Use `Interpolate` with `0.6 - 0.8` only when you intentionally want the stronger behavior closest to the reference project.

## Notes

This cannot be perfectly lossless like SDXL artist mixing because Anima uses a non-linear LLM text encoder. The goal is controlled, usable style blending with lower interference between artist tags.

The extension patches Anima cross-attention only during sampling and restores the original forwards after generation. It does not edit text-encoder layers or model files.

## License

See [LICENSE](LICENSE) and [LICENSE_zh.md](LICENSE_zh.md).
