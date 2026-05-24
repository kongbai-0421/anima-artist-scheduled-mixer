# Anima Artist Scheduled Mixer

Anima Artist Scheduled Mixer is a Forge Neo extension for mixing Anima artist tags in a way closer to SDXL-style artist chains.

Instead of asking Anima's LLM text encoder to understand several artist names in one contextualized prompt, the extension encodes every artist row independently and mixes those artist conditionings inside Anima cross-attention during sampling. This follows the same core idea as [An1X3R/Anima-Artist-Mixer](https://github.com/An1X3R/Anima-Artist-Mixer): split artists, encode separately, then blend at cross-attention instead of modifying text-encoder layers.

Special thanks to **An1X3R/Anima-Artist-Mixer** and **汐浮尘/utowo** for the original split-and-encode and cross-attention mixing design.

## Features

- Per-artist rows with artist tag, weight, block range, denoise time range, peak, curve, stage preset, and Shift value.
- Separate Base and Hires. fix artist chains. Hires. fix is disabled by default and inherits Base settings.
- Template save, apply, rename, and delete.
- Performance, Balance, and Quality optimization presets. Balance is the default.
- English/Chinese UI setting, plus an English-first description switch at the top of the panel.
- Text-encoding cache for repeated prompts and artist rows.
- Cross-attention output averaging with batched artist forwards when possible.

## Recommended Defaults

- Optimization: `Balance`
- Combine mode: `output_avg`
- Fusion mode: `interpolate`
- Global artist strength: `0.6 - 0.8`
- Use middle blocks for character identity and late blocks for style.

## Notes

This cannot be perfectly lossless like SDXL artist mixing because Anima uses a non-linear LLM text encoder. The goal is controlled, usable style blending with lower interference between artist tags.

The extension patches Anima cross-attention only during sampling and restores the original forwards after generation. It does not edit text-encoder layers or model files.

## License

See [LICENSE](LICENSE) and [LICENSE_zh.md](LICENSE_zh.md).
