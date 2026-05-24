# Anima 画师串调度混合

Anima 画师串调度混合是一个 Forge Neo 扩展，用来让 Anima 的画师标签更接近 SDXL 画师串的混合方式。

它不会把多个画师名直接交给 Anima 的 LLM 文本编码器一起上下文化理解，而是把每个画师行独立编码，再在采样时把这些画师条件注入并混合到 Anima 的 cross-attention 中。核心思路参考 [An1X3R/Anima-Artist-Mixer](https://github.com/An1X3R/Anima-Artist-Mixer)：拆分画师、独立编码、在 cross-attention 混合，而不是修改文本编码器层。

特别感谢 **An1X3R/Anima-Artist-Mixer** 与 **汐浮尘/utowo** 的原始独立编码和 cross-attention 混合设计。

## 功能

- 每个画师独立一行，可设置画师标签、权重、介入 block、介入时间、峰值、曲线、阶段预设和 Shift。
- 底图与高分辨率修复画师串分离。高分辨率修复默认关闭独立设置，关闭时继承底图。
- 模板保存、应用、重命名和删除。
- 性能、平衡、质量三个优化预设，默认平衡。
- 设置里支持中英文 UI，面板顶部默认英文说明并可切换中文。
- 文本编码缓存，重复画师和提示词时减少编码开销。
- 默认使用 `output_avg + interpolate`，可在可行时批量并行多个画师的 cross-attention forward。

## 推荐默认值

- 优化预设：`Balance`
- 组合模式：`output_avg`
- 融合模式：`interpolate`
- 全局画师强度：`0.6 - 0.8`
- 人物特征优先使用中层 block，画风优先使用偏后 block。

## 注意

Anima 使用非线性的 LLM 文本编码器，因此无法做到 SDXL 画师混合那种近似无损。这个插件的目标是降低画师标签之间的上下文干扰，让风格混合更可控。

插件只在采样期间包装 Anima cross-attention，生成结束后会还原原始 forward；不会修改文本编码器层或模型文件。

## 许可

见 [LICENSE](LICENSE) 与 [LICENSE_zh.md](LICENSE_zh.md)。
