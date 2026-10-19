# Portfolio visuals

[English](README.md) | [한국어](README.ko.md)

The English SVG derivatives below correspond to the retained Korean presentation assets. Their values come from the retained experiment records referenced, with checksums, in `visual-data.json`. They do not change trained models, data splits, selection decisions or scientific reports. Read the main project documentation for scope and limitations. Existing source archives and the portfolio site are unchanged.

The renderer uses Python's standard library and the embedded display data only. Regenerating these visuals does not rerun a model or verify its scientific evaluation.

```bash
python docs/assets/portfolio/render_visuals.py --input docs/assets/portfolio/visual-data.json --output docs/assets/portfolio
python scripts/localize_public_figures.py --png
```

`visual-receipt.json` binds the generated SVG bytes to the display data. The source-download ZIP includes these exact site assets. The website's technical diagrams describe implemented processing steps; they are not new benchmark results.

## Figures

- [kd-thumbnail.svg](en/kd-thumbnail.svg)
- [kd-results.svg](en/kd-results.svg)
- [kd-costs.svg](en/kd-costs.svg)
- [kd-grid.svg](en/kd-grid.svg)
- [kd-errors.svg](en/kd-errors.svg)

The localization command reads the published SVGs, translates labels and exports the Korean chart PNGs with local Chrome/Chromium. It preserves plotted shapes and numeric values, uses no network or models, and writes `docs/figure-localization.json`. Pass `--chrome PATH` if Chrome is not found automatically. The original English scientific charts and Korean portfolio SVGs remain unchanged.
