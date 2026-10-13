# Portfolio visuals

[English](README.md) | [한국어](README.ko.md)

These Korean-labelled SVGs are the presentation assets used on the portfolio site. Their values come from the retained experiment records referenced, with checksums, in `visual-data.json`. They do not change trained models, data splits, selection decisions or scientific reports. Read the main project documentation for scope and limitations.

The renderer uses Python's standard library and the embedded display data only. Regenerating these visuals does not rerun a model or verify its scientific evaluation.

```bash
python docs/assets/portfolio/render_visuals.py --input docs/assets/portfolio/visual-data.json --output docs/assets/portfolio
```

`visual-receipt.json` binds the generated SVG bytes to the display data. The source-download ZIP includes these exact site assets. The website's technical diagrams describe implemented processing steps; they are not new benchmark results.

## Figures

- [kd-thumbnail.svg](kd-thumbnail.svg)
- [kd-results.svg](kd-results.svg)
- [kd-costs.svg](kd-costs.svg)
- [kd-grid.svg](kd-grid.svg)
- [kd-errors.svg](kd-errors.svg)
