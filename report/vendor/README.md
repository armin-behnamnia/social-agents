# Vendored report assets

The report is a single self-contained HTML page *except* for three assets it
used to fetch from public CDNs. This machine has **no route to them** (verified:
`curl` to both hosts returns HTTP 000), so every `<canvas>` chart in the report
drew nothing — `Chart` was never defined — and the render-blocking Google Fonts
stylesheet stalled first paint. Both report generations and the generator
(`report/build_report.py`) now point at these local copies instead.

| file | what | licence | provenance |
|------|------|---------|------------|
| `chart.umd.min.js` | Chart.js **v4.4.7** (UMD build; sets the global `Chart` that the report's 12 `new Chart(...)` calls use) | MIT | already on this machine at `crates/openfang-api/static/vendor/chart.umd.min.js` (see the header comment it carries: *"Original file: /npm/chart.js@4.4.7/dist/chart.umd.js"*) |
| `fonts/LiberationMono-{Regular,Bold,Italic}.ttf` | Liberation Mono, for the report's `.mono` / tabular-numerals stack (previously remote "Monaspace Argon", which never loaded) | SIL OFL 1.1 | system package, `/usr/share/fonts/truetype/liberation/` |

Notes:

- The body serif is still declared as `'EB Garamond', Georgia, serif`. EB
  Garamond is not on this machine, so it falls back to Georgia — the same
  rendering the page had before, minus the failing request. The external
  stylesheet link is therefore gone rather than kept as a best-effort.
- Font stacks name the font that actually loads: the remote family name
  ("Monaspace Argon") was replaced by `'Liberation Mono'` everywhere rather
  than kept as an alias, so nothing claims to be a font it is not.
- `chart.umd.min.js` is committed rather than fetched at generation time
  because the machine it came from is an unrelated checkout; a report that
  cannot be regenerated offline is not much use here.
- To refresh: drop a newer `chart.umd.min.js` (must remain a UMD build exposing
  the global `Chart`) in this directory — no code change needed.
