# Example NeurIPS paper

An **example** paper generated from the `Data_Analysis` pipeline. All numbers are real
outputs (a 400-case pilot, single Llama extractor/judge); **references are illustrative and
must be completed** before any real use.

## Files
- `main.tex` — the paper (NeurIPS 2024 format, `\usepackage[preprint]{neurips_2024}`).
- `figures/` — the seven figures referenced by `main.tex` (from `outputs/`).
- `example_neurips_paper.md` — the same content in Markdown (easier to read/skim).

## Open in Overleaf (2 minutes)
`main.tex` needs the official style file `neurips_2024.sty`, which isn't included here.

**Easiest path:**
1. In Overleaf, create a project from the **"NeurIPS 2024"** template (Templates → search
   "NeurIPS 2024"). It ships with `neurips_2024.sty` (and `.bst`).
2. Replace that project's `main.tex` with **this** `main.tex`.
3. Upload the **`figures/`** folder into the project.
4. Recompile (pdfLaTeX). Done.

**Alternative:** download `neurips_2024.sty` from the NeurIPS website, drop it beside
`main.tex` + `figures/`, upload the whole `paper/` folder to Overleaf, compile.

## Notes
- Compiles in **preprint** mode (authors visible, no submission number). For the anonymized
  submission, remove `[preprint]` from the `\usepackage{neurips_2024}` line.
- References use a manual `thebibliography` (no `.bib` needed). Swap in a real `.bib` +
  `\bibliography{...}` when you complete the citations.
- Figures are `.png`; regenerate any of them by rerunning the relevant Part-2/Part-3 script
  (see the top-level `README`) and re-copying from `outputs/.../figures/`.
