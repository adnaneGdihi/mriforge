# Public overlay

> This file lives **beside** `public_overlay/`, never inside it. Anything under that
> directory is a replacement keyed on its path, so a `README.md` at the overlay root
> overwrites the **project** README in the export. That is not hypothetical: the first
> version of this document sat at `public_overlay/README.md`, and the export dutifully
> published "# Public overlay" as the front page of spectraMR. The dead-overlay ratchet
> cannot catch it — `README.md` *is* a shipped path, so the substitution was valid,
> just not intended. Which is why the run summary now prints every overlaid path.

Files here **replace** their counterparts in the exported tree. The path under this
directory is the path in the export: `docs/index.rst` here becomes `docs/index.rst`
there.

**Replace-only.** An overlay file whose target is not already shipping replaces
nothing, and `export_public_tree.py --strict` reports it as a *dead overlay* and exits
2 — the same ratchet a dead allowance or a dead denial gets. This direction is
load-bearing: if the overlay could *add* a path it would become a second way to publish
a file, and `public_allowlist.txt` would stop being the single answer to "what ships".
The allowlist decides membership; the overlay decides only content.

## Why it exists

Some files are correct on `dev` and wrong in the distribution, and editing them on
`dev` to suit the export would redden this branch. Before this directory existed the
substitution was made by hand in the published repository — which is how a 105-line
`docs/index.rst` came to live there and nowhere else, invisible here and destined to be
destroyed by the next export.

## What is in it

| File | Why |
|---|---|
| `docs/index.rst` | The published site's `toctree` must name only pages that ship. This tree's index also reaches `docs/api/` (175 generated pages), `docs/contributing/` and `cluster_verification` — none of which are in the public curation. Two sidebars for two sites is not one invariant with two owners; it is two documents. Sphinx runs with `-W` on Read the Docs, so a `toctree` entry naming an absent page is a build failure, not a warning. |

## Adding one

1. Put the file at its export-relative path here.
2. Re-export with `--strict` and confirm `overlaid : N` counts it.
3. If it reports `DEAD OVERLAY`, the target is not shipping — fix the **allowlist**,
   not the overlay.
