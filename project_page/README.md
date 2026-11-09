# Mercy project page

A static academic project page built with HTML, CSS and vanilla JavaScript, adapted from the requested [NEXUS4SCI-Mat template](https://github.com/SuleynanAuir/NEXUS4SCI-Mat-Project-Page). Technical content follows `ppt_all.pdf` and the supplied assets.

## Local preview

From the repository root:

```sh
python3 project_page/scripts/build_site.py --output /tmp/mercy-preview
python3 -m http.server 8000 --directory /tmp/mercy-preview
```

Open `http://localhost:8000`. Choose a new or empty output directory. No npm install is required.

## Files

- `index.html`: project content, semantic results table and nine-object error-map grid.
- `styles.css`: responsive academic layout, matching the reference palette and typography.
- `script.js`: accessible image dialog, keyboard navigation and GIF playback control.
- `assets/web/`: optimized public figures, animated GIF, table data and visualization mapping.
- `scripts/build_site.py`: validates local links and packages only the required public files.
- `../.github/workflows/mercy-pages.yml`: GitHub Actions deployment to GitHub Pages.

The page works without JavaScript: the table and figures remain visible, full-size figures open as ordinary links, and the GIF can be downloaded. JavaScript adds the dialog and controlled animation. The 4K MP4 autoplays muted and loops inline, with a pause control. The original full-resolution GIF remains downloadable.

## GitHub Pages

The target address for this repository is:

`https://suleynanauir.github.io/MERCY-Monocular-Event-Locally-RigidMotion-Coherent-4DGS-Reconstruct/`

In repository **Settings → Pages**, choose **GitHub Actions** as the publishing source. The workflow deploys when `project_page/` changes on `main`, or when manually started from Actions. It publishes only the output of the packaging script, not the repository root or unrelated experiment files.

To validate and package locally, use a new or empty output directory:

```sh
node --check project_page/script.js
python3 project_page/scripts/build_site.py --output /tmp/mercy-site
```

All page URLs are relative so the site works under a GitHub repository subpath. If hosting under a different repository, update the repository links and `og:image` URL in `index.html`.

## Scientific content

The table is a transcription of PDF page 83 and `assets/tables/01_reconstruction.png`. `assets/web/results.json` contains rows in the order displayed, with columns: object, three PSNR values, three SSIM values, three LPIPS values, and three IoU values. Within each metric, the method order is MaTGS, MonoGS, Mercy.

The summary gain of +6.75 dB is the difference between object-wise mean PSNR values. The −69.2% LPIPS change is computed from the two object-wise means. Neither is a new evaluation.

Error previews retain the first ranked frame from each original `05_error_maps.png`. Four panels are displayed as a 2×2 group inside each of the nine object tiles. Clicking a tile opens the unmodified complete figure, including four sampled frames and foreground CDF/tail plots. The grid retains the shared RGB error and V3−V2 difference scales. Method names follow the supplied PDF rather than legacy labels in report metadata.

The downloadable GIF is the unmodified 3840 × 2160 original. Inline playback uses a fast-start H.264 MP4 at the same resolution, preserving the original 150 frames at 12.5 FPS over 12 seconds, without synthesized motion or sped-up playback. This substantially reduces transfer and GIF decoding overhead; the page provides a pause control and still poster. On-image reference-camera diagnostics include fitting/selection and are not presented as an independent benchmark.

The full-resolution GIF is stored as three binary `.part` files with a `.parts.json` manifest to accommodate upload limits. The packaging script verifies their SHA-256 checksum and size, then restores the exact original GIF. Only the restored GIF is published; no browser-side assembly is needed.

The header uses two explicit title lines, with the M / E / R / C / y letters emphasized. Keywords stay in one row (horizontally scrollable on narrow devices). The square nine-object grid scales to viewport height and retains three columns on mobile; use the existing full-image dialog to inspect details.

The first page aligns the supplied wide motion pipeline on the left with the animation on the right; a color-emphasized overview sits beneath the pipeline. Author names include mailto contacts. Section navigation has been removed; small screens use stacked content. A short explanation sits below the animation.

Tables 1, 2 and 5 use the supplied high-resolution PNGs in `assets/tables/` to exactly preserve publication typography, rules, shading and numbers. All open in the figure dialog. Table 1 also retains its semantic text table for screen readers. The three summary cards report unchanged values.
