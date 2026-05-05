# SwiftCAD project page

Static project page for **SwiftCAD: Efficient Parametric CAD Generation with
Shared Decoder Transformers** (CVPR 2026 Workshop 3D4S). This directory is
served via GitHub Pages.

## GitHub Pages configuration

In the repository on GitHub:

1. Go to **Settings -> Pages**.
2. Under **Build and deployment -> Source**, select **Deploy from a branch**.
3. Set **Branch** to `main` and **Folder** to `/docs`.
4. Click **Save**. After a minute the page will be live at
   `https://<your-github-username>.github.io/SwiftCAD/`.

## Local preview

The page uses only relative paths and CDN-hosted CSS, so it works either
opened directly via `file://` or served locally:

```bash
cd docs
python3 -m http.server 8000
# then open http://localhost:8000
```

## File layout

```
docs/
├── index.html                # main page
├── README.md                 # this file
└── static/
    ├── css/index.css         # custom styles (Bulma loaded from CDN)
    ├── js/index.js           # empty stub (no JS behavior)
    ├── images/               # paper figures (PNG)
    │   ├── teaser.png
    │   ├── figure_2_method.png
    │   ├── figure_3_qualitative.png
    │   ├── figure_4_attention.png
    │   └── figure_5_failure.png
    └── pdfs/
        └── README.md         # placeholder for swiftcad.pdf
```

The page loads Bulma 0.9.4, FontAwesome 6.5.1, and Academicons 1.9.4 from
public CDNs (jsDelivr / Cloudflare). No build step is required.

## TODOs

- **Paper PDF.** Drop the camera-ready PDF into `static/pdfs/swiftcad.pdf` and
  un-disable the "Paper" button in `index.html` (replace the `<span class="button is-rounded is-disabled">`
  with an `<a class="button is-rounded" href="static/pdfs/swiftcad.pdf">`).
- **arXiv link.** Once the preprint is posted on arXiv, replace the disabled
  arXiv `<span>` in `index.html` with a real `<a>` pointing to the arXiv page.
- **Author homepages.** If desired, wrap each author's name in
  `index.html` with a personal-page `<a href="...">` link.
- **Favicon.** Optional: drop a `favicon.ico` into `docs/` and add
  `<link rel="icon" href="favicon.ico">` to `<head>` in `index.html`.
