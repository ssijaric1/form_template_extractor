# Form-template extractor

Recovers the blank printed template from many phone scans of the same filled paper form (e.g. attendance registers), and can subtract that template from any scan to isolate just the handwriting. Everything runs locally — no images leave the machine.

## Setup

Python 3.10+. From the project folder, create a virtual environment:

```bash
# Windows:
python -m venv .venv

# macOS / Linux:
python3 -m venv .venv
```

Activate it:

```bash
# Windows (PowerShell or CMD):
.venv\Scripts\activate

# macOS / Linux:
source .venv/bin/activate
```

Then install:

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The browser opens automatically.

## Using the app

On the **Data** tab, either:

- upload your own scans (or a `.zip` of folders, one folder per form type), **or**
- download the bundled sample dataset from the card below the uploader and drop it back into the uploader, **or**
- click **Generate demo data** to render synthetic forms on the fly.

Then press **▶ Run this set**.

Walk through the remaining tabs:

1. **Cleaning report** — per-scan quality metrics, with overrides for which scans to include.
2. **Alignment** — registration quality across the stack.
3. **Template** — the extracted blank template, with sliders for the vote threshold, despeckle, signature-cleaning row coverage, bridge kernel, and border width. Downloads for `template.png`, `heatmap.png`, and `metrics.csv`.
4. **Inspect a scan** — pick any scan to see template (red) vs. its own handwriting (blue) side by side.

## Files

```
app.py            Streamlit UI
pipeline.py       extraction algorithms (importable, UI-independent)
synth.py          synthetic register generator
samples/          bundled sample dataset (downloadable from the UI)
requirements.txt  dependencies
```
