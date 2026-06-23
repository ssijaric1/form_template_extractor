# Form-template extractor

Recovers the blank printed template from many phone scans of the same filled form (in this case it is attendance sheets), and can subtract that template from any scan to isolate just the handwriting.  
The app can be run locally, following the instructions below, or checked out at https://formtemplateextractor.streamlit.app/.  
The app provides a folder with three different forms. All of the scans were taken using a phone camera.

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

Then install the Python dependencies:

```bash
pip install -r requirements.txt
```

## System dependencies

The **signature auto-clean** feature uses Tesseract OCR to locate the "Signature" column header. 
If you skip this, the app still works, but you will need to adjust the signature bands manually via sliders.

- **Linux (Debian/Ubuntu):**
  ```bash
  sudo apt install tesseract-ocr tesseract-ocr-eng
  ```

- **macOS:**
  ```bash
  brew install tesseract
  ```

- **Windows:**
  Download and install from GitHub - UB-Mannheim/tesseract:
  https://github.com/UB-Mannheim/tesseract/wiki

  Then ensure `tesseract` is available in your PATH.

After installing the system binary, install the Python wrapper (it's already in `requirements.txt`, but you can run this manually):

```bash
pip install pytesseract
```

> **For Streamlit Cloud deployments**, system dependencies are handled automatically via the `packages.txt` file in this repository – no action needed.

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

1. **Cleaning report**: per-scan quality metrics, with overrides for which scans to include.
2. **Alignment**: registration quality across the stack.
3. **Template**: the extracted blank template, with sliders for the vote threshold, despeckle, signature-cleaning row coverage, bridge kernel, and border width. Downloads for `template.png`, `heatmap.png`, and `metrics.csv`.
4. **Inspect a scan** — pick any scan to see template (red) vs. its own handwriting (blue) side by side.

## Files

```text
app.py            Streamlit UI
pipeline.py       extraction algorithms (importable, UI-independent)
synth.py          synthetic register generator
samples/          bundled sample dataset (downloadable from the UI)
packages.txt      system packages for Streamlit Cloud (Tesseract)
requirements.txt  Python dependencies
```

