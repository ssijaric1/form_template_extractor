from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import pipeline as pl
import synth

st.set_page_config(page_title="Form template extractor", page_icon="📄",
                   layout="wide")

st.markdown("""
<style>
  [data-testid="stHeaderActionElements"] { display: none !important; }
  h1 a, h2 a, h3 a, h4 a, h5 a, h6 a { display: none !important; }

  .stApp, .stApp > header, .stApp > footer {
    background-color: #112324;
    color: #e8e5dd;
  }

  [data-testid="stTabs"] {
      background: #112324;
      margin: 0 !important;
      padding: 0 !important;
  }

  [data-testid="stTabs"] > div:first-child {
      margin: 0 !important;
      padding: 0 !important;
      background: #112324;
  }

  [data-testid="stTabs"] [data-baseweb="tab-list"] {
      display: flex;
      gap: 6px;
      padding: 0 6px !important;
      margin: 0 !important;
      border: none !important;
      background: transparent !important;
      position: relative !important;
      z-index: 2 !important;
      overflow: visible !important;
  }

  [data-testid="stTabs"] [data-baseweb="tab-highlight"],
  [data-testid="stTabs"] [data-baseweb="tab-border"] {
      display: none !important;
  }

  [data-testid="stTabs"] [data-baseweb="tab"] {
      background: #112324;
      color: #8a9e8a;
      border: 1px solid transparent !important;
      border-bottom: none !important;
      border-radius: 8px 8px 0 0 !important;
      padding: 8px 18px;
      margin: 0 !important;
      transition: 0.2s;
  }

  [data-testid="stTabs"] [data-baseweb="tab"]:hover {
      background: #1a3a3a;
      color: #e8e5dd;
      border-color: #3a553a !important;
  }

  [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
      background: #1a3a3a;
      color: #f5f0e8;
      border: 1px solid #668479 !important;
      border-bottom: none !important;
      margin-bottom: 0 !important;
      position: relative !important;
      z-index: 10 !important;
      overflow: visible !important;
  }

  [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"]::after {
      content: "";
      position: absolute;
      left: 0;
      right: 0;
      bottom: -2px;
      height: 4px;
      background: #1a3a3a;
      z-index: 10;
  }

  [data-testid="stTabs"] [data-baseweb="tab-panel"] {
      background: #1a3a3a;
      border: 1px solid #668479 !important;
      margin: 0 !important;
      padding: 1.2rem 1.5rem !important;
      position: relative !important;
      z-index: 1 !important;
  }

  [data-testid="stTabs"] [data-baseweb="tab-panel"] > div {
      margin: 0 !important;
      padding: 0 !important;
  }

  [data-testid="stTabs"] h2,
  [data-testid="stTabs"] h3,
  [data-testid="stTabs"] h4,
  [data-testid="stTabs"] .stMarkdown,
  [data-testid="stTabs"] .stCaption,
  [data-testid="stTabs"] label,
  [data-testid="stTabs"] .stMetric label,
  [data-testid="stTabs"] .stMetric .stMetricValue {
    color: #e8e5dd !important;
  }
  [data-testid="stTabs"] [data-testid="stMetric"] {
    background: #112324;
    border: 1px solid #2a4a3a;
  }
  [data-testid="stTabs"] [data-testid="stMetric"] label {
    color: #b8c9ad !important;
  }
  [data-testid="stTabs"] [data-testid="stMetric"] .stMetricValue {
    color: #f5f0e8 !important;
  }
  [data-testid="stTabs"] .stDataFrame,
  [data-testid="stTabs"] .stTable {
    background: #112324;
  }
  [data-testid="stTabs"] .stDataFrame thead th,
  [data-testid="stTabs"] .stTable thead th {
    background: #1a3a3a;
    color: #e8e5dd;
  }
  [data-testid="stTabs"] .stDataFrame tbody td,
  [data-testid="stTabs"] .stTable tbody td {
    color: #d0d8c8;
  }
  [data-testid="stTabs"] .stButton > button,
  [data-testid="stTabs"] .stDownloadButton > button {
    background: #668479;
    color: #112324;
    border: none;
    font-weight: 600;
    padding: 0.4rem 1rem;
  }
  [data-testid="stTabs"] .stButton > button:hover,
  [data-testid="stTabs"] .stDownloadButton > button:hover {
    background: #557a6e;
    color: #000;
  }
  [data-testid="stTabs"] .stSlider > div > div > div > div {
    background: #2a4a3a;
  }
  [data-testid="stTabs"] .stSlider > div > div > div > div > div {
    background: #668479;
  }
  [data-testid="stTabs"] .stNumberInput input,
  [data-testid="stTabs"] .stTextInput input,
  [data-testid="stTabs"] .stSelectbox select {
    background: #112324;
    border: 1px solid #2a4a3a;
    color: #e8e5dd;
  }
  [data-testid="stTabs"] .stNumberInput input:focus,
  [data-testid="stTabs"] .stTextInput input:focus {
    border-color: #668479;
  }
  [data-testid="stTabs"] .stFileUploader {
    background: #112324;
    border: 2px dashed #2a4a3a;
  }
  [data-testid="stTabs"] .stFileUploader:hover {
    border-color: #668479;
  }
  [data-testid="stTabs"] [data-testid="stImage"] img {
    border: 1px solid #2a4a3a;
  }
  [data-testid="stTabs"] .streamlit-expanderHeader {
    background: #112324;
    color: #e8e5dd;
    border: 1px solid #2a4a3a;
  }
  [data-testid="stTabs"] .streamlit-expanderHeader:hover {
    background: #1a3a3a;
  }
  [data-testid="stTabs"] .streamlit-expanderContent {
    background: #112324;
    border: 1px solid #2a4a3a;
    border-top: none;
  }
  [data-testid="stTabs"] .stAlert {
    background: #112324;
    border-left: 4px solid #668479;
    color: #e8e5dd;
  }
  [data-testid="stTabs"] .stAlert svg {
    fill: #668479 !important;
  }

  /* Bordered containers: sharp edges, visible border, dark-green like the page */
  [data-testid="stVerticalBlockBorderWrapper"],
  [data-testid="stVerticalBlockBorderWrapper"] * {
    border-radius: 0 !important;
  }
  [data-testid="stVerticalBlockBorderWrapper"] {
    background: #112324 !important;
    border: 1px solid #668479 !important;
  }

  .stButton > button, .stDownloadButton > button {
    background: #668479;
    color: #112324;
    border: none;
    font-weight: 600;
    padding: 0.4rem 1rem;
  }
  .stButton > button:hover, .stDownloadButton > button:hover {
    background: #557a6e;
    color: #000;
    box-shadow: 0 2px 8px rgba(102, 132, 121, 0.3);
  }
  .stButton > button[kind="primary"] {
    background: #668479;
    color: #112324;
  }
  .stButton > button[kind="primary"]:hover {
    background: #557a6e;
  }
  [data-testid="stMetric"] {
    background: #112324;
    padding: 1rem;
    border: 1px solid #2a4a3a;
  }
  [data-testid="stMetric"] label {
    color: #b8c9ad !important;
  }
  [data-testid="stMetric"] .stMetricValue {
    color: #f5f0e8 !important;
  }
  .stDataFrame, .stTable {
    background: #112324;
  }
  .stDataFrame thead th, .stTable thead th {
    background: #1a3a3a;
    color: #e8e5dd;
  }
  .stDataFrame tbody td, .stTable tbody td {
    color: #d0d8c8;
  }
  .stSlider > div > div > div > div {
    background: #2a4a3a;
  }
  .stSlider > div > div > div > div > div {
    background: #668479;
  }
  .stNumberInput input, .stTextInput input, .stSelectbox select {
    background: #112324;
    border: 1px solid #2a4a3a;
    color: #e8e5dd;
  }
  .stNumberInput input:focus, .stTextInput input:focus {
    border-color: #668479;
  }
  .stFileUploader {
    background: #112324;
    border: 2px dashed #2a4a3a;
  }
  .stFileUploader:hover {
    border-color: #668479;
  }
  [data-testid="stImage"] img {
    border: 1px solid #2a4a3a;
  }
  .streamlit-expanderHeader {
    background: #112324;
    color: #e8e5dd;
    border: 1px solid #2a4a3a;
  }
  .streamlit-expanderHeader:hover {
    background: #1a3a3a;
  }
  .streamlit-expanderContent {
    background: #112324;
    border: 1px solid #2a4a3a;
    border-top: none;
  }
  .stProgress > div > div > div {
    background-color: #668479 !important;
  }
  .stAlert {
    background: #112324;
    border-left: 4px solid #668479;
    color: #e8e5dd;
  }
  .stAlert svg {
    fill: #668479 !important;
  }
  .stCaption, .stMarkdown small {
    color: #8a9e8a;
  }
  hr {
    border-color: #2a4a3a;
    margin: 1.5rem 0;
  }
  [data-testid="stSidebar"] {
    background: #112324;
    border-right: 1px solid #2a4a3a;
  }
  [data-testid="stSidebar"] .stMarkdown {
    color: #e8e5dd;
  }
  ::-webkit-scrollbar {
    width: 6px;
  }
  ::-webkit-scrollbar-track {
    background: #112324;
  }
  ::-webkit-scrollbar-thumb {
    background: #2a4a3a;
  }
  ::-webkit-scrollbar-thumb:hover {
    background: #668479;
  }
</style>
""", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("demo_items", None)
ss.setdefault("res", None)
ss.setdefault("all_items", None)
ss.setdefault("include_map", {})
ss.setdefault("inspect_cache", {})
ss.setdefault("groups", {})
ss.setdefault("results", {})
ss.setdefault("items_by_group", {})
ss.setdefault("active_group", None)
ss.setdefault("_shown_group", None)
ss.setdefault("uploaded_groups", {})
ss.setdefault("demo_groups", {})
ss.setdefault("data_source", "Uploaded")

def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", str(s))]

run_params = {}

def thumb(data: bytes, width: int = 260) -> np.ndarray | None:
    img = pl.imdecode_bytes(data)
    if img is None:
        return None
    img = pl.resize_max(img, width)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

_IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")

def _parse_zip(data: bytes) -> dict[str, list[tuple[str, bytes]]]:
    out: dict[str, list[tuple[str, bytes]]] = {}
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return out
    with zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            path = info.filename.replace("\\", "/")
            if "__MACOSX/" in path:
                continue
            parts = path.strip("/").split("/")
            base = parts[-1]
            if not base or base.startswith(".") \
                    or not base.lower().endswith(_IMG_EXT):
                continue
            group = parts[-2] if len(parts) >= 2 else "(root)"
            out.setdefault(group, []).append((base, zf.read(info)))
    return out

def groups_from_uploads(uploads) -> dict[str, list[tuple[str, bytes]]]:
    groups: dict[str, list[tuple[str, bytes]]] = {}
    for f in uploads:
        name = f.name
        if name.lower().endswith(".zip"):
            for g, items in _parse_zip(f.getvalue()).items():
                groups.setdefault(g, []).extend(items)
        elif name.lower().endswith(_IMG_EXT):
            groups.setdefault("uploaded", []).append((name, f.getvalue()))
    return groups

def _activate(name: str):
    ss.active_group = name
    ss._shown_group = name
    ss.res = ss.results.get(name)
    ss.all_items = ss.items_by_group.get(name)
    ss.inspect_cache = {}

def _run_items(name: str, items: list[tuple[str, bytes]]):
    bar = st.progress(0.0, text=f"Starting {name} ...")
    def tick(stage, i, n):
        bar.progress(min(i / max(n, 1), 1.0),
                     text=f"[{name}] {stage} ({i}/{n})")
    res = pl.run_pipeline(items, run_params, progress=tick)
    bar.empty()
    ss.results[name] = res
    ss.items_by_group[name] = items
    _activate(name)
    return res

def run_now(items: list[tuple[str, bytes]]):
    name = ss.active_group or "uploaded"
    res = _run_items(name, items)
    pick = res.get("auto_pick")
    pick_note = ""
    if pick:
        ct, co = res.get("auto_scores") or (None, None)
        pick_note = (f" Auto-chose **{pick}** alignment "
                     f"(table {ct} vs orb {co} vote concentration).")
    st.success(f"[{name}] {res['n_used']} of {len(items)} scans aligned and "
               f"used. Reference: {res['ref_name']}.{pick_note}")

def run_group(name: str):
    res = _run_items(name, ss.groups[name])
    st.success(f"[{name}] {res['n_used']} of {len(ss.groups[name])} scans "
               f"aligned and used. Reference: {res['ref_name']}")

def run_all_groups():
    names = sorted(ss.groups, key=_natural_key)
    for name in names:
        if len(ss.groups[name]) >= 3:
            _run_items(name, ss.groups[name])
    if names:
        _activate(names[0])
    st.success(f"Processed {len(ss.results)} template set(s). Pick one in the "
               "selector to view it.")

def current_template():
    res = ss.res
    return pl.extract_template(res["freq"], res["coverage"],
                               ss.get("thr", pl.DEFAULT_PARAMS["vote_threshold"]),
                               ss.get("min_cov", pl.DEFAULT_PARAMS["min_coverage"]),
                               ss.get("min_blob", pl.DEFAULT_PARAMS["min_blob"]),
                               ss.get("bridge", pl.DEFAULT_PARAMS["bridge"]),
                               res["params"]["tolerance"],
                               low_threshold=0.0,
                               context_gate=ss.get("ctx", 0.0),
                               grid_reconstruct=ss.get("grid", pl.DEFAULT_PARAMS["grid_reconstruct"]),
                               grid_low=pl.DEFAULT_PARAMS["grid_low"],
                               grid_thickness=pl.DEFAULT_PARAMS["grid_thickness"],
                               signature_bands=(res.get("signature_bands", [])
                                                if ss.get("auto_sig", True) else []),
                               sig_clean_threshold=ss.get("sig_row_thr", 0.90),
                               sig_bridge_ratio=ss.get("sig_bridge_div", 5),
                               sig_border_divisor=ss.get("sig_border_div", 80))

def inspect(name: str):
    key = (name, repr(sorted(ss.res["params"].items())))
    if key in ss.inspect_cache:
        return ss.inspect_cache[key]
    items = dict(ss.all_items)
    ref_name = ss.res["ref_name"]
    with st.spinner(f"Re-aligning {name} ..."):
        out = pl.realign_one(name, items[name], ref_name, items[ref_name],
                             ss.res["params"])
    if len(ss.inspect_cache) > 6:
        ss.inspect_cache.pop(next(iter(ss.inspect_cache)))
    ss.inspect_cache[key] = out
    return out

st.title("Blank-template extraction from filled forms")

tabs = st.tabs(["1 · Data", "2 · Cleaning report", "3 · Alignment",
                "4 · Template", "5 · Inspect a scan"])

with tabs[0]:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Load scans")
        uploads = st.file_uploader(
            "Phone scans of the filled forms, or a .zip with multiple form types.",
            type=["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp", "zip"],
            accept_multiple_files=True)
        st.caption("Each subfolder is detected as a separate template and run "
                   "on its own. Loose images count as one template.")

        sample_path = Path(__file__).parent / "samples" / "dataset.zip"
        if sample_path.exists():
            with st.container(border=True):
                st.markdown("**Sample dataset**")
                st.caption("A dataset of scanned attendance sheets from our "
                           " University. All default parameters are tuned for these specific templates.")
                st.download_button(
                    "⬇ Download dataset.zip",
                    data=sample_path.read_bytes(),
                    file_name="dataset.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
        else:
            st.caption("(Place a `samples/dataset.zip` in the repo to expose "
                       "a downloadable sample dataset here.)")

        st.markdown("**or**")
        c1, c2 = st.columns([2, 1])
        n_demo = c1.slider("Synthetic demo forms", 6, 40, 14)
        if c2.button("Generate demo data", use_container_width=True):
            with st.spinner("Rendering synthetic registers ..."):
                ss.demo_items = synth.make_dataset_bytes(n_demo)
                ss.data_source = "Synthetic"
                ss.results, ss.items_by_group = {}, {}
                ss.res, ss.active_group, ss._shown_group = None, None, None
                st.rerun()

    with right:
        st.subheader("Run")

        with st.container(border=True):
            st.markdown("**Which dataset do you want to use?**")
            source = st.radio(
                "",
                ["Uploaded", "Synthetic"],
                index=0 if ss.get("data_source", "Uploaded") == "Uploaded" else 1,
                key="data_source_radio_right",
                horizontal=True,
                label_visibility="collapsed"
            )
            if source != ss.get("data_source"):
                ss.data_source = source
                ss.results, ss.items_by_group = {}, {}
                ss.res, ss.active_group, ss._shown_group = None, None, None
                st.rerun()

        if ss.data_source == "Uploaded":
            if uploads:
                ss.uploaded_groups = groups_from_uploads(uploads)
            ss.groups = ss.uploaded_groups if ss.uploaded_groups else {}
        else:
            if ss.demo_items:
                ss.groups = {"demo": ss.demo_items}
            else:
                ss.groups = {}
                st.info("No synthetic data generated yet. Click 'Generate demo data' on the left.")

        group_names = sorted(ss.groups, key=_natural_key)
        if not group_names:
            st.info("Load scans, a .zip of form folders, or generate demo data on the left.")
        else:
            if len(group_names) > 1:
                st.success(f"Detected {len(group_names)} template sets.")
                st.table({"template set": group_names,
                          "scans": [len(ss.groups[n]) for n in group_names]})
                if ("group_selector" not in ss
                        or ss.group_selector not in group_names):
                    ss.group_selector = ss.active_group
                chosen = st.selectbox("View / run which set", group_names,
                                      key="group_selector")
            else:
                chosen = group_names[0]

            ss.active_group = chosen
            if chosen != ss._shown_group:
                ss.res = ss.results.get(chosen)
                ss.all_items = ss.items_by_group.get(chosen)
                ss.inspect_cache = {}
                ss._shown_group = chosen

            items = ss.groups[chosen]
            disabled = len(items) < 3
            b1, b2 = st.columns(2)
            if b1.button("▶ Run this set", type="primary",
                         use_container_width=True, disabled=disabled):
                run_group(chosen)
            if len(group_names) > 1:
                all_disabled = any(len(v) < 3 for v in ss.groups.values())
                if b2.button("▶▶ Run all sets", use_container_width=True,
                             disabled=all_disabled):
                    run_all_groups()
            if disabled:
                st.caption("This set has fewer than 3 scans — add more for a "
                           "reliable vote.")
            done = [n for n in group_names if ss.results.get(n)]
            if done:
                st.caption("Processed: " + ", ".join(done))

    active_items = ss.groups.get(ss.active_group, []) if group_names else []
    if active_items:
        st.divider()
        st.subheader(f"Preview — {ss.active_group}")
        cols = st.columns(6)
        for i, (name, data) in enumerate(active_items[:12]):
            t = thumb(data)
            if t is not None:
                cols[i % 6].image(t, caption=name)

with tabs[1]:
    st.subheader("Data cleaning & preparation")
    if ss.res is None:
        st.info("Run the pipeline on the Data tab first.")
    else:
        df = ss.res["metrics"].copy()
        c1, c2, c3 = st.columns(3)
        blur_thr = c1.number_input("Flag if blur score below", 0.0, 5000.0,
                                   60.0, 10.0,
                                   help="Variance of the Laplacian — low "
                                        "means out of focus.")
        ssim_thr = c2.number_input("Flag if SSIM below", 0.0, 1.0, 0.35, 0.05,
                                   help="Structural similarity to the "
                                        "reference after alignment.")
        apply_flags = c3.checkbox("Untick flagged scans automatically", True)

        flags = []
        for _, r in df.iterrows():
            f = []
            if r["status"] == "rejected":
                f.append(r["reason"])
            if r["blur_var"] < blur_thr:
                f.append("blurry")
            if pd.notna(r["ssim"]) and r["ssim"] < ssim_thr \
                    and r["role"] != "reference":
                f.append("low similarity")
            flags.append(", ".join(f))
        df["flags"] = flags
        default_include = df["status"] == "included"
        if apply_flags:
            default_include &= df["flags"] == ""
        df["include"] = default_include

        show_cols = ["include", "name", "role", "method", "status", "flags",
                     "blur_var", "brightness", "contrast", "ink_ratio", "ssim",
                     "grid_iou", "inliers", "reproj_err", "page_warped",
                     "width", "height"]
        show_cols = [c for c in show_cols if c in df.columns]
        edited = st.data_editor(df[show_cols], hide_index=True,
                                use_container_width=True, key="clean_editor",
                                disabled=[c for c in show_cols
                                          if c != "include"])
        ss.include_map = dict(zip(edited["name"], edited["include"]))

        kept = int(edited["include"].sum())
        c1, c2 = st.columns([1, 3])
        if c1.button(f"Re-run with {kept} selected scans",
                     disabled=kept < 3):
            chosen = [it for it in ss.all_items
                      if ss.include_map.get(it[0], False)]
            run_now(chosen)
            st.rerun()
        c2.caption("Rejected scans (alignment failed / unreadable) are "
                   "excluded automatically; flags are suggestions you can "
                   "override before re-running.")

        st.divider()
        g1, g2 = st.columns(2)
        g1.markdown("**Sharpness per scan** (variance of Laplacian)")
        g1.bar_chart(df.set_index("name")["blur_var"])
        g2.markdown("**Similarity to reference after alignment** (SSIM)")
        g2.bar_chart(df.set_index("name")["ssim"])

with tabs[2]:
    st.subheader("Registration quality")
    if ss.res is None:
        st.info("Run the pipeline on the Data tab first.")
    else:
        df = ss.res["metrics"]
        ok = df[df["status"] == "included"]
        has_grid = df["grid_iou"].notna().any() if "grid_iou" in df else False
        methods = (df[df["role"] == "scan"]["method"].value_counts().to_dict()
                   if "method" in df else {})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Scans aligned", f"{len(ok)}/{len(df)}")
        if has_grid:
            c2.metric("Median grid overlap",
                      f"{ok['grid_iou'].dropna().median():.3f}")
        else:
            c2.metric("Median inliers", int(ok["inliers"].median()))
        c3.metric("Median SSIM", f"{ok['ssim'].median():.3f}")
        n_table = methods.get("table", 0)
        n_fallback = methods.get("table->orb", 0) + methods.get("orb", 0)
        c4.metric("Table-aligned", f"{n_table} (ORB: {n_fallback})")

        if has_grid:
            st.markdown("**Grid overlap per scan** (IoU of printed rule lines "
                        "with the reference) — higher = better registration.")
            st.bar_chart(df.set_index("name")["grid_iou"])
        else:
            st.markdown("**RANSAC inliers per scan**: more inliers = more "
                        "printed structure matched.")
            st.bar_chart(df.set_index("name")["inliers"])

        bad = df[df["status"] == "rejected"]
        if len(bad):
            st.warning("Rejected: " + "; ".join(
                f"{r['name']} ({r['reason']})" for _, r in bad.iterrows()))
        if n_fallback and n_table:
            st.caption(f"{n_fallback} scan(s) fell back to ORB feature "
                       "matching because no clear table frame was found — "
                       "that's expected and handled automatically.")
        st.caption("Each scan is registered onto the sharpest "
                   "one, either by mapping the table's four corners onto a "
                   "common upright frame or by ORB keypoints + RANSAC "
                   "homography — whichever stacks the printed structure more "
                   "tightly (chosen automatically per form set). A final ECC "
                   "locks the rule lines together.")

with tabs[3]:
    st.subheader("Extracted template")
    if ss.res is None:
        st.info("Run the pipeline on the Data tab first.")
    else:
        res = ss.res
        st.caption(f"Template set: **{ss.active_group}**")
        c1, c2, c3 = st.columns(3)
        gl = (ss.active_group or "").lower()
        if "type1" in gl:
            def_thr, def_blob = 0.50, 60
        elif "type2" in gl:
            def_thr = 0.55
            def_blob = pl.DEFAULT_PARAMS["min_blob"]
        elif "type3" in gl:
            def_thr = 0.55
            def_blob = pl.DEFAULT_PARAMS["min_blob"]
        elif "demo" in gl:
            def_thr, def_blob = 0.60, 32
        else:
            def_thr = pl.DEFAULT_PARAMS["vote_threshold"]
            def_blob = pl.DEFAULT_PARAMS["min_blob"]
        ss.thr = c1.slider("Vote threshold (ink frequency)", 0.30, 0.95,
                           def_thr, 0.05, key=f"thr_{ss.active_group}",
                           help="A pixel becomes template if it is ink in at "
                                "least this fraction of aligned scans.")
        ss.min_blob = c2.slider("Despeckle: min component size (px)", 0, 60,
                                def_blob, 1, key=f"blob_{ss.active_group}")
        ss.bridge = c3.slider("Bridge broken lines (px)", 0, 8,
                              pl.DEFAULT_PARAMS["bridge"], 1,
                              key=f"bridge_{ss.active_group}")

        ss.ctx = st.slider("Remove ink ringed by handwriting (context gate)",
                           0.0, 0.9, 0.0, 0.05,
                           help="Your 'slim red with lots of blue around it' "
                                "cleaner. Drops a small ink blob if the thin "
                                "ring just outside it is mostly handwriting — "
                                "so a printed number/letter in its own empty "
                                "cell stays, but a signature fragment buried in "
                                "writing goes. 0 = off; 0.6–0.7 works well.")

        ocr_ok, ocr_msg = pl.tesseract_status()
        has_sig = "signature_bands" in res
        if not has_sig:
            ss.auto_sig = False
            st.info("Signature auto-clean: re-run this form set — the cached "
                    "result predates the updated pipeline (no detection ran).")
        elif not ocr_ok:
            ss.auto_sig = False
            st.warning("Signature auto-clean needs the Tesseract OCR engine, "
                       "which isn't installed here, so no 'Signature' column "
                       "could be detected. Install it and re-run:\n\n"
                       "`sudo apt install tesseract-ocr` (Linux) or "
                       "`brew install tesseract` (macOS), then "
                       "`pip install pytesseract`.")
        else:
            gl = (ss.active_group or "").lower()
            if "type3" in gl:
                def_row_thr, def_bridge_div, def_border_div = 0.98, 12, 80
            else:
                def_row_thr, def_bridge_div, def_border_div = 0.90, 5, 80

            with st.expander("Signature cleaning settings", expanded=False):
                ss.sig_row_thr = st.slider(
                    "Row coverage threshold", 0.70, 0.99, float(def_row_thr), 0.01,
                    key=f"sigrowthr_{ss.active_group}",
                    help="Fraction of a row that must be ink (after bridging) for "
                         "it to be kept as a divider. Higher = stricter.")
                ss.sig_bridge_div = st.slider(
                    "Bridge kernel divisor", 3, 20, int(def_bridge_div), 1,
                    key=f"sigbridgediv_{ss.active_group}",
                    help="Horizontal closing kernel is width // this. Larger "
                         "divisor = narrower kernel = bridges only small gaps.")
                ss.sig_border_div = st.slider(
                    "Border width divisor", 20, 200, int(def_border_div), 1,
                    key=f"sigborderdiv_{ss.active_group}",
                    help="Preserved left/right border width is column width // "
                         "this. Larger divisor = thinner preserved border.")

            sig_bands = pl.suggest_signature_bands(
                res["freq"], res.get("mean_gray"),
                pad_right=pl.DEFAULT_PARAMS["sig_pad_right"],
                pad_left=pl.DEFAULT_PARAMS["sig_pad_left"],
                y_pad_factor=pl.DEFAULT_PARAMS["sig_y_pad_factor"])

            if ("type2" in gl or "type3" in gl) and sig_bands:
                H, _ = res["freq"].shape
                target_yp = 630.0 / H
                sig_bands = [(x0, x1, target_yp) for x0, x1, _ in sig_bands]
                res["signature_bands"] = sig_bands
            else:
                res["signature_bands"] = sig_bands

            nsig = len(sig_bands)

            with st.expander("Override: manually adjust signature bands", expanded=False):
                H, W = res["freq"].shape
                st.caption("Fine-tune the OCR-detected columns or add/edit them manually. "
                          "x0/x1 are horizontal fractions (0–1). "
                          "Y-protect is how many pixels down from the top of the image "
                          "to keep the header safe from cleaning.")
                manual_bands = []
                for i, (x0, x1, yp) in enumerate(sig_bands):
                    st.write(f"**Column {i + 1}** (x: {x0:.2f}–{x1:.2f}, y-protect: {int(yp*H)}px)")
                    m1, m2, m3 = st.columns(3)
                    x0_adj = m1.slider(f"Left (x0)", 0.0, 1.0, float(x0), 0.01,
                                       key=f"band{i}_x0_{ss.active_group}")
                    x1_adj = m2.slider(f"Right (x1)", 0.0, 1.0, float(x1), 0.01,
                                       key=f"band{i}_x1_{ss.active_group}")
                    yp_px = int(yp * H)
                    yp_px_adj = m3.slider(f"Y-protect (px)", 0, H, yp_px, 10,
                                          key=f"band{i}_yp_{ss.active_group}",
                                          help="Pixels from top of image. Everything "
                                               "above this line in the column is protected "
                                               "from cleaning (keeps the header).")
                    yp_adj = yp_px_adj / H
                    manual_bands.append((x0_adj, x1_adj, yp_adj))

                if st.button("+ Add another column", key=f"add_band_{ss.active_group}"):
                    if manual_bands:
                        avg_x0 = sum(b[0] for b in manual_bands) / len(manual_bands)
                        avg_x1 = sum(b[1] for b in manual_bands) / len(manual_bands)
                        avg_yp = sum(b[2] for b in manual_bands) / len(manual_bands)
                        new_x0 = min(1.0, avg_x1 + 0.05)
                        new_x1 = min(1.0, new_x0 + (avg_x1 - avg_x0))
                        manual_bands.append((new_x0, new_x1, avg_yp))
                    else:
                        manual_bands.append((0.7, 0.85, 0.10))
                    st.rerun()

                if manual_bands:
                    res["signature_bands"] = manual_bands
                    nsig = len(manual_bands)

            ss.auto_sig = st.checkbox(
                f"Auto-clean signature columns ({nsig} band{'s' if nsig != 1 else ''})", True,
                help="Erases handwriting in 'Signature' columns "
                     "while keeping the dotted dividers, borders and header. "
                     "Heavy overlapping signatures may leave a little residue.")
            if nsig == 0:
                st.caption("No signature columns detected or configured.")

        mask = current_template()
        template_img = pl.render_mask(mask)
        heat = pl.freq_heatmap(res["freq"])

        show_overlays = st.checkbox(
            "Show signature band overlays", False,
            key=f"show_overlays_{ss.active_group}",
            help="Draw the green column bounds and yellow y-protect line "
                 "on top of the template.")

        if show_overlays and ss.auto_sig and res.get("signature_bands"):
            import cv2
            H, W = res["freq"].shape
            viz = cv2.cvtColor(template_img.copy(), cv2.COLOR_GRAY2BGR)
            for (x0, x1, yp) in res.get("signature_bands", []):
                xa, xb = int(x0 * W), int(x1 * W)
                yp_px = int(yp * H)
                cv2.rectangle(viz, (xa, 0), (xb, H), (0, 255, 0), 2)
                cv2.line(viz, (xa, yp_px), (xb, yp_px), (0, 255, 255), 2)
            template_img_viz = viz
            caption = "Template with signature bands (green=bounds, yellow=y-protect line)"
        else:
            template_img_viz = cv2.cvtColor(template_img, cv2.COLOR_GRAY2BGR) if len(template_img.shape) == 2 else template_img
            caption = "Extracted template"

        v1, v2 = st.columns(2)
        v1.image(template_img_viz, caption=caption,
                 use_container_width=True)
        v2.image(heat, caption="Ink-frequency heatmap: red/yellow = ink in "
                               "most scans (template), blue = rare ink "
                               "(handwriting)", use_container_width=True)
        with st.expander("Average of all aligned scans (soft preview)"):
            st.image(res["mean_gray"], use_container_width=True)

        d1, d2, d3 = st.columns(3)
        g = ss.active_group
        d1.download_button("Download template.png",
                           pl.png_bytes(template_img), f"template_{g}.png",
                           "image/png", use_container_width=True)
        d2.download_button("Download heatmap.png", pl.png_bytes(heat),
                           f"heatmap_{g}.png", "image/png",
                           use_container_width=True)
        d3.download_button("Download metrics.csv",
                           res["metrics"].to_csv(index=False).encode(),
                           f"metrics_{g}.csv", "text/csv",
                           use_container_width=True)

with tabs[4]:
    st.subheader("Per-scan view: template vs. handwriting")
    if ss.res is None:
        st.info("Run the pipeline on the Data tab first.")
    else:
        names = [n for n, _ in ss.all_items]
        name = st.selectbox("Scan", names)
        pre, ref, ares = inspect(name)
        if not ares.ok:
            st.error(f"This scan could not be aligned: {ares.reason}")
        else:
            mask = current_template()
            resid = pl.handwriting_residual(ares.warped_ink, mask, grow=2)
            overlay = np.full((*mask.shape, 3), 255, np.uint8)
            overlay[mask] = (205, 35, 35)
            overlay[resid] = (35, 70, 205)
            blend = cv2.addWeighted(ref.gray, 0.5, ares.warped_gray, 0.5, 0)

            a, b, c = st.columns(3)
            a.image(ares.warped_gray, caption="Aligned scan",
                    use_container_width=True)
            b.image(overlay, caption="Template (red) + this scan's "
                                     "handwriting (blue)",
                    use_container_width=True)
            c.image(pl.render_mask(resid),
                    caption="Extracted handwriting only (template "
                            "subtracted)", use_container_width=True)
            with st.expander("Alignment check (reference × scan blend)"):
                st.image(blend, use_container_width=True)
