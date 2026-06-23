"""
pipeline.py — extract the blank template from many filled-in scans of the same form.

Idea in one line
----------------
The printed template is the only thing that is (nearly) identical across all
scans. After geometrically registering every scan onto a common reference,
a per-pixel vote over the whole stack separates "ink in (almost) every form"
(= printed template) from "ink in only a few forms" (= handwriting, shadows,
specks, desk background).

Stages
------
1. Preprocess  : resize -> page detection + perspective crop -> illumination
                 flattening -> adaptive binarisation -> (reference only) deskew.
2. Register    : ORB keypoints + ratio-test matching + RANSAC homography with
                 geometric sanity checks; optional ECC refinement.
3. Accumulate  : streaming per-pixel ink counts (the full image stack is never
                 held in RAM, so 200+ photos are fine on a laptop).
4. Extract     : ink-frequency threshold -> despeckle -> optional line bridging.
5. Analyse     : per-scan quality metrics, SSIM, convergence experiment.

Everything is classical computer vision (no learned models) on purpose:
with ~200 unlabeled scans there is not enough data to train anything, but
there is more than enough redundancy for robust statistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd


DEFAULT_PARAMS = dict(
    max_side=4000,          # working resolution, longest image side in px:
    page_detect=False,      # finding and rectifying the sheet of paper:
    deskew_reference=True,  # straightening the reference so outputs are upright:
    deskew_scans=False,     # also deskewing every scan before alignment:
    binarize="adaptive",    # "adaptive" or "sauvola" binarisation:
    sauvola_window=0,       # Sauvola neighbourhood, 0 = auto:
    sauvola_k=0.2,          # Sauvola sensitivity, lower keeps more ink:
    adaptive_block=35,      # adaptive-threshold neighbourhood, odd:
    adaptive_C=15,          # adaptive-threshold offset:
    sharpen=0.0,            # unsharp-mask strength before binarisation:
    orb_features=8000,      # ORB keypoints per image:
    orb_on_lines=False,     # detecting ORB on the printed-line mask:
    match_ratio=0.75,       # Lowe ratio test:
    ransac_thresh=4.0,      # RANSAC reprojection threshold in px:
    min_inliers=30,         # rejecting registration below this many inliers:
    use_ecc=True,           # ECC sub-pixel polish, ~2 s/scan:
    align_method="auto",    # "auto" | "table" | "orb":
    align_orb_bias=0.01,    # in auto, vote-concentration margin favouring table:
    table_align=True,       # coarse-aligning on the table frame instead of ORB:
    min_grid_iou=0.0,       # extra rule-line overlap required of table fits:
    line_refine=True,       # final ECC on the printed grid only:
    tolerance=1,            # dilating ink before voting, eroding after:
    vote_threshold=0.60,    # ink-frequency above which a pixel is template:
    low_threshold=0.0,      # hysteresis floor, 0 = plain threshold:
    min_coverage=0.40,      # pixel must be covered by this fraction of scans:
    peak_gate=0.0,          # suppressing diffuse repeated-handwriting blobs:
    context_gate=0.0,       # dropping small ink ringed by handwriting:
    context_ruthless=False, # also dropping near-certain blobs when ringed:
    grid_reconstruct=False, # rebuilding and redrawing rule lines clean:
    grid_low=0.35,          # vote floor for detecting a faint rule:
    grid_thickness=2,       # redrawn rule width in px:
    min_blob=0,             # despeckle, dropping components smaller than this:
    bridge=0,               # closing gaps up to this many px along lines:
    linesnap=False,         # guarded rule-line snap after table alignment:
    sig_pad_left=0.45,      # signature-band left padding past the 'S':
    sig_pad_right=0.33,     # signature-band right padding past the word:
    sig_y_pad_factor=0.9,   # signature-band padding below the header word:
)

CHECKPOINTS = [3, 5, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 200]


def imdecode_bytes(data: bytes) -> Optional[np.ndarray]:
    """Decode raw image bytes to a BGR array (None if unreadable)."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                         interpolation=cv2.INTER_AREA)
    return img


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], np.float32)


def four_point_warp(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Perspective-rectify the region inside `quad` (tl,tr,br,bl)."""
    tl, tr, br, bl = quad
    w = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    h = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    w, h = max(w, 32), max(h, 32)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderValue=255)


def detect_page_quad(gray: np.ndarray) -> Optional[np.ndarray]:
    """Find the sheet of paper as the largest convex quadrilateral.

    Returns the quad in *full-resolution* coordinates, or None.
    """
    h, w = gray.shape
    scale = 800.0 / max(h, w)
    small = cv2.resize(gray, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA) if scale < 1 else gray
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    area_img = small.shape[0] * small.shape[1]
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:6]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            if cv2.contourArea(approx) >= 0.45 * area_img:
                quad = order_quad(approx.reshape(4, 2))
                if scale < 1:
                    quad = quad / scale
                return quad.astype(np.float32)
    return None


def flatten_illumination(gray: np.ndarray) -> np.ndarray:
    """Remove shadows / uneven lighting by dividing out the background."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)
    bg = np.maximum(bg, 1)
    return cv2.divide(gray, bg, scale=255)


def unsharp_mask(gray: np.ndarray, amount: float,
                 radius: float = 1.2) -> np.ndarray:
    """Sharpen a soft/blurry grayscale scan with a standard unsharp mask.

    `sharp = gray + amount * (gray - blur(gray))`. Recovers faint rule lines
    so they survive adaptive thresholding, and gives ORB crisper corners for
    registration. `addWeighted` saturates back to uint8. amount <= 0 is a
    no-op. Keep it modest: too much amplifies sensor/JPEG noise.
    """
    if amount <= 0:
        return gray
    blur = cv2.GaussianBlur(gray, (0, 0), radius)
    return cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)


def sauvola_ink(gray: np.ndarray, window: int = 0, k: float = 0.2,
                R: float = 128.0) -> np.ndarray:
    """Sauvola local adaptive binarisation -> ink mask (ink = 255).

    Threshold per pixel: T = mean * (1 + k*(std/R - 1)), computed over a local
    window via box filters (integral-image equivalent, O(1) per pixel). More
    robust than a single Gaussian-adaptive cut on camera scans with smear /
    uneven light: it suppresses faint background while keeping thin strokes,
    which is exactly what dilutes the per-pixel vote on blurry forms.
    """
    g = gray.astype(np.float32)
    if window <= 0:
        window = max(15, (min(gray.shape) // 40) | 1)
    window |= 1
    mean = cv2.boxFilter(g, -1, (window, window), borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(g * g, -1, (window, window),
                            borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    thresh = mean * (1.0 + k * (std / R - 1.0))
    return ((g < thresh).astype(np.uint8)) * 255


def estimate_skew_deg(ink: np.ndarray) -> float:
    """Estimate residual rotation from the long horizontal rule lines."""
    h, w = ink.shape
    lines = cv2.HoughLinesP(ink, 1, np.pi / 180, threshold=120,
                            minLineLength=w // 3, maxLineGap=8)
    if lines is None:
        return 0.0
    angles, weights = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(ang) < 20:
            angles.append(ang)
            weights.append(math.hypot(x2 - x1, y2 - y1))
    if not angles:
        return 0.0
    order = np.argsort(angles)
    a = np.array(angles)[order]
    cum = np.cumsum(np.array(weights, float)[order])
    return float(a[np.searchsorted(cum, cum[-1] / 2)])


def rotate_keep_size(img: np.ndarray, angle_deg: float,
                     border: int) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderValue=border)


@dataclass
class Preproc:
    gray: np.ndarray
    ink: np.ndarray
    page_warped: bool = False
    skew_deg: float = 0.0


def preprocess(bgr: np.ndarray, params: dict, deskew: bool = False) -> Preproc:
    img = resize_max(bgr, int(params["max_side"]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    page_warped = False
    if params.get("page_detect", True):
        quad = detect_page_quad(gray)
        if quad is not None:
            gray = four_point_warp(gray, quad)
            page_warped = True

    norm = flatten_illumination(gray)
    norm = unsharp_mask(norm, float(params.get("sharpen", 0.0)))
    if params.get("binarize", "adaptive") == "sauvola":
        ink = sauvola_ink(norm, window=int(params.get("sauvola_window", 0)),
                          k=float(params.get("sauvola_k", 0.2)))
    else:
        block = int(params["adaptive_block"]) | 1
        ink = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, block,
                                    int(params["adaptive_C"]))
    ink = cv2.medianBlur(ink, 3)

    skew = 0.0
    if deskew or params.get("deskew_scans", False):
        skew = estimate_skew_deg(ink)
        if abs(skew) > 0.15:
            norm = rotate_keep_size(norm, skew, border=255)
            ink = rotate_keep_size(ink, skew, border=0)
            ink = (ink > 127).astype(np.uint8) * 255
    return Preproc(gray=norm, ink=ink, page_warped=page_warped, skew_deg=skew)


def quality_metrics(gray: np.ndarray, ink: np.ndarray) -> dict:
    """Cheap per-scan quality indicators used for the cleaning step."""
    return dict(
        blur_var=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        brightness=float(gray.mean()),
        contrast=float(gray.std()),
        ink_ratio=float((ink > 0).mean()),
    )


@dataclass
class Reference:
    name: str
    gray: np.ndarray
    ink: np.ndarray
    kp: list = field(default_factory=list)
    des: Optional[np.ndarray] = None

    @property
    def size_wh(self) -> tuple[int, int]:
        h, w = self.gray.shape
        return w, h


def build_reference(name: str, pre: Preproc, params: dict) -> Reference:
    orb = cv2.ORB_create(nfeatures=int(params["orb_features"]),
                         scaleFactor=1.2, nlevels=8, fastThreshold=10)
    det = pre.gray
    if params.get("orb_on_lines", False):
        lines = _printed_line_mask(pre.gray)
        if np.count_nonzero(lines) > 0.02 * pre.gray.size:
            det = lines
    kp, des = orb.detectAndCompute(det, None)
    return Reference(name=name, gray=pre.gray, ink=pre.ink, kp=kp, des=des)


@dataclass
class AlignResult:
    ok: bool
    reason: str = ""
    inliers: int = 0
    reproj_err: float = 0.0
    ssim: float = 0.0
    grid_iou: float = np.nan
    method: str = "orb"
    H: Optional[np.ndarray] = None
    warped_gray: Optional[np.ndarray] = None
    warped_ink: Optional[np.ndarray] = None
    valid: Optional[np.ndarray] = None


def _homography_sane(H: np.ndarray, src_shape, ref_wh) -> bool:
    h, w = src_shape
    W, Hh = ref_wh
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]],
                       np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H).reshape(4, 2)
    if not cv2.isContourConvex(warped.astype(np.float32)):
        return False
    area = cv2.contourArea(warped.astype(np.float32))
    ratio = area / float(W * Hh)
    return 0.35 <= ratio <= 2.5


def ssim_gray(a: np.ndarray, b: np.ndarray, inset: float = 0.04) -> float:
    """Plain-NumPy SSIM (Gaussian 11x11, sigma 1.5) on a central crop."""
    h, w = a.shape
    iy, ix = int(h * inset), int(w * inset)
    a = a[iy:h - iy, ix:w - ix].astype(np.float64)
    b = b[iy:h - iy, ix:w - ix].astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    g = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_a, mu_b = g(a), g(b)
    va = g(a * a) - mu_a ** 2
    vb = g(b * b) - mu_b ** 2
    cov = g(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * cov + C2)) / \
        ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))
    return float(s.mean())


def align_to_reference(pre: Preproc, ref: Reference, params: dict,
                       matcher: Optional[cv2.BFMatcher] = None) -> AlignResult:
    if ref.des is None or len(ref.kp) < 10:
        return AlignResult(False, "reference has too few features")
    orb = cv2.ORB_create(nfeatures=int(params["orb_features"]),
                         scaleFactor=1.2, nlevels=8, fastThreshold=10)
    det = pre.gray
    if params.get("orb_on_lines", False):
        lines = _printed_line_mask(pre.gray)
        if np.count_nonzero(lines) > 0.02 * pre.gray.size:
            det = lines
    kp, des = orb.detectAndCompute(det, None)
    if des is None or len(kp) < 10:
        return AlignResult(False, "too few features")

    bf = matcher or cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = bf.knnMatch(des, ref.des, k=2)
    ratio = float(params["match_ratio"])
    good = [m for m, *rest in (p for p in pairs if len(p) == 2)
            if m.distance < ratio * rest[0].distance]
    if len(good) < max(8, int(params["min_inliers"]) // 2):
        return AlignResult(False, f"only {len(good)} matches")

    src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([ref.kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inmask = cv2.findHomography(src, dst, cv2.RANSAC,
                                   float(params["ransac_thresh"]))
    if H is None or inmask is None:
        return AlignResult(False, "homography failed")
    inliers = int(inmask.sum())
    if inliers < int(params["min_inliers"]):
        return AlignResult(False, f"only {inliers} inliers", inliers=inliers)
    if not _homography_sane(H, pre.gray.shape, ref.size_wh):
        return AlignResult(False, "degenerate homography", inliers=inliers)

    proj = cv2.perspectiveTransform(src[inmask.ravel() == 1], H)
    err = float(np.linalg.norm(
        proj - dst[inmask.ravel() == 1], axis=2).mean())

    W, Hh = ref.size_wh
    warped_gray = cv2.warpPerspective(pre.gray, H, (W, Hh),
                                      flags=cv2.INTER_LINEAR, borderValue=255)
    warped_ink = cv2.warpPerspective(pre.ink, H, (W, Hh),
                                     flags=cv2.INTER_NEAREST, borderValue=0)
    ones = np.full(pre.gray.shape, 255, np.uint8)
    valid = cv2.warpPerspective(ones, H, (W, Hh),
                                flags=cv2.INTER_NEAREST, borderValue=0) > 0

    if params.get("use_ecc", False):
        warped_gray, warped_ink, valid = _ecc_refine(
            ref.gray, warped_gray, warped_ink, valid)
    if params.get("line_refine", True):
        warped_gray, warped_ink, valid = _line_refine(
            ref.gray, warped_gray, warped_ink, valid)
    if params.get("linesnap", False):
        warped_gray, warped_ink, valid = _linesnap_refine(
            ref.gray, warped_gray, warped_ink, valid)

    s = ssim_gray(ref.gray, warped_gray)
    return AlignResult(True, "", inliers=inliers, reproj_err=err, ssim=s,
                       method="orb", H=H, warped_gray=warped_gray,
                       warped_ink=warped_ink, valid=valid)


def _ecc_refine(ref_gray, warped_gray, warped_ink, valid):
    """Optional sub-pixel polish with ECC. Falls back silently on failure."""
    try:
        scale = 0.5
        rs = cv2.resize(ref_gray, None, fx=scale, fy=scale)
        ws = cv2.resize(warped_gray, None, fx=scale, fy=scale)
        warp = np.eye(3, 3, dtype=np.float32)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
        _, warp = cv2.findTransformECC(rs, ws, warp,
                                       cv2.MOTION_HOMOGRAPHY, crit, None, 5)
        S = np.diag([scale, scale, 1.0]).astype(np.float32)
        Hf = np.linalg.inv(S) @ warp @ S
        h, w = ref_gray.shape
        f = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        g = cv2.warpPerspective(warped_gray, Hf, (w, h), flags=f,
                                borderValue=255)
        i = cv2.warpPerspective(warped_ink, Hf, (w, h),
                                flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
                                borderValue=0)
        v = cv2.warpPerspective(valid.astype(np.uint8) * 255, Hf, (w, h),
                                flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
                                borderValue=0) > 0
        return g, i, v
    except cv2.error:
        return warped_gray, warped_ink, valid


def _printed_line_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask of the long printed rule lines (horizontal + vertical).

    Morphological opening with a long thin kernel keeps only strokes that run
    far enough to be table rules, dropping text and handwriting. These lines
    are (almost) identical on every scan, so they are an ideal registration
    target — the robust generalisation of "align on the table edge".
    """
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 31, 10)
    h, w = gray.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 25), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 25)))
    horiz = cv2.morphologyEx(ink, cv2.MORPH_OPEN, hk)
    vert = cv2.morphologyEx(ink, cv2.MORPH_OPEN, vk)
    return cv2.max(horiz, vert)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def _line_refine(ref_gray, warped_gray, warped_ink, valid, scale=0.5):
    """Final registration polish on the printed GRID only.

    Extracts the rule-line masks of the reference and the (already roughly
    aligned) scan and runs ECC to lock the two grids together. Because only
    the shared printed lines drive it — not handwriting, shadows or speckle —
    it removes the residual 1-3px jitter that otherwise smears the per-pixel
    vote. The adjustment is accepted only if it raises line-mask overlap, so a
    poor fit is discarded rather than allowed to hurt the result.
    """
    try:
        ref_lines = _printed_line_mask(ref_gray)
        cur_lines = _printed_line_mask(warped_gray)
        before = _mask_iou(ref_lines, cur_lines)
        if before <= 0.0:
            return warped_gray, warped_ink, valid
        rs = cv2.GaussianBlur(cv2.resize(ref_lines, None, fx=scale, fy=scale),
                              (0, 0), 1.5).astype(np.float32)
        cs = cv2.GaussianBlur(cv2.resize(cur_lines, None, fx=scale, fy=scale),
                              (0, 0), 1.5).astype(np.float32)
        warp = np.eye(3, 3, dtype=np.float32)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-6)
        _, warp = cv2.findTransformECC(rs, cs, warp, cv2.MOTION_HOMOGRAPHY,
                                       crit, None, 5)
        S = np.diag([scale, scale, 1.0]).astype(np.float32)
        Hf = np.linalg.inv(S) @ warp @ S
        h, w = ref_gray.shape
        fl = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        fn = cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP
        g = cv2.warpPerspective(warped_gray, Hf, (w, h), flags=fl,
                                borderValue=255)
        after = _mask_iou(ref_lines, _printed_line_mask(g))
        if after <= before:
            return warped_gray, warped_ink, valid
        i = cv2.warpPerspective(warped_ink, Hf, (w, h), flags=fn,
                                borderValue=0)
        v = cv2.warpPerspective(valid.astype(np.uint8) * 255, Hf, (w, h),
                                flags=fn, borderValue=0) > 0
        return g, i, v
    except cv2.error:
        return warped_gray, warped_ink, valid


def _rule_line_positions(gray: np.ndarray, axis: int) -> np.ndarray:
    """Centre positions of long printed rule lines along one axis.

    axis=0 -> horizontal rules (returns y centres); axis=1 -> vertical (x).
    A long thin morphological opening keeps only table rules, then 1-D peaks
    in the projection profile give their positions.
    """
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 31, 10)
    h, w = gray.shape
    if axis == 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 8), 1))
        prof = cv2.morphologyEx(ink, cv2.MORPH_OPEN, k).sum(1).astype(float)
    else:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 8)))
        prof = cv2.morphologyEx(ink, cv2.MORPH_OPEN, k).sum(0).astype(float)
    prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 5), 0).ravel()
    if prof.max() <= 0:
        return np.array([])
    thr = 0.30 * prof.max()
    pos, i, n = [], 0, len(prof)
    while i < n:
        if prof[i] > thr:
            j = i
            while j < n and prof[j] > thr:
                j += 1
            seg = prof[i:j]
            pos.append(float((np.arange(i, j) * seg).sum() / seg.sum()))
            i = j
        else:
            i += 1
    return np.array(pos)


def _monotonic_match(ref_pos: np.ndarray, scn_pos: np.ndarray,
                     tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Order-preserving 1-to-1 nearest match (so the remap can't fold)."""
    i = j = 0
    sp, rp = [], []
    while i < len(scn_pos) and j < len(ref_pos):
        d = scn_pos[i] - ref_pos[j]
        if abs(d) <= tol:
            sp.append(scn_pos[i]); rp.append(ref_pos[j]); i += 1; j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return np.array(sp), np.array(rp)


def _axis_map(scn_pos: np.ndarray, ref_pos: np.ndarray,
              size: int) -> Optional[np.ndarray]:
    """Monotone lookup: output coord (ref) -> input coord (scan), via linear
    interpolation of matched rule positions, identity-clamped at the edges."""
    if len(scn_pos) < 2:
        return None
    xs = np.concatenate(([0.0], ref_pos, [size - 1.0]))
    ys = np.concatenate(([0.0], scn_pos, [size - 1.0]))
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    keep = np.concatenate(([True], np.diff(xs) > 1e-3))
    xs, ys = xs[keep], ys[keep]
    return np.interp(np.arange(size, dtype=np.float32), xs, ys).astype(np.float32)


def _linesnap_refine(ref_gray, warped_gray, warped_ink, valid):
    """Snap an already table-aligned scan's rule lines onto the reference's.

    The correction is a separable, monotone resampling (rows onto the
    reference's horizontal rules, columns onto its vertical rules), so unlike a
    free homography on noisy line matches it can never scatter content. It is
    accepted only if it raises rule-line overlap with the reference; otherwise
    the input is returned unchanged.
    """
    h, w = ref_gray.shape
    ry = _rule_line_positions(ref_gray, 0)
    sy = _rule_line_positions(warped_gray, 0)
    rx = _rule_line_positions(ref_gray, 1)
    sx = _rule_line_positions(warped_gray, 1)
    sy_m, ry_m = _monotonic_match(ry, sy, tol=0.015 * h)
    sx_m, rx_m = _monotonic_match(rx, sx, tol=0.015 * w)
    if len(sy_m) < 4 and len(sx_m) < 3:
        return warped_gray, warped_ink, valid
    ymap = _axis_map(sy_m, ry_m, h)
    xmap = _axis_map(sx_m, rx_m, w)
    if ymap is None and xmap is None:
        return warped_gray, warped_ink, valid
    if ymap is None:
        ymap = np.arange(h, dtype=np.float32)
    if xmap is None:
        xmap = np.arange(w, dtype=np.float32)
    map_x = np.tile(xmap, (h, 1))
    map_y = np.repeat(ymap.reshape(-1, 1), w, axis=1)
    g = cv2.remap(warped_gray, map_x, map_y, cv2.INTER_LINEAR, borderValue=255)
    before = _mask_iou(_printed_line_mask(ref_gray),
                       _printed_line_mask(warped_gray))
    after = _mask_iou(_printed_line_mask(ref_gray), _printed_line_mask(g))
    if after <= before:
        return warped_gray, warped_ink, valid
    i = cv2.remap(warped_ink, map_x, map_y, cv2.INTER_NEAREST, borderValue=0)
    v = cv2.remap(valid.astype(np.uint8) * 255, map_x, map_y,
                  cv2.INTER_NEAREST, borderValue=0) > 0
    return g, i, v


def detect_grid_quad(gray: np.ndarray) -> Optional[np.ndarray]:
    """Find the table's outer border as a quad (tl, tr, br, bl), or None.

    The table border plus its internal rules form one large connected blob of
    ink, while handwriting and header text are smaller, separate blobs. Taking
    the largest connected component and fitting its 4 corners is robust to
    rotation. Detection is done at a fixed ~1000px working size (then the
    corners are scaled back) so it behaves identically at any working
    resolution — fixed-size morphology was previously failing at high res.
    """
    H, W = gray.shape
    scale = 1000.0 / max(H, W)
    small = (cv2.resize(gray, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_AREA) if scale < 1 else gray)
    h, w = small.shape
    ink = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 31, 10)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    if n <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[idx, cv2.CC_STAT_AREA] < 0.05 * h * w:
        return None
    comp = (labels == idx).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        quad = approx.reshape(4, 2).astype(np.float32)
    else:
        quad = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
    if scale < 1:
        quad = quad / scale
    return order_quad(quad)


def _top_edge_angle(quad: np.ndarray) -> float:
    """Tilt of a quad's top edge (tl->tr) in degrees; 0 when upright."""
    (x0, y0), (x1, y1) = quad[0], quad[1]
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


def axis_aligned_box(quad: np.ndarray) -> np.ndarray:
    """The axis-aligned bounding rectangle of `quad` as a (tl,tr,br,bl) quad."""
    x0, y0 = quad.min(0)
    x1, y1 = quad.max(0)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)


def _canon_rect_from_quad(quad: np.ndarray) -> np.ndarray:
    """An upright rectangle of the table's TRUE proportions, kept at the
    quad's own top-left so the header above and the margins around are
    preserved (the table is not stretched to fill the frame)."""
    tl, tr, br, bl = quad
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    x0, y0 = quad.min(0)
    return np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]],
                    np.float32)


def _warp_quad(gray, ink, src_quad, dst_quad, out_wh):
    """Warp so src_quad lands on dst_quad; returns (gray, ink, valid)."""
    W, Hh = out_wh
    H = cv2.getPerspectiveTransform(src_quad.astype(np.float32),
                                    dst_quad.astype(np.float32))
    g = cv2.warpPerspective(gray, H, (W, Hh), flags=cv2.INTER_LINEAR,
                            borderValue=255)
    i = cv2.warpPerspective(ink, H, (W, Hh), flags=cv2.INTER_NEAREST,
                            borderValue=0)
    ones = np.full(gray.shape, 255, np.uint8)
    v = cv2.warpPerspective(ones, H, (W, Hh), flags=cv2.INTER_NEAREST,
                            borderValue=0) > 0
    return g, i, v, H


def align_on_grid(pre: Preproc, ref: Reference, canon_quad: np.ndarray,
                  params: dict) -> AlignResult:
    """Register a scan by mapping its detected table corners onto `canon_quad`
    (a fixed upright rectangle). A single perspective rectification overlays
    the table and uprights the whole page — header included — with no
    full-page feature matching that could drift on the dense rows. ECC / line
    polish are optional and off by default; accepted only if rule-line overlap
    with the reference passes a floor.
    """
    quad = detect_grid_quad(pre.gray)
    if quad is None:
        return AlignResult(False, "no table frame found", method="table")
    g, i, v, H = _warp_quad(pre.gray, pre.ink, quad, canon_quad, ref.size_wh)
    if not _homography_sane(H, pre.gray.shape, ref.size_wh):
        return AlignResult(False, "degenerate table fit", method="table")
    if params.get("line_refine", True):
        g, i, v = _line_refine(ref.gray, g, i, v)
    if params.get("linesnap", False):
        g, i, v = _linesnap_refine(ref.gray, g, i, v)
    gi = _mask_iou(_printed_line_mask(ref.gray), _printed_line_mask(g))
    s = ssim_gray(ref.gray, g)
    ok = gi >= float(params.get("min_grid_iou", 0.0))
    return AlignResult(ok, "" if ok else f"low grid overlap ({gi:.2f})",
                       inliers=0, reproj_err=0.0, ssim=s, grid_iou=gi,
                       method="table", H=H, warped_gray=g, warped_ink=i,
                       valid=v)


def _canonical_reference(ref_pre: "Preproc"):
    """Make the reference upright by rectifying its table frame.

    Mapping the (possibly tilted/perspective) table corners to an upright
    rectangle of the table's true proportions straightens the entire planar
    page in one step — header and margins included. Returns (canonical
    Preproc, canon_rect) where canon_rect is the fixed rectangle every other
    scan is mapped onto, or (ref_pre, None) if no table frame is found.
    """
    gray, ink = ref_pre.gray, ref_pre.ink
    q = detect_grid_quad(gray)
    if q is None:
        return ref_pre, None
    canon_rect = _canon_rect_from_quad(q)
    h, w = gray.shape
    g, i, _, _ = _warp_quad(gray, ink, q, canon_rect, (w, h))
    return Preproc(gray=g, ink=i, page_warped=ref_pre.page_warped,
                   skew_deg=ref_pre.skew_deg), canon_rect


class TemplateAccumulator:
    """Per-pixel statistics over aligned scans, O(1) memory in #scans."""

    def __init__(self, shape: tuple[int, int]):
        self.ink_count = np.zeros(shape, np.uint16)
        self.valid_count = np.zeros(shape, np.uint16)
        self.gray_sum = np.zeros(shape, np.float32)
        self.n = 0

    def add(self, warped_gray, warped_ink, valid):
        v = valid.astype(np.uint16)
        self.ink_count += ((warped_ink > 0) & valid).astype(np.uint16)
        self.valid_count += v
        self.gray_sum += warped_gray.astype(np.float32) * valid
        self.n += 1

    def freq(self) -> np.ndarray:
        return self.ink_count / np.maximum(self.valid_count, 1)

    def coverage(self) -> np.ndarray:
        return self.valid_count / max(self.n, 1)

    def mean_gray(self) -> np.ndarray:
        m = self.gray_sum / np.maximum(self.valid_count, 1)
        m[self.valid_count == 0] = 255
        return np.clip(m, 0, 255).astype(np.uint8)


def _hysteresis(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """Keep connected components of `weak` that contain a `strong` pixel.

    Hysteresis thresholding (as in Canny): faint, low-vote rule lines are
    retained only where they connect to a confident, high-vote seed, so the
    soft structure visible in the frequency heatmap survives while isolated
    low-vote speckle (occasional handwriting / noise) is dropped.
    """
    n, labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    seeds = np.unique(labels[strong])
    seeds = seeds[seeds != 0]
    if seeds.size == 0:
        return np.zeros_like(strong)
    return np.isin(labels, seeds)


def _grid_line_centers(line_mask: np.ndarray, axis: int) -> list[float]:
    """Centre coordinates of rule lines in a line-only mask.

    axis=0 -> horizontal lines (returns row/y centres);
    axis=1 -> vertical lines (returns column/x centres).
    """
    prof = line_mask.sum(axis=1 if axis == 0 else 0).astype(np.float64)
    if prof.max() <= 0:
        return []
    prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 5), 0).ravel()
    thr = 0.12 * prof.max()
    centres, i, n = [], 0, len(prof)
    while i < n:
        if prof[i] > thr:
            j = i
            while j < n and prof[j] > thr:
                j += 1
            seg = prof[i:j]
            centres.append(float((np.arange(i, j) * seg).sum() / seg.sum()))
            i = j
        else:
            i += 1
    return centres


def _dense_span(centers: list[float], tol: float = 1.8) -> tuple[float, float]:
    """Span of the longest run of near-uniformly spaced positions.

    The table body is many rules at a regular pitch; stray header underlines
    sit above it separated by a large gap. Returns the (start, end) of the
    biggest regularly-spaced cluster, so the table box excludes those header
    lines instead of stretching up to them.
    """
    c = sorted(float(v) for v in centers)
    if len(c) < 3:
        return c[0], c[-1]
    gaps = np.diff(c)
    med = float(np.median(gaps))
    if med <= 0:
        return c[0], c[-1]
    thr = tol * med
    best = (0, 0)
    i, n = 0, len(gaps)
    while i < n:
        if gaps[i] <= thr:
            j = i
            while j < n and gaps[j] <= thr:
                j += 1
            if (j - i) > (best[1] - best[0]):
                best = (i, j)
            i = j
        else:
            i += 1
    return c[best[0]], c[best[1]]


def reconstruct_grid(freq: np.ndarray, low: float = 0.35,
                     thickness: int = 2, close_border: bool = True
                     ) -> Optional[np.ndarray]:
    """Rebuild the table's rule lines from the multi-form vote map.

    This is a regularisation of the *similarity* result, not single-image
    detection: it works on `freq` (per-pixel agreement across every aligned
    form). A deliberately low threshold keeps faint-but-present edges that the
    final vote cut discards, isolates long horizontal / vertical runs (rules,
    not text or handwriting) by morphological opening, finds each rule's
    centre, then redraws them as clean straight lines spanning the full table
    extent -- so a rule that is only faintly voted on one side (the lever-arm /
    curl problem) is recovered along its whole length. Returns a boolean grid
    mask, or None if no table-like line structure is found.
    """
    H, W = freq.shape
    soft = (freq >= low).astype(np.uint8) * 255
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, W // 4), 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, H // 4)))
    horiz = cv2.morphologyEx(soft, cv2.MORPH_OPEN, kh)
    vert = cv2.morphologyEx(soft, cv2.MORPH_OPEN, kv)
    ys = _grid_line_centers(horiz, axis=0)
    xs = _grid_line_centers(vert, axis=1)
    if len(ys) < 2 or len(xs) < 2:
        return None

    y_top, y_bot = _dense_span(ys)
    x_left, x_right = min(xs), max(xs)

    grid = np.zeros((H, W), np.uint8)
    xa, xb = int(round(x_left)), int(round(x_right))
    ya, yb = int(round(y_top)), int(round(y_bot))
    span_x = max(1, xb - xa); span_y = max(1, yb - ya)
    t = max(1, int(thickness))
    band = max(2, t + 1)

    if close_border:
        cv2.rectangle(grid, (xa, ya), (xb, yb), 255, t)

    for y in ys:
        yi = int(round(y))
        in_table = (ya - band) <= yi <= (yb + band)
        if in_table and (abs(yi - ya) <= band or abs(yi - yb) <= band):
            continue
        row = horiz[max(0, yi - band):yi + band + 1, xa:xb + 1].any(axis=0)
        cols = np.where(row)[0]
        if cols.size == 0:
            continue
        if in_table and (cols.max() - cols.min()) >= 0.5 * span_x:
            x0, x1 = xa, xb
        else:
            x0, x1 = xa + int(cols.min()), xa + int(cols.max())
        cv2.line(grid, (x0, yi), (x1, yi), 255, t)

    for x in xs:
        xi = int(round(x))
        if abs(xi - xa) <= band or abs(xi - xb) <= band:
            continue
        colband = vert[ya:yb + 1, max(0, xi - band):xi + band + 1].any(axis=1)
        rows = np.where(colband)[0]
        if rows.size and (rows.max() - rows.min()) >= 0.5 * span_y:
            y0, y1 = ya, yb
        elif rows.size:
            y0, y1 = ya + int(rows.min()), ya + int(rows.max())
        else:
            continue
        cv2.line(grid, (xi, y0), (xi, y1), 255, t)
    return grid > 0


def _line_like(mask: np.ndarray) -> np.ndarray:
    """Pixels of `mask` that belong to a long horizontal or vertical run."""
    m = mask.astype(np.uint8) * 255
    H, W = mask.shape
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, W // 4), 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, H // 4)))
    return ((cv2.morphologyEx(m, cv2.MORPH_OPEN, kh) > 0) |
            (cv2.morphologyEx(m, cv2.MORPH_OPEN, kv) > 0))


def _clean_signature_band(band: np.ndarray, y_protect: int,
                          t: int = 2,
                          row_threshold: float = 0.90,
                          bridge_ratio: int = 5,
                          border_divisor: int = 80) -> np.ndarray:
    """Erase handwriting from one signature column while keeping the dotted row
    dividers, the column's left/right borders and the printed header.

    Parameters:
        band: binary mask of the column (ink=255)
        y_protect: pixels from top to protect
        t: vertical dilation for divider rows
        row_threshold: coverage fraction needed to keep a row (0-1)
        bridge_ratio: divisor for the bridging kernel width (w // bridge_ratio)
        border_divisor: divisor for the preserved border width (w // border_divisor)
    """
    h, w = band.shape
    if w < 6 or band.sum() == 0:
        return band
    bm = band.astype(np.uint8) * 255

    ksize = max(8, w // bridge_ratio)
    closed = cv2.morphologyEx(bm, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(
                                  cv2.MORPH_RECT, (ksize, 1)))
    rowcov = (closed > 0).mean(axis=1)
    keep = np.zeros((h, w), dtype=bool)

    for y in np.where(rowcov > row_threshold)[0]:
        keep[max(0, y - t):min(h, y + t + 1), :] = True

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    horiz_mask = cv2.morphologyEx(bm, cv2.MORPH_OPEN, horiz_kernel) > 0
    keep = keep & horiz_mask

    bt = max(2, w // max(1, border_divisor))
    keep[:, :bt] = True
    keep[:, -bt:] = True
    yp = max(0, min(int(y_protect), h))
    keep[:yp, :] = True

    return band & keep


def clean_signature_columns(mask: np.ndarray, bands: list,
                            t: int = 2,
                            row_threshold: float = 0.90,
                            bridge_ratio: int = 5,
                            border_divisor: int = 80) -> np.ndarray:
    """Erase handwriting from signature columns while keeping the dotted row
    dividers, the column borders and the printed header. Bands are (x0, x1) or
    (x0, x1, y_protect) as fractions of width/height."""
    if not bands:
        return mask
    H, W = mask.shape
    out = mask.copy()
    for bd in bands:
        x0f, x1f = bd[0], bd[1]
        yprot = int((bd[2] if len(bd) > 2 else 0.10) * H)
        a = int(round(min(x0f, x1f) * W)); b = int(round(max(x0f, x1f) * W))
        a = max(0, min(a, W - 1)); b = max(a + 1, min(b, W))
        if b - a < 6:
            continue
        out[:, a:b] = _clean_signature_band(out[:, a:b], yprot, t,
                                            row_threshold, bridge_ratio,
                                            border_divisor)
    return out


def tesseract_status() -> tuple:
    """(available, message) for the OCR backend used by signature auto-clean."""
    try:
        import pytesseract
        return True, f"Tesseract {pytesseract.get_tesseract_version()}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def suggest_signature_bands(freq: np.ndarray,
                            mean_gray: Optional[np.ndarray] = None,
                            pad_right: float = 0.33,
                            pad_left: float = 0.45,
                            y_pad_factor: float = 0.9) -> list:
    """OCR-locate the 'Signature' column header(s). The band is the word box
    extended right by ~1/3 of the word (snapped to the next strong vertical
    rule if one is near) and left to the nearest dotted divider strictly left
    of the word (fallback to pad_left if none found). Returns a list of
    (x0, x1) fractions of width, or [] if OCR is unavailable / finds no
    'Signature' header."""
    try:
        import pytesseract
    except Exception:
        return []
    H, W = freq.shape
    img = (mean_gray if mean_gray is not None
           else ((1 - (freq >= 0.6).astype(np.uint8)) * 255).astype(np.uint8))
    vsoft = (freq >= 0.50).astype(np.uint8) * 255
    walls = sorted(_grid_line_centers(cv2.morphologyEx(
        vsoft, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, H // 4)))), 1))
    vbridged = cv2.morphologyEx(vsoft, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(6, H // 80))))
    dotted_walls = _grid_line_centers(cv2.morphologyEx(
        vbridged, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, H // 6)))), 1)

    crop = img[:max(1, int(0.32 * H)), :]
    scale = max(0.5, min(2200.0 / max(1, crop.shape[1]), 4.0))
    up = cv2.resize(crop, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    found = []
    for psm in (11, 6, 4):
        try:
            d = pytesseract.image_to_data(up, config=f"--psm {psm}",
                                          output_type=pytesseract.Output.DICT)
        except Exception:
            continue
        for i, txt in enumerate(d["text"]):
            tl = txt.strip().lower()
            if "signatur" not in tl or ":" in txt:
                continue
            xl = d["left"][i] / scale
            ww = max(1.0, d["width"][i] / scale)
            cx = (xl + ww / 2) / W
            if cx < 0.22:
                continue
            word_right = xl + ww
            x1 = word_right + pad_right * ww
            cand = [w for w in walls if word_right <= w <= x1 + 0.6 * ww]
            if cand:
                x1 = min(cand, key=lambda w: abs(w - x1))

            left_candidates = [w for w in dotted_walls if w < xl]
            if left_candidates:
                x0 = max(left_candidates)
            else:
                x0 = xl - pad_left * ww

            wtop = d["top"][i] / scale
            wh = max(1.0, d["height"][i] / scale)
            y_prot = (wtop + wh + y_pad_factor * wh) / H
            found.append((max(0.0, x0 / W), min(1.0, x1 / W), cx, y_prot))
        if len(found) >= 2:
            break

    found.sort(key=lambda s: s[2])

    if len(found) >= 2:
        x0_1, x1_1, cx_1, yp_1 = found[0]
        x0_2, x1_2, cx_2, yp_2 = found[1]
        width2 = x1_2 - x0_2
        expected_x0_1 = x1_1 - width2
        if abs(x0_1 - expected_x0_1) > 0.02:
            found[0] = (max(0.0, expected_x0_1), x1_1, cx_1, yp_1)

    return [(x0, x1, yp) for x0, x1, _, yp in found]


def extract_template(freq: np.ndarray, coverage: np.ndarray,
                     vote_threshold: float, min_coverage: float,
                     min_blob: int, bridge: int,
                     tolerance: int = 0,
                     low_threshold: float = 0.0,
                     peak_gate: float = 0.0,
                     context_gate: float = 0.0,
                     context_ruthless: bool = False,
                     grid_reconstruct: bool = False,
                     grid_low: float = 0.35,
                     grid_thickness: int = 2,
                     signature_bands: Optional[list] = None,
                     sig_clean_threshold: float = 0.90,
                     sig_bridge_ratio: int = 5,
                     sig_border_divisor: int = 80) -> np.ndarray:
    """Return boolean template mask (True = printed ink).

    If the vote map was built from `tolerance`-dilated ink masks, pass the
    same value here so the mask is eroded back to true stroke width.

    When ``0 < low_threshold < vote_threshold`` the threshold becomes a
    hysteresis: pixels at or above ``vote_threshold`` are confident seeds, and
    fainter pixels down to ``low_threshold`` are kept only where they connect
    to a seed. This recovers the soft, slightly mis-registered rule lines you
    can see in the heatmap but that a single hard cut drops, without admitting
    isolated speckle. ``low_threshold = 0`` falls back to plain thresholding.

    ``peak_gate`` separates printed ink from handwriting that recurs in the
    same place (e.g. a regular signer). Printed ink lands on the exact same
    pixels every scan -> a thin ridge with a near-1.0 core standing out from a
    cool neighbourhood. Repeated handwriting jitters -> a diffuse, medium-warm
    plateau. A pixel is kept only if its vote exceeds the local average by at
    least ``peak_gate`` (or is essentially certain, freq >= 0.95). 0 = off.
    """
    covered = coverage >= min_coverage
    if 0.0 < low_threshold < vote_threshold:
        strong = (freq >= vote_threshold) & covered
        weak = (freq >= low_threshold) & covered
        mask = _hysteresis(strong, weak)
    else:
        mask = (freq >= vote_threshold) & covered

    if peak_gate > 0.0:
        f = freq.astype(np.float32)
        local = cv2.GaussianBlur(f, (0, 0), 9.0)
        peaky = (f - local) >= float(peak_gate)
        mask = mask & (peaky | (freq >= 0.95))

    m = mask.astype(np.uint8) * 255

    if tolerance > 0:
        k = 2 * int(tolerance) + 1
        m = cv2.erode(m, np.ones((k, k), np.uint8))

    if bridge > 0:
        k = 2 * int(bridge) + 1
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kh) | \
            cv2.morphologyEx(m, cv2.MORPH_CLOSE, kv)

    if min_blob > 1:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        keep = np.zeros(n, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= int(min_blob)
        m = (keep[labels]).astype(np.uint8) * 255

    if context_gate > 0.0:
        m = _context_clean(m, freq, vote_threshold, context_gate,
                            protect_certain=not context_ruthless)

    mask = m > 0
    if signature_bands:
        mask = clean_signature_columns(mask, signature_bands,
                                       t=2,
                                       row_threshold=sig_clean_threshold,
                                       bridge_ratio=sig_bridge_ratio,
                                       border_divisor=sig_border_divisor)
    if grid_reconstruct:
        grid = reconstruct_grid(freq, low=grid_low, thickness=grid_thickness)
        if grid is not None:
            mask = (mask & ~_line_like(mask)) | grid
    return mask


def _context_clean(m: np.ndarray, freq: np.ndarray, vote_thr: float,
                   gate: float, warm_lo: float = 0.15, ring: int = 4,
                   big_area: int = 600, protect_certain: bool = True) -> np.ndarray:
    """Drop small ink components ringed by handwriting.

    For each connected component small enough to be a glyph (not a rule line),
    look at a thin ring just outside it and measure the fraction that is "warm"
    -- medium ink frequency, i.e. recurring handwriting. A printed number or
    letter sits in its own near-empty cell, so its ring is cold and it is kept;
    a signature fragment that happened to vote high is buried in handwriting,
    so its ring is warm and it is removed. Large components (the grid, long
    rules) are always kept. When ``protect_certain`` is True (default) a
    near-certain pixel (freq >= 0.95) is never cut; set it False for the
    "ruthless" variant that also removes consistent recurring marks.
    """
    warm = ((freq >= warm_lo) & (freq < vote_thr)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    k = np.ones((2 * ring + 1, 2 * ring + 1), np.uint8)
    out = (m > 0)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= big_area:
            continue
        comp = (labels == i)
        ring_px = (cv2.dilate(comp.astype(np.uint8), k) > 0) & ~comp
        if ring_px.sum() == 0:
            continue
        if warm[ring_px].mean() >= gate:
            if protect_certain and freq[comp].max() >= 0.95:
                continue
            out[comp] = False
    return (out.astype(np.uint8) * 255)


def render_mask(mask: np.ndarray) -> np.ndarray:
    """Boolean ink mask -> white page with black ink (uint8)."""
    out = np.full(mask.shape, 255, np.uint8)
    out[mask] = 0
    return out


def freq_heatmap(freq: np.ndarray) -> np.ndarray:
    """Ink-frequency map as an RGB heatmap (for the analytics view)."""
    u8 = np.clip(freq * 255, 0, 255).astype(np.uint8)
    cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    bgr = cv2.applyColorMap(u8, cmap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def handwriting_residual(warped_ink: np.ndarray, template_mask: np.ndarray,
                         grow: int = 2, min_blob: int = 6) -> np.ndarray:
    """Ink present in one scan but not in the template = the filled content."""
    k = 2 * grow + 1
    tm = cv2.dilate(template_mask.astype(np.uint8) * 255,
                    np.ones((k, k), np.uint8))
    resid = cv2.bitwise_and(warped_ink, cv2.bitwise_not(tm))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(resid, 8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_blob
    return (keep[labels]) & True


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def pick_reference_index(items: list[tuple[str, bytes]]) -> tuple[int, list]:
    """Pass 1 (cheap): basic metrics on a downscaled copy; sharpest scan wins."""
    metas = []
    for name, data in items:
        img = imdecode_bytes(data)
        if img is None:
            metas.append(dict(name=name, readable=False, blur_var=0.0,
                              brightness=0.0, width=0, height=0))
            continue
        h, w = img.shape[:2]
        small = resize_max(img, 1000)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        metas.append(dict(name=name, readable=True,
                          blur_var=float(cv2.Laplacian(g, cv2.CV_64F).var()),
                          brightness=float(g.mean()), width=w, height=h))
    readable = [i for i, m in enumerate(metas) if m["readable"]]
    if not readable:
        raise ValueError("No readable images were provided.")
    ref_idx = max(readable, key=lambda i: metas[i]["blur_var"])
    return ref_idx, metas


def _probe_concentration(items: list[tuple[str, bytes]], base_p: dict,
                         table: bool, sample: int = 8) -> float:
    """Cheap measure of how well one alignment method stacks the stack.

    Aligns a small sample at reduced resolution (no ECC / line polish) and
    returns the fraction of pixels whose ink frequency is >= 0.7. Higher means
    the printed structure lands on the same pixels across scans -- i.e. the
    method aligns this particular form better.
    """
    sub = items[:min(sample, len(items))]
    pp = {**base_p, "align_method": "table" if table else "orb",
          "table_align": table, "use_ecc": False, "linesnap": False,
          "line_refine": bool(base_p.get("line_refine", True)),
          "max_side": min(1400, int(base_p["max_side"]))}
    res = run_pipeline(sub, pp, keep_example=False)
    return float((res["freq"] >= 0.7).mean())


def run_pipeline(items: list[tuple[str, bytes]], params: dict,
                 progress: Optional[Callable[[str, int, int], None]] = None,
                 keep_example: bool = True) -> dict:
    """Run the full pipeline on (filename, bytes) pairs.

    Returns a dict with: metrics DataFrame, reference name, freq / coverage /
    mean_gray maps, convergence snapshots, and one example aligned scan.
    """
    p = {**DEFAULT_PARAMS, **params}
    tick = progress or (lambda *_: None)

    auto_pick, auto_scores = None, None
    if p.get("align_method", "auto") == "auto" and len(items) >= 3:
        tick("Choosing alignment method", 0, len(items))
        try:
            c_table = _probe_concentration(items, p, table=True)
            c_orb = _probe_concentration(items, p, table=False)
            use_table = (c_table - c_orb) > float(p.get("align_orb_bias", 0.01))
            p["table_align"] = use_table
            auto_pick = "table" if use_table else "orb"
            auto_scores = (round(c_table, 3), round(c_orb, 3))
        except Exception:
            p["table_align"] = bool(p.get("table_align", True))
    elif p.get("align_method") in ("table", "orb"):
        p["table_align"] = (p["align_method"] == "table")

    tick("Scoring scans", 0, len(items))
    ref_idx, metas = pick_reference_index(items)

    ref_name, ref_bytes = items[ref_idx]
    ref_pre = preprocess(imdecode_bytes(ref_bytes), p,
                         deskew=p["deskew_reference"])
    canon_quad = None
    table_align = bool(p["table_align"])
    if table_align:
        ref_pre, canon_quad = _canonical_reference(ref_pre)
        if canon_quad is None:
            table_align = False
    ref = build_reference(ref_name, ref_pre, p)
    shape = ref.gray.shape
    acc = TemplateAccumulator(shape)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    tol = int(p["tolerance"])
    dil_k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8) if tol > 0 else None

    def vote_ink(ink: np.ndarray) -> np.ndarray:
        return cv2.dilate(ink, dil_k) if dil_k is not None else ink

    rows, snapshots, example = [], [], None
    checkpoints = sorted({c for c in CHECKPOINTS if c <= len(items)})

    def snapshot_if_due():
        if checkpoints and acc.n >= checkpoints[0]:
            checkpoints.pop(0)
            mask = extract_template(acc.freq(), acc.coverage(),
                                    p["vote_threshold"], p["min_coverage"],
                                    p["min_blob"], p["bridge"], tol,
                                    low_threshold=p["low_threshold"],
                                    peak_gate=p["peak_gate"],
                                    context_gate=p["context_gate"],
                                    grid_reconstruct=p["grid_reconstruct"],
                                    grid_low=p["grid_low"],
                                    grid_thickness=p["grid_thickness"])
            snapshots.append((acc.n, mask))

    full_valid = np.ones(shape, bool)
    acc.add(ref.gray, vote_ink(ref.ink), full_valid)
    qm = quality_metrics(ref.gray, ref.ink)
    rows.append(dict(name=ref_name, role="reference", status="included",
                     reason="", inliers=len(ref.kp), reproj_err=0.0, ssim=1.0,
                     grid_iou=np.nan,
                     method="table" if table_align else "reference",
                     page_warped=ref_pre.page_warped,
                     skew_deg=round(ref_pre.skew_deg, 2), **qm,
                     width=metas[ref_idx]["width"],
                     height=metas[ref_idx]["height"]))
    snapshot_if_due()

    for i, (name, data) in enumerate(items):
        tick(f"Aligning {name}", i + 1, len(items))
        if i == ref_idx:
            continue
        if not metas[i]["readable"]:
            rows.append(dict(name=name, role="scan", status="rejected",
                             reason="unreadable file", inliers=0,
                             reproj_err=np.nan, ssim=np.nan,
                             grid_iou=np.nan, method="-",
                             page_warped=False, skew_deg=0.0, blur_var=0.0,
                             brightness=0.0, contrast=0.0, ink_ratio=0.0,
                             width=metas[i]["width"],
                             height=metas[i]["height"]))
            continue

        pre = preprocess(imdecode_bytes(data), p, deskew=False)
        qm = quality_metrics(pre.gray, pre.ink)
        if table_align:
            res = align_on_grid(pre, ref, canon_quad, p)
            if not res.ok:
                fb = align_to_reference(pre, ref, p, matcher=bf)
                if fb.ok:
                    fb.method = "table->orb"
                    res = fb
        else:
            res = align_to_reference(pre, ref, p, matcher=bf)
        if res.ok:
            acc.add(res.warped_gray, vote_ink(res.warped_ink), res.valid)
            snapshot_if_due()
            if keep_example and example is None:
                example = dict(name=name, warped_ink=res.warped_ink.copy(),
                               warped_gray=res.warped_gray.copy())
            status, reason = "included", ""
        else:
            status, reason = "rejected", res.reason
        rows.append(dict(name=name, role="scan", status=status, reason=reason,
                         inliers=res.inliers, reproj_err=round(res.reproj_err, 2),
                         ssim=round(res.ssim, 3),
                         grid_iou=(round(res.grid_iou, 3)
                                   if res.grid_iou == res.grid_iou else np.nan),
                         method=res.method, page_warped=pre.page_warped,
                         skew_deg=0.0, **qm,
                         width=metas[i]["width"], height=metas[i]["height"]))

    freq, coverage, mean_gray = acc.freq(), acc.coverage(), acc.mean_gray()
    df = pd.DataFrame(rows)
    for c in ("blur_var", "brightness", "contrast"):
        df[c] = df[c].round(1)
    df["ink_ratio"] = df["ink_ratio"].round(4)
    sig_bands = suggest_signature_bands(freq, mean_gray,
                                        pad_right=p["sig_pad_right"],
                                        pad_left=p["sig_pad_left"],
                                        y_pad_factor=p["sig_y_pad_factor"])

    return dict(metrics=df, ref_name=ref_name, shape=shape, n_used=acc.n,
                freq=freq, coverage=coverage, mean_gray=mean_gray,
                snapshots=snapshots, example=example, params=p,
                signature_bands=sig_bands,
                auto_pick=auto_pick, auto_scores=auto_scores)


def convergence_table(snapshots: list[tuple[int, np.ndarray]],
                      final_mask: np.ndarray) -> pd.DataFrame:
    """How does the template stabilise as more forms are added?"""
    rows, prev = [], None
    for n, mask in snapshots:
        rows.append(dict(n_forms=n,
                         iou_vs_final=round(iou(mask, final_mask), 4),
                         iou_vs_previous=(round(iou(mask, prev), 4)
                                          if prev is not None else np.nan)))
        prev = mask
    return pd.DataFrame(rows)


def realign_one(name: str, data: bytes, ref_name: str, ref_bytes: bytes,
                params: dict) -> tuple[Preproc, Reference, AlignResult]:
    """Re-run preprocessing + alignment for a single scan (inspection view)."""
    p = {**DEFAULT_PARAMS, **params}
    ref_pre = preprocess(imdecode_bytes(ref_bytes), p,
                         deskew=p["deskew_reference"])
    canon_quad = None
    table_align = bool(p["table_align"])
    if table_align:
        ref_pre, canon_quad = _canonical_reference(ref_pre)
        if canon_quad is None:
            table_align = False
    ref = build_reference(ref_name, ref_pre, p)
    if name == ref_name:
        full_valid = np.ones(ref.gray.shape, bool)
        res = AlignResult(True, "", len(ref.kp), 0.0, 1.0, np.nan,
                          "table" if table_align else "reference", np.eye(3),
                          ref.gray, ref.ink, full_valid)
        return ref_pre, ref, res
    pre = preprocess(imdecode_bytes(data), p, deskew=False)
    if table_align:
        res = align_on_grid(pre, ref, canon_quad, p)
        if not res.ok:
            fb = align_to_reference(pre, ref, p)
            if fb.ok:
                fb.method = "table->orb"
                res = fb
    else:
        res = align_to_reference(pre, ref, p)
    return pre, ref, res


def png_bytes(img: np.ndarray) -> bytes:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def ocr_template(template_img: np.ndarray) -> tuple[Optional[pd.DataFrame], str]:
    """Optional: read the printed labels off the extracted template."""
    try:
        import pytesseract
    except ImportError:
        return None, ("pytesseract is not installed "
                      "(pip install pytesseract + the Tesseract binary).")
    try:
        d = pytesseract.image_to_data(template_img,
                                      output_type=pytesseract.Output.DICT)
    except Exception as e:
        return None, f"Tesseract failed: {e}"
    rows = [dict(text=t.strip(), conf=float(c), x=x, y=y, w=w, h=h)
            for t, c, x, y, w, h in zip(d["text"], d["conf"], d["left"],
                                        d["top"], d["width"], d["height"])
            if t.strip() and float(c) > 40]
    return pd.DataFrame(rows), ""
