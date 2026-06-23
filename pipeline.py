from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd


DEFAULT_PARAMS = dict(
    max_side=4000,
    page_detect=False,
    deskew_reference=True,
    deskew_scans=False,
    binarize="adaptive",
    adaptive_block=35,
    adaptive_C=15,
    sharpen=0.0,
    orb_features=8000,
    orb_on_lines=False,
    match_ratio=0.75,
    ransac_thresh=4.0,
    min_inliers=30,
    use_ecc=True,
    align_method="auto",
    align_orb_bias=0.01,
    table_align=True,
    min_grid_iou=0.0,
    line_refine=True,
    tolerance=1,
    vote_threshold=0.60,
    min_coverage=0.40,
    min_blob=0,
    bridge=0,
    sig_pad_left=0.45,
    sig_pad_right=0.33,
    sig_y_pad_factor=0.9,
)


def imdecode_bytes(data: bytes) -> Optional[np.ndarray]:
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
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], np.float32)


def four_point_warp(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = quad
    w = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    h = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    w, h = max(w, 32), max(h, 32)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderValue=255)


def detect_page_quad(gray: np.ndarray) -> Optional[np.ndarray]:
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
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)
    bg = np.maximum(bg, 1)
    return cv2.divide(gray, bg, scale=255)


def unsharp_mask(gray: np.ndarray, amount: float,
                 radius: float = 1.2) -> np.ndarray:
    if amount <= 0:
        return gray
    blur = cv2.GaussianBlur(gray, (0, 0), radius)
    return cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)


def estimate_skew_deg(ink: np.ndarray) -> float:
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

    s = ssim_gray(ref.gray, warped_gray)
    return AlignResult(True, "", inliers=inliers, reproj_err=err, ssim=s,
                       method="orb", H=H, warped_gray=warped_gray,
                       warped_ink=warped_ink, valid=valid)


def _ecc_refine(ref_gray, warped_gray, warped_ink, valid):
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


def detect_grid_quad(gray: np.ndarray) -> Optional[np.ndarray]:
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


def _canon_rect_from_quad(quad: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = quad
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    x0, y0 = quad.min(0)
    return np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]],
                    np.float32)


def _warp_quad(gray, ink, src_quad, dst_quad, out_wh):
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
    quad = detect_grid_quad(pre.gray)
    if quad is None:
        return AlignResult(False, "no table frame found", method="table")
    g, i, v, H = _warp_quad(pre.gray, pre.ink, quad, canon_quad, ref.size_wh)
    if not _homography_sane(H, pre.gray.shape, ref.size_wh):
        return AlignResult(False, "degenerate table fit", method="table")
    if params.get("line_refine", True):
        g, i, v = _line_refine(ref.gray, g, i, v)
    gi = _mask_iou(_printed_line_mask(ref.gray), _printed_line_mask(g))
    s = ssim_gray(ref.gray, g)
    ok = gi >= float(params.get("min_grid_iou", 0.0))
    return AlignResult(ok, "" if ok else f"low grid overlap ({gi:.2f})",
                       inliers=0, reproj_err=0.0, ssim=s, grid_iou=gi,
                       method="table", H=H, warped_gray=g, warped_ink=i,
                       valid=v)


def _canonical_reference(ref_pre: Preproc):
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


def _grid_line_centers(line_mask: np.ndarray, axis: int) -> list[float]:
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


def _clean_signature_band(band: np.ndarray, y_protect: int,
                          t: int = 2,
                          row_threshold: float = 0.90,
                          bridge_ratio: int = 5,
                          border_divisor: int = 80) -> np.ndarray:
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
                     signature_bands: Optional[list] = None,
                     sig_clean_threshold: float = 0.90,
                     sig_bridge_ratio: int = 5,
                     sig_border_divisor: int = 80) -> np.ndarray:
    covered = coverage >= min_coverage
    mask = (freq >= vote_threshold) & covered

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

    mask = m > 0
    if signature_bands:
        mask = clean_signature_columns(mask, signature_bands,
                                       t=2,
                                       row_threshold=sig_clean_threshold,
                                       bridge_ratio=sig_bridge_ratio,
                                       border_divisor=sig_border_divisor)
    return mask


def render_mask(mask: np.ndarray) -> np.ndarray:
    out = np.full(mask.shape, 255, np.uint8)
    out[mask] = 0
    return out


def freq_heatmap(freq: np.ndarray) -> np.ndarray:
    u8 = np.clip(freq * 255, 0, 255).astype(np.uint8)
    cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    bgr = cv2.applyColorMap(u8, cmap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def handwriting_residual(warped_ink: np.ndarray, template_mask: np.ndarray,
                         grow: int = 2, min_blob: int = 6) -> np.ndarray:
    k = 2 * grow + 1
    tm = cv2.dilate(template_mask.astype(np.uint8) * 255,
                    np.ones((k, k), np.uint8))
    resid = cv2.bitwise_and(warped_ink, cv2.bitwise_not(tm))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(resid, 8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_blob
    return (keep[labels]) & True


def pick_reference_index(items: list[tuple[str, bytes]]) -> tuple[int, list]:
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
    sub = items[:min(sample, len(items))]
    pp = {**base_p, "align_method": "table" if table else "orb",
          "table_align": table, "use_ecc": False,
          "line_refine": bool(base_p.get("line_refine", True)),
          "max_side": min(1400, int(base_p["max_side"]))}
    res = run_pipeline(sub, pp, keep_example=False)
    return float((res["freq"] >= 0.7).mean())


def run_pipeline(items: list[tuple[str, bytes]], params: dict,
                 progress: Optional[Callable[[str, int, int], None]] = None,
                 keep_example: bool = True) -> dict:
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

    rows, example = [], None

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
                example=example, params=p,
                signature_bands=sig_bands,
                auto_pick=auto_pick, auto_scores=auto_scores)


def realign_one(name: str, data: bytes, ref_name: str, ref_bytes: bytes,
                params: dict) -> tuple[Preproc, Reference, AlignResult]:
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
