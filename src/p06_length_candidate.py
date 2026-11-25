"""Candidate screw length from the visible head rim and tip of a single side-on screw."""
import cv2
import numpy as np

from src.p06_geometry import dimension, require


def _connected_edges(gray, box):
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.1)
    edges = cv2.Canny(blurred, 38, 85)
    # Thread crests and the thin lip may be interrupted by specular reflections.
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x1, y1, x2, y2 = box
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 150:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        overlap = max(0, min(x+w, x2)-max(x, x1)) * max(0, min(y+h, y2)-max(y, y1))
        if overlap / (w*h) > .85:
            candidates.append((area, contour))
    require(candidates, 'TARGET_EDGE_COMPONENT_MISSING')
    candidates.sort(key=lambda item: item[0], reverse=True)
    area, contour = candidates[0]
    require(len(candidates) == 1 or candidates[1][0] < .25*area, 'TARGET_EDGE_COMPONENT_AMBIGUOUS')
    x, y, w, h = cv2.boundingRect(contour)
    require(x > 1 and y > 1 and x+w < gray.shape[1]-1 and y+h < gray.shape[0]-1,
            'EDGE_COMPONENT_TOUCHES_CROP')
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [contour - [[[x,y]]]], -1, 255, -1)
    ys, xs = np.nonzero(mask)
    return np.c_[xs+x, ys+y]


def _profile(points, axis):
    normal = np.array([-axis[1], axis[0]])
    u = points @ axis
    v = points @ normal
    start = u.min()
    bins = np.rint(u - start).astype(int)
    low = np.full(bins.max()+1, np.inf)
    high = np.full(len(low), -np.inf)
    np.minimum.at(low, bins, v)
    np.maximum.at(high, bins, v)
    indices = np.arange(len(low))
    valid = np.isfinite(low)
    return (np.interp(indices, indices[valid], low[valid]),
            np.interp(indices, indices[valid], high[valid]), start, normal)


def underhead_length(image_bgr, scale, *, target_box=None):
    """Return a crop-local candidate dimension; raise MeasurementFailure for ambiguity."""
    if (not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8 or
            image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or min(image_bgr.shape[:2]) < 10):
        raise ValueError('Need a uint8 BGR object crop')
    scale = np.asarray(scale, dtype=float)
    if scale.shape != (2,) or not np.isfinite(scale).all() or min(scale) <= 0:
        raise ValueError('Need finite positive x/y scales')
    height, width = image_bgr.shape[:2]
    box = np.asarray((0, 0, width, height) if target_box is None else target_box, dtype=float)
    if (box.shape != (4,) or not np.isfinite(box).all() or
            not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)):
        raise ValueError('Invalid target box')
    points = _connected_edges(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), box)
    center = points.mean(axis=0)
    points = points - center
    _, axes = np.linalg.eigh(np.cov(points.T))
    axis = axes[:, -1]
    for iteration in range(3):
        low, high, start, normal = _profile(points, axis)
        extent = high - low
        n = len(low)
        require(n >= 30, 'AXIAL_SUPPORT_TOO_SHORT')
        end = max(4, n//3)
        if np.quantile(extent[:end], .9) < np.quantile(extent[-end:], .9):
            axis = -axis
            low, high, start, normal = _profile(points, axis)
        body = np.arange(int(.42*n), int(.84*n))
        slope = np.polyfit(body, (high[body]+low[body])/2, 1)[0]
        require(abs(slope) < .55, 'SCREW_AXIS_AMBIGUOUS')
        if iteration < 2:
            axis = axis + slope*normal
            axis /= np.linalg.norm(axis)
    middle = np.median((high[body] + low[body])/2)
    radius = np.median((high[body] - low[body])/2)
    require(radius > 3, 'SHAFT_EDGE_SUPPORT_LOW')
    u = points @ axis - start
    v = points @ normal - middle
    head = u < .48*n
    shoulders = []
    for side in (-1, 1):
        side_v = side*v
        head_radius = np.quantile(side_v[head], .99)
        require(head_radius > 1.22*radius, 'HEAD_LIP_NOT_EXPOSED')
        levels = radius + (head_radius-radius)*np.linspace(.32, .75, 7)
        ends = []
        for level in levels:
            selected = head & (abs(side_v-level) < 1.5)
            if selected.any():
                ends.append(u[selected].max())
        require(len(ends) >= 5 and np.ptp(ends) < .55*radius, 'HEAD_RIM_DISCONTINUOUS')
        shoulders.append(np.median(ends))
    require(abs(shoulders[1]-shoulders[0]) < .8*radius, 'HEAD_RIMS_DISAGREE')
    tip = []
    for level in np.linspace(-.35*radius, .35*radius, 7):
        selected = abs(v-level) < 1.5
        if selected.any():
            tip.append(u[selected].max())
    require(len(tip) >= 5 and np.ptp(tip) < .35*radius, 'TIP_EDGE_DISCONTINUOUS')
    start_u, end_u = np.mean(shoulders), np.median(tip)
    require(end_u - start_u > 2*radius, 'SHAFT_LENGTH_UNRESOLVED')
    origin = center + start*axis + middle*normal
    return dimension(origin + start_u*axis, origin + end_u*axis, scale)
