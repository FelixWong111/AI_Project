"""Candidate full-object axial projection. No nominal dimensions or decisions."""
import cv2
import numpy as np

from src.p06_geometry import DEFAULT_PARAMETERS, MeasurementFailure, isolate_target, silhouette


def axial_projection(image_bgr, target_box, scale):
    scale = np.asarray(scale, float)
    if scale.shape != (2,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Need positive finite x/y scale')
    box = np.asarray(target_box, float)
    if (box.shape != (4,) or not np.isfinite(box).all() or
            not 0 <= box[0] < box[2] <= image_bgr.shape[1] or
            not 0 <= box[1] < box[3] <= image_bgr.shape[0]):
        raise ValueError('Invalid crop-local target box')
    box = np.r_[np.floor(box[:2]), np.ceil(box[2:])].astype(int)
    result = dict(status='UNMEASURABLE', reason='', decision=None)
    try:
        isolated = isolate_target(image_bgr, box, DEFAULT_PARAMETERS)
        gray = cv2.cvtColor(isolated, cv2.COLOR_BGR2GRAY)
        contour, diagnostics = silhouette(gray, DEFAULT_PARAMETERS)
        x,y,w,h = cv2.boundingRect(contour)
        mask = np.zeros((h,w),np.uint8)
        cv2.drawContours(mask,[contour - [[[x,y]]]],-1,1,-1)
        yy,xx = np.nonzero(mask)
        points = np.c_[xx+x,yy+y] / scale
        center = points.mean(axis=0)
        values, vectors = np.linalg.eigh(np.cov((points-center).T))
        if values[-1] <= 1.2 * values[0]:
            raise MeasurementFailure('PROJECTION_AXIS_AMBIGUOUS')
        axis = vectors[:,-1]
        projected = (points-center) @ axis
        low,high = float(projected.min()),float(projected.max())
        line = np.array([center+low*axis,center+high*axis])*scale
        trimmed = np.quantile(projected,[.005,.995])
        result.update(status='PROJECTED_CANDIDATE',line_xy=line.tolist(),
                      pixels=float(np.linalg.norm(line[1]-line[0])),
                      mm_estimate=high-low,trimmed_projection_mm=float(trimmed[1]-trimmed[0]),
                      axis_eigenvalue_ratio=float(values[-1]/values[0]),
                      contours_xy=[contour[:,0].tolist()],diagnostics=diagnostics,
                      definition='full visible silhouette projection along filled-mask PCA axis; not under-head length')
    except MeasurementFailure as error:
        result['reason'] = str(error)
    return result
