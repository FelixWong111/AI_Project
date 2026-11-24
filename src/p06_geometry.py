"""Independent candidate image metrology. No detector, catalogue or PASS/FAIL.

Input: one isolated object crop, BGR uint8, original-pixel x/y scales. Output:
crop-local lines/contours, pixel lengths and approximate coplanar millimetres.
Class ID selects geometry only; it never supplies an expected physical size.
"""
from __future__ import annotations
import math
import cv2
import numpy as np

DEFAULT_PARAMETERS = dict(background_offset=35, close_kernel=5, minimum_area=150,
                          canny_low=35, canny_high=100, circle_axis_ratio=.82,
                          circle_residual_fraction=.035, nut_spread_fraction=.12)


class MeasurementFailure(ValueError):
    def __init__(self,reason,diagnostics=None):
        super().__init__(reason)
        self.diagnostics=diagnostics or {}


def require(condition, reason):
    if not condition:raise MeasurementFailure(reason)


def dimension(a, b, scale):
    a,b=np.asarray(a,float),np.asarray(b,float)
    return dict(line_xy=[a.tolist(),b.tolist()],pixels=float(np.linalg.norm(b-a)),
                mm_estimate=float(np.linalg.norm((b-a)/scale)))


def sharp_edge(values, positive):
    """Outermost narrow transition, rejecting broad soft-shadow ramps."""
    signal=values if positive else -values
    peaks=np.flatnonzero((signal[1:-1]>=signal[:-2])&(signal[1:-1]>=signal[2:]))+1
    peaks=[int(i) for i in peaks if signal[i]>max(2.,.4*float(signal.max()))]
    accepted=[]
    for i in peaks:
        left=right=i
        while left>0 and signal[left-1]>.5*signal[i]:left-=1
        while right<len(signal)-1 and signal[right+1]>.5*signal[i]:right+=1
        if right-left<=6:accepted.append(i)
    if not accepted:return None
    return accepted[-1] if positive else accepted[0]


def shaft_sections(samples, normal_positions, midpoint, rough_width, units_per_mm):
    smooth=cv2.GaussianBlur(samples.astype(np.float32),(5,1),1)
    derivatives=np.gradient(smooth,axis=1)
    sections=[]
    for row,g in enumerate(derivatives):
        left=sharp_edge(g[:midpoint],False)
        right=sharp_edge(g[midpoint:],True)
        if left is None or right is None:continue
        right+=midpoint
        if right-left<4 or not .5*rough_width<normal_positions[right]-normal_positions[left]<1.15*rough_width:continue
        sections.append((row,left,right,float(normal_positions[right]-normal_positions[left])))
    require(len(sections)>=max(12,.4*len(samples)),'INSUFFICIENT_SHARP_SHAFT_SECTIONS')
    widths=np.array([s[3] for s in sections]);centres=np.array([(normal_positions[s[1]]+normal_positions[s[2]])/2 for s in sections])
    # Reflection edges may truncate individual sections. Retain a dominant
    # centre/width track, not isolated sharp peaks or the widest shadow outlier.
    typical=float(np.median(widths))
    neighbours=(np.abs(widths[:,None]-widths[None,:])<.18*typical)&(np.abs(centres[:,None]-centres[None,:])<.12*typical)
    mode=int(np.argmax(neighbours.sum(axis=1)))
    keep=(np.abs(widths-widths[mode])<.26*widths[mode])&(np.abs(centres-centres[mode])<.15*widths[mode])
    details=dict(section_widths_mm=(widths/units_per_mm).tolist(),section_retained_fraction=float(keep.sum()/len(samples)))
    if keep.sum()<max(12,.35*len(samples)):
        raise MeasurementFailure('SHAFT_NO_DOMINANT_EDGE_TRACK',details)
    widths=widths[keep];centres=centres[keep];sections=[s for s,k in zip(sections,keep) if k]
    if np.quantile(widths,.9)/np.quantile(widths,.1)>=1.4:
        raise MeasurementFailure('SHAFT_SECTION_WIDTHS_INCONSISTENT',details)
    require(np.median(np.abs(centres-np.median(centres)))<.1*np.median(widths),'SHAFT_SECTION_CENTRES_INCONSISTENT')
    # Upper envelope of many observed axial sections, never a nominal M4/M5 size.
    crest=float(np.quantile(widths,.9))
    selected=min(sections,key=lambda s:abs(s[3]-crest))
    return selected,sections


def silhouette(gray, params):
    require(float(np.ptp(gray))>=5,'NO_FOREGROUND')
    blurred=cv2.GaussianBlur(gray,(3,3),0)
    border=np.r_[gray[0],gray[-1],gray[:,0],gray[:,-1]]
    otsu,_=cv2.threshold(blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    threshold=min(otsu,float(np.median(border))-params['background_offset'])
    require(threshold>0,'INSUFFICIENT_BACKGROUND_CONTRAST')
    mask=(blurred<=threshold).astype(np.uint8)*255
    size=params['close_kernel']
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((size,size),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours=sorted([c for c in contours if cv2.contourArea(c)>=params['minimum_area']],key=cv2.contourArea,reverse=True)
    require(bool(contours),'NO_FOREGROUND')
    center=(gray.shape[1]/2,gray.shape[0]/2)
    centred=[c for c in contours if cv2.pointPolygonTest(c,center,False)>=0]
    if len(contours)>1 and cv2.contourArea(contours[1])>=.15*cv2.contourArea(contours[0]):
        require(len(centred)==1,'MULTIPLE_FOREGROUND_OBJECTS')
    contour=centred[0] if centred else contours[0]
    # A closed, strong-gradient metal edge can exclude a soft Otsu shadow halo.
    edges=cv2.Canny(blurred,params['canny_low'],params['canny_high'])
    edges=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    strong,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    strong=[c for c in strong if cv2.pointPolygonTest(c,center,False)>=0 and
            .5*cv2.contourArea(contour)<cv2.contourArea(c)<1.15*cv2.contourArea(contour)]
    method='OTSU_DARK_SILHOUETTE'
    if strong:
        contour=max(strong,key=cv2.contourArea);method='CLOSED_STRONG_EDGE'
    x,y,w,h=cv2.boundingRect(contour)
    require(x>1 and y>1 and x+w<gray.shape[1]-1 and y+h<gray.shape[0]-1,'FOREGROUND_TOUCHES_CROP')
    return contour,dict(segmentation_threshold=float(threshold),silhouette_method=method)


def axial_edge(gray, xy, guess, diameter, offsets, reason, *, exposed_tip=False):
    """Find an agreed metal-to-background transition on neighbouring axial tracks."""
    reach = max(6, int(.25 * diameter))
    positions = np.arange(guess-reach, guess+reach+1)
    tracks = np.asarray([xy(positions[:, None], offset) for offset in offsets], np.float32)
    require((tracks[:,:,0]>=0).all() and (tracks[:,:,0]<gray.shape[1]-1).all() and
            (tracks[:,:,1]>=0).all() and (tracks[:,:,1]<gray.shape[0]-1).all(), reason)
    values = cv2.remap(gray.astype(np.float32), tracks[:,:,0], tracks[:,:,1], cv2.INTER_LINEAR)
    derivatives = np.gradient(cv2.GaussianBlur(values,(7,1),1.3),axis=1)
    locations = []
    for signal,g in zip(values,derivatives):
        if exposed_tip:
            # An internal bright thread crest can be stronger than the real
            # tip. Require the last edge followed by a sustained background
            # plateau on this track, with no return to dark metal.
            tail = np.median(signal[-4:])
            background = np.median(np.r_[gray[0],gray[-1],gray[:,0],gray[:,-1]])
            require(abs(tail-background)<40,reason)
            peaks = np.flatnonzero((g[1:-1]>=g[:-2]) & (g[1:-1]>g[2:]) & (g[1:-1]>=3))+1
            supported = []
            for j in peaks:
                if j<4 or j+5>=len(g) or max(g[j-3],g[j+3])>=.8*g[j]:
                    continue
                outside = signal[j+3:]
                if (np.quantile(np.abs(outside-tail),.8)<18 and
                        np.median(outside)-np.median(signal[j-4:j-1])>=10):
                    supported.append(j)
            require(supported,reason)
            i = supported[-1]
        else:
            i = int(g.argmax())
        require(3 <= i < len(g)-3 and g[i] >= 3 and
                max(g[i-3],g[i+3]) < .8*g[i], reason)
        locations.append(positions[i])
    require(np.ptp(locations) <= max(4.,.15*diameter), reason)
    return float(np.median(locations))


def body_profile_geometry(samples, normal_positions, mid, body, center, axis, normal,
                          start, width, scale, sref, gray, edge_audit):
    """Locate the shoulder first, then measure a supported section inside the shaft."""
    tip = len(width) - 1
    rough = float(np.quantile(width[body], .9))
    head_width = float(max(width[:len(width)//2]))
    transition = rough + .3 * (head_width - rough)
    head = np.flatnonzero(width[:int(.55 * len(width))] > transition)
    require(len(head) >= 3 and head_width > 1.2 * rough, 'HEAD_SHAFT_SEPARATION_FAILED')
    shoulder = int(head[-1]) + 1
    span = tip - shoulder
    # Exclude both the flared neck and the tip before estimating either edge.
    inside = (body >= shoulder + max(.22 * span, .5 * rough)) & (body <= shoulder + .8 * span)
    require(inside.sum() >= 8, 'SHAFT_SECTION_NOT_CLEAR_OF_HEAD_OR_TIP')
    signal = np.quantile(samples[inside], .25, axis=0).astype(np.float32)
    gradient = np.gradient(cv2.GaussianBlur(signal[None, :], (0, 0), 1.5).ravel())
    # The pooled profile suppresses thread gaps. Require a real section with
    # both same-polarity transitions near the pooled edges before drawing a line.
    edges = []
    for half, offset, side in ((-gradient[:mid], 0, 0), (gradient[mid:], mid, -1)):
        strong = np.flatnonzero(half > max(2., .4 * float(half.max())))
        require(strong.size, 'SHAFT_EDGE_GRADIENT_WEAK')
        group = np.split(strong, np.flatnonzero(np.diff(strong) > 1) + 1)[side]
        edges.append(offset + int(group[np.argmax(half[group])]))
    left, right = edges
    diameter = right - left
    require(.55 * rough < diameter < 1.15 * rough, 'SHAFT_EDGE_AMBIGUOUS_REFLECTION_OR_SHADOW')
    require(head_width > 1.5 * diameter, 'HEAD_SHAFT_SEPARATION_FAILED')
    require(span > 1.4 * diameter, 'SHAFT_TOO_SHORT_OR_WRONG_POSE')
    derivatives = np.gradient(cv2.GaussianBlur(samples[inside], (7, 1), 1.5), axis=1)
    radius = max(2, round(.04 * diameter))
    supported = []
    for edge, sign in ((left, -1), (right, 1)):
        lo, hi = max(1, edge-radius), min(len(signal)-1, edge+radius+1)
        local = sign * derivatives[:, lo:hi]
        strongest = lo + local.argmax(axis=1)
        supported.append((strongest, local.max(axis=1) > max(2., .4 * abs(gradient[edge]))))
    (ls, valid_left), (rs, valid_right) = supported
    valid = valid_left & valid_right
    interior_rows = body[inside]
    if edge_audit:
        profiles = samples[inside]
        rows = np.arange(len(profiles))
        background = np.median(np.r_[gray[0],gray[-1],gray[:,0],gray[:,-1]])
        band = max(4, round(.075*diameter))
        for positions, outward in ((ls,-1), (rs,1)):
            inner = np.clip(positions-outward*band,0,profiles.shape[1]-1)
            outer = np.clip(positions+outward*band,0,profiles.shape[1]-1)
            contrast = profiles[rows,outer] - profiles[rows,inner]
            # A narrow transition must account for a substantial fraction of
            # the contrast. A bright stripe surrounded by dark metal fails the
            # exterior/background test even if its local gradient is strong.
            step = profiles[rows,np.clip(positions+3*outward,0,profiles.shape[1]-1)] - profiles[rows,np.clip(positions-3*outward,0,profiles.shape[1]-1)]
            valid &= ((contrast >= 8) & (step >= .55*contrast) &
                      (np.abs(profiles[rows,outer]-background) <=
                       np.abs(profiles[rows,inner]-background)-4))
        # Insist on multiple supported regions along the shaft, not one glint.
        bands = np.array_split(valid, 3)
        require(all(band.sum() >= max(2,.08*len(band)) for band in bands),
                'SHAFT_EXTERIOR_TRACK_DISCONTINUOUS')
    matches = np.flatnonzero(valid)
    require(len(matches) >= max(3, .1 * inside.sum()), 'SHAFT_PAIRED_EDGES_UNSUPPORTED')
    row = min(matches, key=lambda i: (abs(ls[i]-left) + abs(rs[i]-right),
                                      abs(interior_rows[i] - (shoulder + .5*span))))
    u = interior_rows[row]
    a, b = normal_positions[[ls[row], rs[row]]]
    middle = (a+b)/2

    def xy(axial, transverse):
        return (center + axis*(start+axial) + normal*transverse) * scale/sref

    if edge_audit:
        shoulder = axial_edge(gray, xy, shoulder, diameter, [middle-.64*diameter, middle+.64*diameter],
                              'UNDERHEAD_BASE_NOT_CONFIRMED')
        tip = axial_edge(gray, xy, tip, diameter, [middle-.16*diameter, middle, middle+.16*diameter],
                         'SHAFT_TIP_NOT_CONFIRMED', exposed_tip=True)
        require(shoulder+.5*diameter < u < tip and tip-shoulder > 1.4*diameter,
                'SHAFT_SECTION_NOT_CLEAR_OF_HEAD_OR_TIP')
    dims = dict(underhead_length=dimension(xy(shoulder, middle), xy(tip, middle), scale),
                thread_major_diameter=dimension(xy(u, a), xy(u, b), scale))
    return dims, dict(shaft_profile_columns=int(inside.sum()), paired_edge_sections=len(matches),
                      diameter_method='paired observed gradients on interior shaft; pooled profile seed',
                      underhead_method='two exposed shoulder tracks' if edge_audit else 'silhouette shoulder transition',
                      tip_method='three axial tip tracks' if edge_audit else 'silhouette end',
                      shaft_line_fraction=float((u-shoulder)/span),
                      warning='Candidate visible silhouette. Shoulder, tip and shadow require review.')


def screw_geometry(contour, scale, gray, method):
    # Work in isotropic units (~original pixels), retaining exact inverse mapping.
    sref=float(np.mean(scale))
    x,y,w,h=cv2.boundingRect(contour)
    filled=np.zeros((h,w),np.uint8)
    cv2.drawContours(filled,[contour-np.array([[[x,y]]])],-1,255,-1)
    yy,xx=np.nonzero(filled)
    # Filled cross-sections avoid one-sided/empty bins on oblique pixel outlines.
    points=np.c_[xx+x,yy+y].astype(float)*sref/scale
    center=points.mean(axis=0)
    _,vectors=np.linalg.eigh(np.cov((points-center).T));axis=vectors[:,-1]

    def profile(direction):
        normal=np.array([-direction[1],direction[0]])
        uv=np.stack(((points-center)@direction,(points-center)@normal),axis=1)
        start=float(uv[:,0].min());bins=np.rint(uv[:,0]-start).astype(int)
        lo=np.full(bins.max()+1,np.inf);hi=np.full(len(lo),-np.inf)
        np.minimum.at(lo,bins,uv[:,1]);np.maximum.at(hi,bins,uv[:,1])
        valid=np.isfinite(lo)&np.isfinite(hi)
        require(valid.sum()>.8*len(lo),'DISCONTINUOUS_SCREW_PROFILE')
        ii=np.arange(len(lo));lo=np.interp(ii,ii[valid],lo[valid]);hi=np.interp(ii,ii[valid],hi[valid])
        width=hi-lo
        width=np.convolve(np.pad(width,(2,2),mode='edge'),np.ones(5)/5,mode='valid')
        return start,lo,hi,width,normal

    # The broad end is the head; refine axis with centres of the narrow body.
    for _ in range(3):
        start,lo,hi,width,normal=profile(axis);n=len(width)
        require(n>=20,'SCREW_TOO_SMALL')
        if np.max(width[:max(3,n//3)])<np.max(width[-max(3,n//3):]):
            axis=-axis;start,lo,hi,width,normal=profile(axis)
        body=np.arange(int(.4*n),int(.85*n))
        slope=np.polyfit(body,(lo[body]+hi[body])/2,1)[0]
        require(abs(slope)<.8,'SCREW_AXIS_AMBIGUOUS')
        axis=axis+slope*normal;axis/=np.linalg.norm(axis)
    start,lo,hi,width,normal=profile(axis);n=len(width)
    body=np.arange(int(.4*n),int(.85*n))
    # Strong normal gradients separate metal from the broad, soft shadow halo.
    normal_positions=np.arange(math.floor(lo[body].min())-8,math.ceil(hi[body].max())+9,dtype=float)
    grid=center+axis[None,None,:]*(start+body[:,None,None])+normal[None,None,:]*normal_positions[None,:,None]
    pixel_grid=grid*scale/sref
    samples=cv2.remap(gray.astype(np.float32),pixel_grid[:,:,0].astype(np.float32),pixel_grid[:,:,1].astype(np.float32),cv2.INTER_LINEAR)
    mid=int(np.argmin(np.abs(normal_positions-np.median((lo[body]+hi[body])/2))))
    require(3<mid<len(normal_positions)-4,'SHAFT_METAL_PROFILE_UNRESOLVED')
    if method in ('body_profile','edge_audit'):
        return body_profile_geometry(samples,normal_positions,mid,body,center,axis,normal,
                                     start,width,scale,sref,gray,method=='edge_audit')
    selected,sections=shaft_sections(samples,normal_positions,mid,float(np.quantile(width[body],.9)),sref)
    section_row,left,right,_=selected
    shaft=float(normal_positions[right]-normal_positions[left])
    rough_shaft=float(np.quantile(width[body],.9));head_width=float(np.max(width[:int(.5*n)]))
    require(.55*rough_shaft<shaft<1.15*rough_shaft,'SHAFT_EDGE_AMBIGUOUS_REFLECTION_OR_SHADOW')
    section_u=int(body[section_row])
    def section_xy(v):return (center+axis*(start+section_u)+normal*v)*scale/sref
    partial_diameter=dimension(section_xy(normal_positions[left]),section_xy(normal_positions[right]),scale)
    try:
        transition_level=rough_shaft+.3*(head_width-rough_shaft)
        head=np.flatnonzero(width[:int(.55*n)]>transition_level)
        require(len(head)>=3 and head_width>1.5*shaft and head_width>1.2*rough_shaft,'HEAD_SHAFT_SEPARATION_FAILED')
        shoulder=int(head[-1])+1
        require(shoulder<int(.55*n) and shoulder>1,'UNDERHEAD_PLANE_AMBIGUOUS')
        # A half-diameter clearance also excludes the flared head/neck region
        # when the preliminary silhouette transition lies too far into the head.
        require(shoulder+max(.15*(n-1-shoulder),.5*shaft)<section_u<shoulder+.85*(n-1-shoulder),
                'SHAFT_SECTION_NOT_CLEAR_OF_HEAD_OR_TIP')
        middle=float(np.median([(normal_positions[s[1]]+normal_positions[s[2]])/2 for s in sections]))
        # Independently check the two exposed sides just outside the shaft. A vague
        # width transition alone is not a reliable under-head seating plane.
        search=np.arange(max(0,int(shoulder-.3*shaft)),min(int(.55*n),int(shoulder+.3*shaft)+1))
        require(len(search)>=7,'UNDERHEAD_SEARCH_TOO_SHORT')
        planes=[]
        for sign in (-1,1):
            track=(center+axis[None,:]*(start+search[:,None])+normal[None,:]*(middle+sign*.65*shaft))*scale/sref
            values=cv2.remap(gray.astype(np.float32),track[:,0].astype(np.float32).reshape(1,-1),track[:,1].astype(np.float32).reshape(1,-1),cv2.INTER_LINEAR)
            values=cv2.GaussianBlur(values,(5,1),1).ravel();gradient=np.gradient(values)
            index=int(np.argmax(gradient))
            require(1<index<len(search)-2 and gradient[index]>3,'UNDERHEAD_EDGE_NOT_VISIBLE_BOTH_SIDES')
            half=np.flatnonzero(gradient>.5*gradient[index])
            group=next(g for g in np.split(half,np.flatnonzero(np.diff(half)>1)+1) if index in g)
            require(len(group)<=7,'UNDERHEAD_EDGE_TOO_DIFFUSE')
            planes.append(int(search[index]))
        require(abs(planes[0]-planes[1])<max(4,.15*shaft),'UNDERHEAD_TWO_SIDES_DISAGREE')
        shoulder=float(np.mean(planes))
        shaft_end=n-1
        require(shaft_end-shoulder>1.4*shaft,'SHAFT_TOO_SHORT_OR_WRONG_POSE')
        stable=np.arange(int(shoulder+.15*(shaft_end-shoulder)),int(shoulder+.85*(shaft_end-shoulder)))
    except MeasurementFailure as error:
        # An unresolved head/body identity cannot support even a partial diameter.
        if str(error) not in ('SHAFT_SECTION_NOT_CLEAR_OF_HEAD_OR_TIP','HEAD_SHAFT_SEPARATION_FAILED','UNDERHEAD_PLANE_AMBIGUOUS'):
            error.diagnostics.update(partial_dimensions={'thread_major_diameter':partial_diameter},
                                     partial_requires_review=True)
        error.diagnostics.update(
                                 section_widths_mm=[float(s[3]/sref) for s in sections[::max(1,len(sections)//64)]],
                                 sharp_section_count=len(sections),shaft_section_count=len(samples))
        raise
    j=int(body[section_row])
    def xy(u,v):return (center+axis*(start+u)+normal*v)*scale/sref
    dims=dict(underhead_length=dimension(xy(shoulder,middle),xy(shaft_end,middle),scale),
              thread_major_diameter=dimension(xy(j,normal_positions[left]),xy(j,normal_positions[right]),scale))
    require(dims['thread_major_diameter']['pixels']>0,'SHAFT_WIDTH_UNRESOLVED')
    shoulder_line=[xy(shoulder,middle-.65*shaft).tolist(),xy(shoulder,middle+.65*shaft).tolist()]
    return dims,dict(head_shaft_boundary_xy=shoulder_line,shaft_profile_columns=len(stable),
                     shaft_width_p10_mm=float(np.quantile(width[stable],.1)/sref),
                     dark_silhouette_width_p90_mm=float(np.quantile(width[stable],.9)/sref),
                     metal_profile_diameter_mm=float(shaft/sref),diameter_method='90th-percentile envelope of sharp per-section observed edges',
                     section_widths_mm=[float(s[3]/sref) for s in sections[::max(1,len(sections)//64)]],
                     sharp_section_count=len(sections),shaft_section_count=len(samples),
                     underhead_side_positions_work_pixels=planes,underhead_method='two exposed axial intensity-edge tracks',
                     warning='Visible silhouette estimate: thread crests/shadow and unconfirmed head style require caliper checks.')


def radial_aperture(gray, scale, outer, fit):
    """Search outward from the visible background through the first dark hole rim."""
    angles = np.arange(72) * (2*np.pi/72)
    rays = np.c_[np.cos(angles),np.sin(angles)]
    radius, center = outer['radius'], outer['center']
    radii = np.arange(0, .74*radius, 1/np.mean(scale))
    require(len(radii)>=16, 'APERTURE_TOO_SMALL_TO_SAMPLE')
    grid = ((center + rays[:,None,:]*radii[None,:,None])*scale).astype(np.float32)
    require((grid[:,:,0]>=0).all() and (grid[:,:,0]<gray.shape[1]-1).all() and
            (grid[:,:,1]>=0).all() and (grid[:,:,1]<gray.shape[0]-1).all(), 'RADIAL_TRACK_OUTSIDE_CROP')
    profiles = cv2.remap(gray.astype(np.float32), grid[:,:,0], grid[:,:,1], cv2.INTER_LINEAR)
    profiles = cv2.GaussianBlur(profiles,(7,1),1.3)
    core_samples = profiles[:,:max(2,len(radii)//10)]
    core = np.median(core_samples)
    noise = 1.4826*np.median(np.abs(core_samples-core))
    bg = np.median(np.r_[gray[0],gray[-1],gray[:,0],gray[:,-1]])
    require(abs(core-bg)<=30, 'APERTURE_CENTER_BACKGROUND_UNCONFIRMED')
    first_returns, extra_gaps = [], []
    for ray, (values, g) in enumerate(zip(profiles,-np.gradient(profiles,axis=1))):
        peaks = np.flatnonzero((g[1:-1]>=g[:-2]) & (g[1:-1]>g[2:]) & (g[1:-1]>=3))+1
        hits = []
        for i in peaks:
            if i<6 or i>=len(g)-6 or radii[i]<=.1*radius:
                continue
            inside = np.median(values[i-6:i-2])
            outside = np.median(values[i+2:i+6])
            if (inside >= core-max(45,6*noise) and outside < core-max(22,5*noise)
                    and inside-outside >= max(18,4*noise)):
                hits.append(int(i))
        if hits:
            first = hits[0]
            first_returns.append((ray,first))
            # A competing rim requires a return to core brightness in between
            # and a similar spacing around the annulus. Texture peaks or a
            # monotonic darkening wall do not establish a second concentric rim.
            for j in hits[1:]:
                gap = j-first
                if 6<gap<.18*radius*np.mean(scale) and np.median(values[j-5:j-2])>=core-max(12,3*noise):
                    extra_gaps.append(gap)
                    break
    require(len(first_returns)>=54,'RADIAL_APERTURE_SUPPORT_LOW')
    gaps = np.asarray(extra_gaps)
    if len(gaps):
        require(max(np.sum(abs(gaps-g)<=2) for g in gaps)<.4*len(first_returns),
                'COMPETING_CONCENTRIC_RIMS')
    rays, radii_idx = np.array(first_returns).T
    median = np.median(radii_idx)
    keep = np.abs(radii_idx-median) < max(.18*median,3.5*np.median(np.abs(radii_idx-median)))
    rays, radii_idx = rays[keep], radii_idx[keep]
    require(len(rays)>=54 and len(set(rays//6))==12,'RADIAL_APERTURE_SUPPORT_LOW')
    points = grid[rays,radii_idx].reshape(-1,1,2)
    fitted = fit(points)
    require(np.linalg.norm(fitted['center']-center)<.12*radius and
            .12*radius<fitted['radius']<.7*radius and fitted['full_residual']<.12,
            'RADIAL_APERTURE_INCONSISTENT')
    return points, fitted, len(points)


def washer_geometry(gray, scale, params, radial=False):
    # A true inner hole must be a child of the centred annular silhouette.
    # Do not search arbitrary circular highlights on the metal face.
    blurred=cv2.GaussianBlur(gray,(5,5),1)
    _,mask=cv2.threshold(blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    contours,hierarchy=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    require(hierarchy is not None,'INNER_OUTER_CIRCULAR_EDGES_NOT_RESOLVED')
    center=(gray.shape[1]/2,gray.shape[0]/2)
    parents=hierarchy[0,:,3]
    outer_indices=[i for i,c in enumerate(contours) if parents[i]<0 and len(c)>=24
                   and cv2.contourArea(c)>=params['minimum_area']
                   and cv2.pointPolygonTest(c,center,False)>=0]
    require(len(outer_indices)==1,'ANNULAR_OUTER_BOUNDARY_NOT_RESOLVED')
    outer_index=outer_indices[0]

    def fit(c,outer_boundary=False):
        points=c[:,0].astype(float)/scale
        # A bright annulus can connect a face-texture branch to the outer edge.
        # Seed from the convex hull, then retain observed boundary support only.
        seed=cv2.convexHull(points.astype(np.float32).reshape(-1,1,2)) if outer_boundary else points.astype(np.float32).reshape(-1,1,2)
        require(len(seed)>=5 and cv2.contourArea(cv2.convexHull(seed))>0,'ELLIPSE_SUPPORT_DEGENERATE')
        ellipse=cv2.fitEllipse(seed)
        keep=np.ones(len(points),bool)
        full_residual=None
        for iteration in range(3):
            ctr=np.array(ellipse[0]);a,b=np.array(ellipse[1])/2;angle=math.radians(ellipse[2])
            require(np.isfinite([*ctr,a,b,angle]).all() and min(a,b)>0,'ELLIPSE_SUPPORT_DEGENERATE')
            q=points-ctr;u=q@np.array([math.cos(angle),math.sin(angle)]);v=q@np.array([-math.sin(angle),math.cos(angle)])
            error=np.abs(np.sqrt((u/a)**2+(v/b)**2)-1)
            if iteration==0:full_residual=float(np.sqrt(np.mean(error**2)))
            keep=error < .06 if outer_boundary else error<=np.quantile(error,.85)
            require(keep.sum()>=24,'CIRCULAR_EDGE_SUPPORT_TOO_SMALL')
            # Require almost a whole observed perimeter; never extrapolate a
            # small specular arc into an accepted hole/outer boundary.
            angles=np.mod(np.arctan2(v[keep]/b,u[keep]/a),2*np.pi)
            bins=np.unique(np.floor(angles/(2*np.pi)*36).astype(int))
            require(len(bins)>=30,'CIRCULAR_EDGE_ANGULAR_COVERAGE_LOW')
            ellipse=cv2.fitEllipse(points[keep].astype(np.float32).reshape(-1,1,2))
        axes=sorted(ellipse[1])
        require(np.isfinite([*ellipse[0],*axes,ellipse[2]]).all() and axes[0]>0,'ELLIPSE_SUPPORT_DEGENERATE')
        require(axes[0]/axes[1]>=params['circle_axis_ratio'],'CIRCULAR_FACE_TILTED_OR_INCOMPLETE')
        ctr=np.array(ellipse[0]);a,b=np.array(ellipse[1])/2;angle=math.radians(ellipse[2])
        q=points[keep]-ctr;u=q@np.array([math.cos(angle),math.sin(angle)]);v=q@np.array([-math.sin(angle),math.cos(angle)])
        residual=float(np.sqrt(np.mean((np.sqrt((u/a)**2+(v/b)**2)-1)**2)))
        require(residual<params['circle_residual_fraction'],'CIRCULAR_EDGE_IRREGULAR')
        direction=np.array([math.cos(angle),math.sin(angle)])
        if b>=a:direction=np.array([-math.sin(angle),math.cos(angle)])
        return dict(center=ctr,radius=max(a,b),residual=residual,axes=axes,direction=direction,
                    full_residual=full_residual,support_fraction=float(keep.mean()),angular_bins=len(bins))

    outer=contours[outer_index];ofit=fit(outer,True);oc,orr=ofit['center'],ofit['radius']
    holes=[];rejected_holes=[]
    for i,c in enumerate(contours):
        ancestor=i;depth=0
        while ancestor>=0 and ancestor!=outer_index:
            ancestor=int(parents[ancestor]);depth+=1
        if ancestor!=outer_index or depth%2!=1 or len(c)<24 or cv2.contourArea(c)<params['minimum_area']:continue
        if cv2.pointPolygonTest(c,tuple(oc*scale),False)<0:continue
        try:
            fitted=fit(c);ic,irr=fitted['center'],fitted['radius']
        except MeasurementFailure as error:
            rejected_holes.append(str(error));continue
        if .12*orr<irr<.7*orr and np.linalg.norm(ic-oc)<.12*orr:
            holes.append((c,fitted,depth))
    if radial:
        inner, ifit, radial_support = radial_aperture(gray,scale,ofit,fit)
        # A fitted threshold contour provides an independent check when it
        # exists. Disagreeing outlines or multiple plausible nested holes
        # cannot establish which physical rim is the aperture.
        require(len(holes)<=1,'INNER_HOLE_NOT_UNAMBIGUOUS')
        depth = None
        if holes:
            _, hierarchy_fit, depth = holes[0]
            require(abs(hierarchy_fit['radius']-ifit['radius']) < max(2/np.mean(scale), .08*ifit['radius'])
                    and np.linalg.norm(hierarchy_fit['center']-ifit['center']) < .1*ifit['radius'],
                    'RADIAL_AND_CONTOUR_HOLE_DISAGREE')
    else:
        if len(holes)!=1:
            raise MeasurementFailure('INNER_HOLE_NOT_UNAMBIGUOUS',dict(aperture_candidates=len(holes),rejected_aperture_fits=rejected_holes))
        inner,ifit,depth=holes[0]
        radial_support = None
    dims={}
    for key,fitted in [('outer_diameter',ofit),('inner_diameter',ifit)]:
        ctr,rad,direction=fitted['center'],fitted['radius'],fitted['direction']
        dims[key]=dimension((ctr-rad*direction)*scale,(ctr+rad*direction)*scale,scale)
    return dims,dict(outer_ellipse_rms_fraction=ofit['residual'],inner_ellipse_rms_fraction=ifit['residual'],
                     outer_ellipse_axes_mm=ofit['axes'],inner_ellipse_axes_mm=ifit['axes'],
                     outer_full_contour_residual=ofit['full_residual'],inner_full_contour_residual=ifit['full_residual'],
                     outer_edge_support_fraction=ofit['support_fraction'],inner_edge_support_fraction=ifit['support_fraction'],
                     aperture_hierarchy_depth=depth,
                     radial_aperture_supported_rays=radial_support,
                     diameter_definition='visible fitted ellipse major axis; minor axis retained; not pose-corrected physical diameter',
                     aperture_method='first supported radial hole boundary; robust ellipse fit' if radial else 'centred odd-depth descendant aperture; robust observed-perimeter ellipse fit',
                     warning='Visible projected axes only. Hole wall, chamfer, shadow and projection require caliper verification.'),[outer,inner]


def nut_geometry(contour, scale, params):
    points=contour[:,0].astype(np.float32)/scale.astype(np.float32)
    hull=cv2.convexHull(points.reshape(-1,1,2));perimeter=cv2.arcLength(hull,True)
    polygons=[cv2.approxPolyDP(hull,f*perimeter,True)[:,0] for f in (.015,.02,.025,.03,.04,.05)]
    polygon=next((p for p in polygons if len(p)==6),None)
    require(polygon is not None,'SIX_FLATS_NOT_RESOLVED')
    lines=[]
    for i in range(6):
        a,b=polygon[i],polygon[(i+1)%6];direction=b-a;length=float(np.linalg.norm(direction));direction/=length
        normal=np.array([-direction[1],direction[0]])
        along=(points-a)@direction;distance=np.abs((points-a)@normal)
        selected=points[(along>.15*length)&(along<.85*length)&(distance<.12*length)]
        require(len(selected)>=6,'INSUFFICIENT_STRAIGHT_FLAT_EDGE')
        fitted=cv2.fitLine(selected,cv2.DIST_HUBER,0,.001,.001).ravel()
        unit=fitted[:2];anchor=fitted[2:];lines.append((unit,anchor))
    values=[];segments=[]
    for i in range(3):
        u,a=lines[i];v,b=lines[i+3]
        require(abs(float(u@v))>math.cos(math.radians(12)),'OPPOSITE_FLATS_NOT_PARALLEL')
        normal=np.array([-u[1],u[0]])
        separation=float((b-a)@normal);endpoint=a+separation*normal
        values.append(abs(separation));segments.append((a*scale,endpoint*scale))
    med=float(np.median(values))
    require((max(values)-min(values))/med<params['nut_spread_fraction'],'NUT_TILT_OR_SHADOW_ASYMMETRY')
    index=int(np.argmin(np.abs(np.array(values)-med)))
    return dict(across_flats=dimension(*segments[index],scale)),dict(
        opposite_flats_mm=values,opposite_flat_lines_xy=[[a.tolist(),b.tolist()] for a,b in segments],
        hexagon_xy=(polygon*scale).tolist(),warning='Visible parallel-flat separation, not across-corners or thread-hole diameter.')


def measure_object(image_bgr, class_id, scale, parameters=None, *, screw_method='body_profile', radial_washer=False):
    if screw_method not in ('body_profile', 'stage1', 'edge_audit'):
        raise ValueError('Unknown screw measurement method')
    if not (isinstance(image_bgr,np.ndarray) and image_bgr.dtype==np.uint8 and image_bgr.ndim==3
            and image_bgr.shape[2]==3 and min(image_bgr.shape[:2])>=10):raise ValueError('Need uint8 HxWx3 BGR crop')
    if type(class_id) is not int or class_id not in range(6):raise ValueError('class_id must be 0..5')
    scale=np.asarray(scale,float)
    if scale.shape!=(2,) or not np.isfinite(scale).all() or np.min(scale)<=0:raise ValueError('Need finite positive x/y px_per_mm')
    params=dict(DEFAULT_PARAMETERS if parameters is None else parameters)
    if set(params)!=set(DEFAULT_PARAMETERS):raise ValueError('Unknown/missing algorithm parameters')
    for name in params:
        if not isinstance(params[name],(int,float)) or not math.isfinite(params[name]) or params[name]<=0:
            raise ValueError('Invalid algorithm parameter: '+name)
    if type(params['close_kernel']) is not int or params['close_kernel']%2!=1:raise ValueError('close_kernel must be a positive odd integer')
    result=dict(status='UNMEASURABLE',reason='',class_id=class_id,decision=None,dimensions={},
                contours_xy=[],diagnostics={},scale_px_per_mm=scale.tolist(),
                scope='candidate_coplanar_visible_geometry; no product tolerance or final decision')
    if class_id==0:
        result['reason']='NON_CATALOGUE_NOT_MEASURED';return result
    gray=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2GRAY)
    try:
        if class_id==5:
            # Check object isolation; bright annuli may have only a narrow dark rim.
            contour,diag=silhouette(gray,params)
            result['diagnostics'].update(diag)
            result['contours_xy']=[contour[:,0].tolist()]
            dims,details,contours=washer_geometry(gray,scale,params,radial_washer)
            result['contours_xy']=[c[:,0].tolist() for c in contours]
        else:
            contour,diag=silhouette(gray,params)
            result['diagnostics'].update(diag)
            result['contours_xy']=[contour[:,0].tolist()]
            if class_id in (1,2,3):dims,details=screw_geometry(contour,scale,gray,screw_method)
            else:dims,details=nut_geometry(contour,scale,params)
        for dim in dims.values():
            line=np.array(dim['line_xy'])
            require(np.isfinite(line).all() and dim['pixels']>0 and dim['mm_estimate']>0,'DEGENERATE_MEASUREMENT_LINE')
            require(np.all(line>=0) and np.all(line[:,0]<gray.shape[1]) and np.all(line[:,1]<gray.shape[0]),'MEASUREMENT_LINE_OUTSIDE_CROP')
        result.update(status='MEASURED_CANDIDATE',dimensions=dims,diagnostics={**diag,**details})
    except MeasurementFailure as error:
        result['reason']=str(error)
        result['diagnostics'].update(error.diagnostics)
    return result
