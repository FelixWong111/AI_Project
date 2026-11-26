"""Independent ruler/geometry utilities; never substitutes nominal labels for measured lengths."""
from __future__ import annotations
import math
import cv2
import numpy as np


def ruler_ticks(image):
    """Fit a detected physical 1-mm tick lattice; absolute ruler origin is not needed for scale."""
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    small=cv2.resize(gray,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
    lines=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(small)[0]
    if lines is None:raise ValueError('No ruler line candidates')
    lines=lines[:,0,:].astype(float)*2
    delta=lines[:,2:]-lines[:,:2];length=np.linalg.norm(delta,axis=1)
    angle=np.mod(np.arctan2(delta[:,1],delta[:,0]),np.pi)
    keep=(length>25)&(length<360);lines=lines[keep];length=length[keep];angle=angle[keep]
    hist,_=np.histogram(angle,bins=180,range=(0,np.pi),weights=length)
    # Smooth neighboring bins to absorb a slightly tilted printed tick family.
    smooth=sum(np.roll(hist,i) for i in range(-2,3));mode=int(np.argmax(smooth))
    theta=(mode+.5)*np.pi/180
    distance=np.abs(np.angle(np.exp(2j*(angle-theta))))/2
    selected=distance<np.deg2rad(3)
    lines=lines[selected];length=length[selected];angle=angle[selected]
    theta=.5*np.angle(np.sum(length*np.exp(2j*angle)))
    direction=np.array([math.cos(theta),math.sin(theta)])
    tick_direction=direction.copy()
    axis=np.array([-direction[1],direction[0]])
    endpoints=lines.reshape(-1,2)
    # Perspective can make ruler edges and printed ticks non-orthogonal. Fit
    # their common endpoint line instead of forcing it perpendicular to ticks.
    original_angle=math.atan2(axis[1],axis[0]);best=None
    for shift in np.linspace(-8,8,161):
        a=original_angle+math.radians(shift);v=np.array([math.cos(a),math.sin(a)])
        normal=np.array([v[1],-v[0]])
        if normal@direction<0:normal=-normal
        q=endpoints@normal;edges=np.arange(q.min()-4,q.max()+9,4)
        h,_=np.histogram(q,bins=edges);i=int(h.argmax());center=(edges[i]+edges[i+1])/2
        close=abs(q-center)<5
        score=(int(close.sum()),float(np.ptp(endpoints[close]@v)) if close.any() else 0)
        if best is None or score>best[0]:best=(score,v,normal,center,close)
    _,axis,direction,baseline,near=best
    for _ in range(3):
        points=endpoints[near];center=points.mean(axis=0)
        _,_,vv=np.linalg.svd(points-center,full_matrices=False);v=vv[0]
        if v@axis<0:v=-v
        normal=np.array([v[1],-v[0]])
        if normal@direction<0:normal=-normal
        axis,direction=v,normal;baseline=float(center@direction)
        near=abs(endpoints@direction-baseline)<6
    q2=lines.reshape(-1,2,2)@direction
    keep=np.min(abs(q2-baseline),axis=1)<12
    lines=lines[keep];length=length[keep]
    line_t=lines.reshape(-1,2,2).mean(axis=1)@axis
    # LSD returns both edges of every black stroke. Measure stroke centers from
    # a grayscale scan inside the tick band instead of mistaking ink width for spacing.
    ts=np.arange(math.floor(line_t.min())-20,math.ceil(line_t.max())+21,dtype=float)
    profiles=[]
    for side in (-1,1):
        values=[]
        for offset in (25,40,55):
            xy=ts[:,None]*axis+(baseline+side*offset)*direction
            sample=cv2.remap(gray,xy[:,0].astype(np.float32)[None,:],xy[:,1].astype(np.float32)[None,:],
                             cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=255)[0]
            values.append(sample)
        profile=np.median(values,axis=0).astype(np.float32)
        background=cv2.GaussianBlur(profile[None,:],(151,1),0)[0]
        contrast=background-profile
        threshold=max(12,float(np.percentile(contrast,95))*.35)
        mask=(contrast>threshold).astype(np.uint8)
        padded=np.r_[0,mask,0];starts=np.flatnonzero(np.diff(padded.astype(int))==1);ends=np.flatnonzero(np.diff(padded.astype(int))==-1)
        centers=[]
        for a,b in zip(starts,ends):
            if 2<=b-a<=25:centers.append(float(np.average(ts[a:b],weights=contrast[a:b])))
        if len(centers)>20:
            diffs=np.diff(centers);regular=diffs[(diffs>15)&(diffs<150)]
            if len(regular):
                spacing=float(np.median(regular));quality=float(np.mean(abs(regular/spacing-np.rint(regular/spacing))<.15))
                profiles.append((quality*len(centers),np.array(centers),side))
    if not profiles:raise ValueError('No reliable ruler tick-band profile')
    _,positions,profile_side=max(profiles,key=lambda v:v[0])
    # Measure the ink run along the actual printed tick direction. A long tick's
    # midpoint need not project onto its baseline position under perspective.
    run=np.arange(-300,301,dtype=float);samples=[]
    centers=positions[:,None]*axis+(baseline+profile_side*40)*direction
    perpendicular=np.array([-tick_direction[1],tick_direction[0]])
    for shift in (-13,0,13):
        xy=centers[:,None,:]+run[None,:,None]*tick_direction+shift*perpendicular
        samples.append(cv2.remap(gray,xy[:,:,0].astype(np.float32),xy[:,:,1].astype(np.float32),cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,borderValue=255).astype(float))
    ink=(samples[0]+samples[2])/2-samples[1];tick_lengths=[]
    for signal in ink:
        threshold=max(12,float(np.percentile(signal,90))*.3)
        mask=cv2.morphologyEx((signal>threshold).astype(np.uint8)[None,:],cv2.MORPH_CLOSE,np.ones((1,7),np.uint8))[0]
        mid=300
        if not mask[mid]:mid=int(np.argmax(signal[290:311]))+290
        left=right=mid
        while left>0 and mask[left-1]:left-=1
        while right<600 and mask[right+1]:right+=1
        tick_lengths.append(float(right-left+1))
    tick_lengths=np.array(tick_lengths)
    gaps=np.diff(positions);valid=gaps[(gaps>8)&(gaps<180)]
    if len(valid)<30:raise ValueError('Too few well-spaced ruler ticks')
    h,edges=np.histogram(valid,bins=np.arange(8,182,1));p=(edges[h.argmax()]+edges[h.argmax()+1])/2
    p=float(np.median(valid[abs(valid-p)<.18*p]))
    steps=np.maximum(1,np.rint(np.diff(positions)/p).astype(int))
    k=np.r_[0,np.cumsum(steps)]
    p,origin=np.polyfit(k,positions,1);residual=positions-(origin+p*k)
    # Keep physical scale variation visible rather than fitting a false periodic alias.
    inliers=np.ones(len(k),dtype=bool)
    # Long printed centimeter ticks must show the ten-subdivision pattern.
    phase_scores=[float(np.median(tick_lengths[inliers&(k%10==phase)])) if np.sum(inliers&(k%10==phase))>=3 else 0 for phase in range(10)]
    phase=int(np.argmax(phase_scores))
    major=np.flatnonzero(inliers&(k%10==phase))
    if inliers.sum()<40 or len(major)<4 or inliers.mean()<.8:raise ValueError('Insufficient minor/centimeter grid support')
    points=[]
    for t,mm,L,err,ok in zip(positions,k,tick_lengths,residual,inliers):
        xy=axis*t+direction*baseline
        points.append(dict(x=float(xy[0]),y=float(xy[1]),relative_mm=int(mm),absolute_ruler_mm=None,
                           tick_length_px=float(L),residual_px=float(err),used=bool(ok)))
    quarters=[]
    for part in np.array_split(np.flatnonzero(inliers),4):
        if len(part)>5:quarters.append(float(np.polyfit(k[part],positions[part],1)[0]))
    return dict(px_per_mm=float(p),axis=axis.tolist(),tick_direction=direction.tolist(),baseline=float(baseline),
                tick_count=int(inliers.sum()),candidate_count=len(points),span_mm=int(np.ptp(k[inliers])),
                residual_rms_px=float(np.sqrt(np.mean(residual[inliers]**2))),
                residual_p95_px=float(np.percentile(abs(residual[inliers]),95)),
                major_tick_phase=phase,phase_length_scores=phase_scores,segment_scales_px_per_mm=quarters,
                center_xy=np.mean([[v['x'],v['y']] for v in points if v['used']],axis=0).tolist(),
                points=points,unit_basis='Metric ruler cm labels and ten minor intervals per centimeter visually checked; distances are relative to the first detected lattice tick, not guessed absolute labels')


def metric_matrix(parameters,frame_size):
    """Image pixels -> arbitrary-origin metric tabletop coordinates (mm)."""
    w,h=frame_size
    ax,ay,shear,gx,gy=parameters
    affine=np.array([[math.exp(ax),shear,0],[0,math.exp(ay),0],[gx/(w/2),gy/(h/2),1]],float)
    center=np.array([[1,0,-w/2],[0,1,-h/2],[0,0,1]],float)
    return affine@center


def transform_points(points,matrix):
    points=np.asarray(points,float);homogeneous=np.c_[points,np.ones(len(points))]@np.asarray(matrix).T
    return homogeneous[:,:2]/homogeneous[:,2,None]


def fit_metric_calibration(rulers,frame_size=(3024,4032),mode='projective'):
    from scipy.optimize import least_squares
    initial=np.array([math.log(1/np.median([r['px_per_mm'] for r in rulers]))]*2+[0,0,0],float)
    active={'global':[0],'anisotropic':[0,1],'affine':[0,1,2],'projective':[0,1,2,3,4]}[mode]
    def expand(values):
        p=initial.copy();p[active]=values
        if mode=='global':p[1]=p[0]
        return p
    def residual(values):
        matrix=metric_matrix(expand(values),frame_size);answer=[]
        for ruler in rulers:
            points=[p for p in ruler['points'] if p['used']];xy=np.array([[p['x'],p['y']] for p in points]);k=np.array([p['relative_mm'] for p in points])
            world=transform_points(xy,matrix);middle=len(k)//2
            answer.extend((np.linalg.norm(world-world[middle],axis=1)-abs(k-k[middle]))/math.sqrt(len(k)))
        return np.array(answer)
    low=np.array([math.log(1/55),math.log(1/55),-.02,-.25,-.25]);high=np.array([math.log(1/20),math.log(1/20),.02,.25,.25])
    result=least_squares(residual,initial[active],bounds=(low[active],high[active]),loss='soft_l1',f_scale=.05,max_nfev=200)
    matrix=metric_matrix(expand(result.x),frame_size)
    return dict(mode=mode,matrix_pixel_to_mm=matrix.tolist(),parameters=expand(result.x).tolist(),fit_success=bool(result.success),
                ruler_errors=[ruler_interval_errors(r,matrix) for r in rulers])


def ruler_interval_errors(ruler,matrix):
    points=[p for p in ruler['points'] if p['used']];xy=np.array([[p['x'],p['y']] for p in points]);k=np.array([p['relative_mm'] for p in points])
    world=transform_points(xy,matrix);errors=[]
    for i in range(len(k)):
        candidates=np.flatnonzero(k==k[i]+10)
        if len(candidates):errors.append(float(np.linalg.norm(world[candidates[0]]-world[i])-10))
    a=np.array(errors)
    return dict(image_id=ruler['image_id'],interval_mm=10,intervals=len(errors),
        mean_error_mm=float(a.mean()),rms_error_mm=float(np.sqrt(np.mean(a*a))),
        p95_absolute_error_mm=float(np.percentile(abs(a),95)),max_absolute_error_mm=float(max(abs(a))))


def radial_points(points,parameters,frame_size=(3024,4032)):
    w,h=frame_size;xy=np.asarray(points,float)-np.array([w/2,h/2]);radius=np.sum(xy*xy,axis=1)/(max(w,h)/2)**2
    scale=math.exp(parameters[0])*(1+parameters[1]*radius+(parameters[2] if len(parameters)>2 else 0)*radius**2)
    return xy*scale[:,None]


def fit_radial_calibration(rulers,frame_size=(3024,4032),order=1):
    from scipy.optimize import least_squares
    initial=np.array([math.log(1/np.median([r['px_per_mm'] for r in rulers]))]+[0]*order)
    def residual(p):
        result=[]
        for ruler in rulers:
            points=[v for v in ruler['points'] if v['used']];xy=np.array([[v['x'],v['y']] for v in points]);k=np.array([v['relative_mm'] for v in points]);world=radial_points(xy,p,frame_size);mid=len(k)//2
            result.extend((np.linalg.norm(world-world[mid],axis=1)-abs(k-k[mid]))/math.sqrt(len(k)))
        return result
    result=least_squares(residual,initial,bounds=([math.log(1/55)]+[-.2]*order,[math.log(1/20)]+[.2]*order),loss='soft_l1',f_scale=.05,max_nfev=200)
    return result.x.tolist()


def radial_interval_errors(ruler,parameters,frame_size=(3024,4032)):
    points=[p for p in ruler['points'] if p['used']];xy=np.array([[p['x'],p['y']] for p in points]);k=np.array([p['relative_mm'] for p in points]);world=radial_points(xy,parameters,frame_size);errors=[]
    for i in range(len(k)):
        candidates=np.flatnonzero(k==k[i]+10)
        if len(candidates):errors.append(float(np.linalg.norm(world[candidates[0]]-world[i])-10))
    a=np.array(errors)
    return dict(image_id=ruler['image_id'],interval_mm=10,intervals=len(a),mean_error_mm=float(a.mean()),
                rms_error_mm=float(np.sqrt(np.mean(a*a))),p95_absolute_error_mm=float(np.percentile(abs(a),95)),
                max_absolute_error_mm=float(max(abs(a))))
