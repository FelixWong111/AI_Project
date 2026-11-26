"""Stage 2 geometric repair: exterior-supported edges, not nominal-size guessing.

The frozen P06 implementation is imported unchanged. This module fixes target
ownership and screw exterior-edge selection; it never consumes labels/orders.
"""
from __future__ import annotations
import math
import cv2
import numpy as np
from src.p06_geometry import (DEFAULT_PARAMETERS,MeasurementFailure,require,dimension,
                              washer_geometry,nut_geometry)


def target_contours(image,box,parameters,excluded_boxes=()):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);h,w=gray.shape
    x1,y1,x2,y2=np.asarray(box,float)
    l,t,r,b=max(0,int(x1)),max(0,int(y1)),min(w,int(np.ceil(x2))),min(h,int(np.ceil(y2)))
    require(l<r and t<b,'INVALID_TARGET_BOX')
    border=np.r_[gray[0],gray[-1],gray[:,0],gray[:,-1]]
    background=float(np.median(border))
    clean=gray.copy();owner=np.ones(gray.shape,np.uint8)
    for peer in excluded_boxes:
        a,c,d,e=map(float,peer)
        intersection=max(0,min(x2,d)-max(x1,a))*max(0,min(y2,e)-max(y1,c))
        aa,cc=max(0,int(a)-3),max(0,int(c)-3);dd,ee=min(w,int(np.ceil(d))+3),min(h,int(np.ceil(e))+3)
        if aa<dd and cc<ee:owner[cc:ee,aa:dd]=0
    # Axis-aligned boxes can overlap even when oblique parts do not. Never
    # erase original target pixels merely because they also fall in a peer box.
    # Ambiguous physical support is checked by contours/shape overlap later.
    owner[t:b,l:r]=1
    clean[owner==0]=round(background)
    blurred=cv2.GaussianBlur(clean,(3,3),0)
    otsu,_=cv2.threshold(blurred[t:b,l:r],0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    threshold=min(float(otsu),background-parameters['background_offset'])
    require(threshold>0,'INSUFFICIENT_BACKGROUND_CONTRAST')
    binary=(blurred<=threshold).astype(np.uint8)*255
    binary[owner==0]=0
    binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    edges=cv2.Canny(blurred,parameters['canny_low'],parameters['canny_high'])
    edges[owner==0]=0
    # Join short missing boundary segments; never reconstruct a bright metal face.
    edges=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    alternatives=[]
    for method,mask in (('DARK_SILHOUETTE',binary),('CLOSED_OBSERVED_EDGES',edges)):
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        scored=[]
        for contour in contours:
            area=cv2.contourArea(contour)
            if area<parameters['minimum_area']:continue
            xx,yy,ww,hh=cv2.boundingRect(contour)
            overlap=max(0,min(xx+ww,r)-max(xx,l))*max(0,min(yy+hh,b)-max(yy,t))
            if overlap<=0:continue
            filled=np.zeros(gray.shape,np.uint8);cv2.drawContours(filled,[contour],-1,1,-1)
            in_target=int(filled[t:b,l:r].sum());fraction=in_target/max(1,int(filled.sum()))
            if fraction<.5:continue
            scored.append((in_target,contour,area,fraction))
        if not scored:continue
        scored.sort(key=lambda q:-q[0]);score,contour,area,fraction=scored[0]
        if len(scored)>1 and scored[1][0]>.35*score:
            # Separate peers already masked; two large residual objects are real ambiguity.
            continue
        xx,yy,ww,hh=cv2.boundingRect(contour)
        if xx<=1 or yy<=1 or xx+ww>=w-1 or yy+hh>=h-1:continue
        alternatives.append(dict(contour=contour,method=method,area=area,target_fraction=fraction))
    require(alternatives,'TARGET_OUTLINE_UNRESOLVED_OR_CLIPPED')
    return gray,clean,alternatives,dict(background_intensity=background,segmentation_threshold=threshold,
                                     isolated_peer_boxes=len(excluded_boxes))


def screw_exterior(contour,gray,scale):
    sref=float(np.mean(scale));h,w=gray.shape
    # Convex support repairs a bright stripe splitting the threshold silhouette
    # for axis estimation only. Measurement endpoints still come from real pixels.
    hull=cv2.convexHull(contour);x,y,rw,rh=cv2.boundingRect(hull)
    filled=np.zeros((rh,rw),np.uint8);cv2.drawContours(filled,[hull-np.array([[[x,y]]])],-1,1,-1)
    yy,xx=np.nonzero(filled);points=np.c_[xx+x,yy+y].astype(float)*sref/scale
    center=points.mean(axis=0);_,vectors=np.linalg.eigh(np.cov((points-center).T));axis=vectors[:,-1]
    boundary=contour[:,0].astype(float)*sref/scale
    def profile(direction):
        normal=np.array([-direction[1],direction[0]]);uv=np.stack(((boundary-center)@direction,(boundary-center)@normal),axis=1)
        start=float(uv[:,0].min());bins=np.rint(uv[:,0]-start).astype(int);n=bins.max()+1
        lo=np.full(n,np.inf);hi=np.full(n,-np.inf);np.minimum.at(lo,bins,uv[:,1]);np.maximum.at(hi,bins,uv[:,1])
        valid=np.isfinite(lo)&np.isfinite(hi);ii=np.arange(n)
        require(valid.sum()>.5*n,'DISCONTINUOUS_SCREW_BOUNDARY')
        lo=np.interp(ii,ii[valid],lo[valid]);hi=np.interp(ii,ii[valid],hi[valid])
        # Smooth over thread pitch noise, not the head/body transition.
        width=cv2.medianBlur((hi-lo).astype(np.float32).reshape(1,-1),5).ravel()
        return normal,start,lo,hi,width
    for _ in range(3):
        normal,start,lo,hi,width=profile(axis);n=len(width);require(n>=30,'SCREW_TOO_SMALL')
        if max(width[:n//3])<max(width[-n//3:]):axis=-axis;normal,start,lo,hi,width=profile(axis)
        body=np.arange(int(.45*n),int(.85*n));slope=np.polyfit(body,(lo[body]+hi[body])/2,1)[0]
        require(abs(slope)<.8,'SCREW_AXIS_AMBIGUOUS');axis+=slope*normal;axis/=np.linalg.norm(axis)
    normal,start,lo,hi,width=profile(axis);n=len(width);body=np.arange(int(.4*n),int(.85*n))
    rough=float(np.quantile(width[body],.9));head_width=float(max(width[:int(.5*n)]))
    require(rough>6 and head_width>1.2*rough,'HEAD_SHAFT_SEPARATION_FAILED')
    head=np.flatnonzero(width[:int(.55*n)]>rough+.3*(head_width-rough));require(len(head)>=3,'HEAD_SHAFT_SEPARATION_FAILED')
    shoulder=float(head[-1]+1);tip=float(n-1);span=tip-shoulder
    sections=np.arange(math.ceil(shoulder+max(.22*span,.5*rough)),math.floor(shoulder+.80*span)+1)
    require(len(sections)>=8,'SHAFT_SECTION_NOT_CLEAR_OF_HEAD_OR_TIP')
    # Extend beyond both rough boundaries: the old seed grid stopped inside metal.
    transverse=np.arange(math.floor(lo[body].min()-max(16,.4*rough)),math.ceil(hi[body].max()+max(16,.4*rough))+1,dtype=float)
    grid=(center+axis[None,None,:]*(start+sections[:,None,None])+normal[None,None,:]*transverse[None,:,None])*scale/sref
    require((grid[:,:,0]>=0).all() and (grid[:,:,0]<w-1).all() and (grid[:,:,1]>=0).all() and (grid[:,:,1]<h-1).all(),'SHAFT_PROFILE_OUTSIDE_CROP')
    samples=cv2.remap(gray.astype(np.float32),grid[:,:,0].astype(np.float32),grid[:,:,1].astype(np.float32),cv2.INTER_LINEAR)
    smoothed=cv2.GaussianBlur(samples,(7,1),1.5);gradient=np.gradient(smoothed,axis=1)
    middle=float(np.median((lo[sections]+hi[sections])/2));mid=int(np.argmin(abs(transverse-middle)))
    # Vectorized neighbourhood statistics avoid a Python scan/median for each
    # candidate pixel. Smoothing is already applied; means quantify sustained
    # background support across the entire far band, not a single bright pixel.
    chosen=[];exterior_rejections=0;edge_indices=[];edge_valid=[]
    band=max(6,round(.12*rough));far_size=max(8,round(.22*rough));inner_size=max(7,round(.10*rough))
    columns=np.arange(smoothed.shape[1]);rows=np.arange(len(sections))
    for side,polarity in ((-1,-1),(1,1)):
        bg=np.median(smoothed[:,:8] if side<0 else smoothed[:,-8:],axis=1)[:,None]
        signed=polarity*gradient
        peaks=np.zeros(signed.shape,bool)
        peaks[:,4:-4]=(signed[:,4:-4]>=2)&(signed[:,4:-4]>=signed[:,3:-5])&(signed[:,4:-4]>signed[:,5:-3])
        peaks &= columns[None,:]<mid if side<0 else columns[None,:]>mid
        far_anchor=(far_size-1,0) if side<0 else (0,0)
        inner_anchor=(0,0) if side<0 else (inner_size-1,0)
        outside_all=cv2.boxFilter(smoothed,-1,(far_size,1),anchor=far_anchor,normalize=True)
        deviation_all=cv2.boxFilter(np.abs(smoothed-bg),-1,(far_size,1),anchor=far_anchor,normalize=True)
        inner_all=cv2.boxFilter(smoothed,-1,(inner_size,1),anchor=inner_anchor,normalize=True)
        far_at=columns+side*(band+3);inner_at=columns-side*3
        supported=(far_at>=0)&(far_at<len(columns))&(inner_at>=0)&(inner_at<len(columns))
        supported &= (columns>=band+far_size+3) if side<0 else (columns<len(columns)-band-far_size-3)
        outside=outside_all[:,np.clip(far_at,0,len(columns)-1)]
        deviation=deviation_all[:,np.clip(far_at,0,len(columns)-1)]
        inside=inner_all[:,np.clip(inner_at,0,len(columns)-1)]
        contrast=np.maximum(1,bg-inside)
        step=smoothed[:,np.clip(columns+side*3,0,len(columns)-1)]-smoothed[:,np.clip(columns-side*3,0,len(columns)-1)]
        valid=peaks&supported[None,:]&(np.abs(outside-bg)<=np.maximum(15,.22*contrast))&(outside-inside>=10)&(deviation<=np.maximum(20,.28*contrast))&(step>=.55*(outside-inside))
        exterior_rejections+=int((peaks&~valid).sum())
        geometric=lo[sections] if side<0 else hi[sections]
        cost=np.where(valid,np.abs(transverse[None,:]-geometric[:,None]),np.inf)
        indices=cost.argmin(axis=1);edge_indices.append(indices);edge_valid.append(np.isfinite(cost[rows,indices]))
    for i in np.flatnonzero(edge_valid[0]&edge_valid[1]):
        left,right=edge_indices[0][i],edge_indices[1][i];diameter=transverse[right]-transverse[left]
        if .55*rough<diameter<1.5*rough:chosen.append((int(i),int(left),int(right),float(diameter)))
    require(len(chosen)>=max(8,.2*len(sections)),'SHAFT_EXTERIOR_EDGES_UNSUPPORTED')
    widths=np.array([q[3] for q in chosen]);major=float(np.quantile(widths,.9))
    # A repeated crest envelope is robust to thread roots; reject support from
    # only one local glint or a heavily changing silhouette.
    bands=[sum(int(sections[q[0]]-sections[0])//max(1,math.ceil(len(sections)/3))==i for q in chosen) for i in range(3)]
    require(min(bands)>=2,'SHAFT_EXTERIOR_TRACK_DISCONTINUOUS')
    i,left,right,diameter=min(chosen,key=lambda q:(abs(q[3]-major),abs(q[0]-len(sections)/2)))
    u=float(sections[i]);a,b=transverse[[left,right]];middle=(a+b)/2
    def xy(axial,normal_position):return (center+axis*(start+axial)+normal*normal_position)*scale/sref
    dims=dict(underhead_length=dimension(xy(shoulder,middle),xy(tip,middle),scale),
              thread_major_diameter=dimension(xy(u,a),xy(u,b),scale))
    spread=float(np.quantile(widths,.9)-np.quantile(widths,.1));require(spread/major<.30,'SHAFT_WIDTH_SUPPORT_INCONSISTENT')
    # Empirical repeatability + pixel quantization; not an absolute/caliper error bound.
    error_px=max(2.,float(np.median(abs(widths-np.median(widths))))*1.4826)
    return dims,dict(silhouette_method='target-owned observed boundary; hull used for axis only',
        diameter_method='90th-percentile envelope of repeated background-connected exterior edges',
        underhead_method='observed silhouette shoulder transition',tip_method='observed silhouette end',
        rough_contour_width_px=rough,diameter_sections=len(chosen),eligible_sections=len(sections),
        exterior_rejected_internal_edges=exterior_rejections,diameter_p10_px=float(np.quantile(widths,.1)),
        diameter_p90_px=major,diameter_repeatability_px=error_px,
        shaft_line_fraction=(u-shoulder)/span,axis_xy=axis.tolist(),
        quality_valid_for_rejection=True,measurement_uncertainty_mm=error_px/sref)


def measure_object(image,class_id,scale,parameters=None,*,target_box=None,excluded_boxes=()):
    if class_id not in (1,2,3,4,5):raise ValueError('Repair geometry requires catalogue class 1..5')
    scale=np.asarray(scale,float);params=dict(DEFAULT_PARAMETERS,**(parameters or {}))
    if scale.shape!=(2,) or not np.isfinite(scale).all() or min(scale)<=0:raise ValueError('Invalid scale')
    h,w=image.shape[:2];box=target_box if target_box is not None else [1,1,w-1,h-1]
    result=dict(status='UNMEASURABLE',reason='',class_id=class_id,dimensions={},dimension_status={},diagnostics={},
                contours_xy=[],scale_px_per_mm=scale.tolist(),scope='repaired Stage2 visible geometry; provisional mm, not caliper truth')
    try:
        gray,clean,contours,diag=target_contours(image,box,params,excluded_boxes)
        successes=[];failures=[]
        for candidate in contours:
            try:
                contour=candidate['contour']
                if class_id in (1,2,3):dims,details=screw_exterior(contour,gray,scale);support=[contour]
                elif class_id==4:dims,details=nut_geometry(contour,scale,params);support=[contour]
                else:
                    # P06 annulus geometry on the isolated target, with neighbours
                    # removed before hierarchy detection (formerly used raw gray).
                    x1,y1,x2,y2=map(float,box);pad=max(16,round(.18*max(x2-x1,y2-y1)))
                    l,t=max(0,int(x1)-pad),max(0,int(y1)-pad);r,b=min(w,int(np.ceil(x2))+pad),min(h,int(np.ceil(y2))+pad)
                    dims,details,support=washer_geometry(clean[t:b,l:r],scale,params,False)
                    for dim in dims.values():dim['line_xy']=(np.asarray(dim['line_xy'])+[l,t]).tolist()
                    support=[c+np.array([[[l,t]]]) for c in support]
                successes.append((dims,details,support,candidate))
                if class_id==5:break  # The hierarchy fit does not depend on outer seed choice.
            except MeasurementFailure as error:failures.append(str(error))
        require(successes,';'.join(dict.fromkeys(failures)))
        # Prefer the larger observed boundary for screws, where an internal
        # reflectance loop is not the full part. Shape fits otherwise prefer
        # the stronger closed edge already validated by the original algorithm.
        chosen=max(successes,key=lambda q:q[3]['area']) if class_id in (1,2,3) else successes[-1]
        dims,details,support,candidate=chosen
        for dim in dims.values():
            line=np.asarray(dim['line_xy']);require(np.isfinite(line).all() and (line>=0).all() and (line[:,0]<w).all() and (line[:,1]<h).all(),'MEASUREMENT_LINE_OUTSIDE_CROP')
        details.setdefault('quality_valid_for_rejection',True)
        details.setdefault('measurement_uncertainty_mm',2/float(np.mean(scale)))
        result.update(status='MEASURED_CANDIDATE',dimensions=dims,contours_xy=[c[:,0].tolist() for c in support],
            diagnostics=dict(diag,**details,selected_contour_method=candidate['method'],candidate_outlines=len(contours)))
    except MeasurementFailure as error:
        result.update(reason=str(error));result['diagnostics'].update(quality_valid_for_rejection=False)
    keys=('underhead_length','thread_major_diameter') if class_id in (1,2,3) else ('across_flats',) if class_id==4 else ('outer_diameter','inner_diameter')
    result['dimension_status']={k:dict(status='CANDIDATE' if k in result['dimensions'] else 'UNMEASURABLE',reason=result['reason']) for k in keys}
    return result
