"""Bounded image-only refinement of the original pooled screw profile."""
import cv2
import numpy as np
from src.p06_geometry import dimension,MeasurementFailure,require


def refine(samples,positions,mid,body,center,axis,normal,start,width,scale,sref,gray):
    def xy(u,v):
        return (center+axis*(start+u)+normal*v)*scale/sref

    dims,failures,details={},{},{}
    rough=float(np.quantile(width[body],.9))
    head_width=float(max(width[:len(width)//2]))
    middle=float(positions[mid])
    shaft=rough
    try:
        signal=np.quantile(samples,.25,axis=0).astype(np.float32)
        signal=cv2.GaussianBlur(signal[None,:],(0,0),1.5).ravel()
        gradient=np.gradient(signal)
        seeds=[]
        for values,offset,side in ((-gradient[:mid],0,0),(gradient[mid:],mid,-1)):
            candidates=np.flatnonzero(values>max(2,.4*float(values.max())))
            require(len(candidates)>0,'SHAFT_EDGE_GRADIENT_WEAK')
            groups=np.split(candidates,np.flatnonzero(np.diff(candidates)>1)+1)
            ordered=groups if side==0 else groups[::-1]
            # Prefer a narrow transition over a broad outer shadow ramp.
            group=ordered[0]
            for candidate in ordered:
                peak=int(candidate[np.argmax(values[candidate])])
                a=b=peak
                while a>0 and values[a-1]>.5*values[peak]:a-=1
                while b<len(values)-1 and values[b+1]>.5*values[peak]:b+=1
                if b-a<=8:
                    group=candidate
                    break
            seeds.append(offset+int(group[np.argmax(values[group])]))
        left,right=seeds
        shaft=float(positions[right]-positions[left])
        require(.25*rough<shaft<1.15*rough,'SHAFT_EDGE_AMBIGUOUS_REFLECTION_OR_SHADOW')
        derivatives=np.gradient(cv2.GaussianBlur(samples,(7,1),1.5),axis=1)
        candidates=[]
        radius=max(3,round(.06*shaft))
        for i,g in enumerate(derivatives):
            edges=[]
            for seed,sign in ((left,-1),(right,1)):
                a,b=max(1,seed-radius),min(len(g)-1,seed+radius+1)
                k=a+int(np.argmax(sign*g[a:b]))
                if sign*g[k]<max(2,.3*abs(gradient[seed])):break
                edges.append(k)
            if len(edges)==2:
                a,b=edges
                candidates.append((i,a,b,float(positions[b]-positions[a])))
        require(len(candidates)>=max(3,.15*len(samples)),'NO_SUPPORTED_CREST_SECTIONS')
        # Choose an observed cross-section near the upper central envelope.
        # No class-specific diameter is consulted.
        crest=float(positions[right]-positions[left])
        i,a,b,_=min(candidates,key=lambda c:(abs(c[3]-crest),abs(c[0]-len(samples)/2)))
        middle=float((positions[a]+positions[b])/2)
        shaft=float(positions[b]-positions[a])
        dims['thread_major_diameter']=dimension(xy(body[i],positions[a]),xy(body[i],positions[b]),scale)
        details.update(supported_sections=len(candidates),diameter_method='observed section nearest narrow pooled-profile edges')
    except MeasurementFailure as error:
        failures['thread_major_diameter']=str(error)
    try:
        require(head_width>1.5*shaft and head_width>1.2*rough,'HEAD_SHAFT_SEPARATION_FAILED')
        level=shaft+.5*(head_width-shaft)
        narrow=width<level
        run=max(5,round(.2*shaft))
        shoulder=None
        for i in range(2,int(.55*len(width))-run):
            if width[:i].max()>level and narrow[i:i+run].all():
                shoulder=i
                break
        require(shoulder is not None,'UNDERHEAD_PLANE_AMBIGUOUS')
        # Side tracks can expose a seating-plane edge hidden by a wide
        # silhouette/shadow. Use an observed edge when several tracks agree;
        # otherwise retain the width-transition candidate and report its origin.
        search=np.arange(max(2,int(shoulder-shaft)),min(int(.55*len(width)),int(shoulder+shaft)))
        planes=[]
        for sign in (-1,1):
            track=xy(search[:,None],middle+sign*.65*shaft).astype(np.float32)
            if len(search)<7:continue
            signal=cv2.remap(gray.astype(np.float32),track[:,0][None,:],track[:,1][None,:],cv2.INTER_LINEAR)
            gradient=np.gradient(cv2.GaussianBlur(signal,(7,1),1.3).ravel())
            k=int(gradient.argmax())
            if 2<=k<len(search)-2 and gradient[k]>3:
                planes.append(float(search[k]))
        if len(planes)==2 and abs(planes[0]-planes[1])<.25*shaft:
            shoulder=float(np.mean(planes))
            details['shoulder_source']='two agreeing exposed edge tracks'
        else:
            details['shoulder_source']='width transition; exposed tracks inconclusive'
        tip=len(width)-1
        require(tip-shoulder>1.4*shaft,'SHAFT_TOO_SHORT_OR_WRONG_POSE')
        dims['underhead_length']=dimension(xy(shoulder,middle),xy(tip,middle),scale)
        details.update(underhead_method='first sustained head-to-shaft width transition',
                       head_shaft_boundary_xy=[xy(shoulder,middle-shaft).tolist(),xy(shoulder,middle+shaft).tolist()])
    except MeasurementFailure as error:
        failures['underhead_length']=str(error)
    details['dimension_failures']=failures
    return dims,details
