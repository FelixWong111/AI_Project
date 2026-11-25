"""Fixed-background candidate extraction and conservative detection association."""
import copy
import math
import hashlib

import cv2
import numpy as np

from .models.baseline3.inference import _frame, _observations

DEFAULTS = dict(difference_threshold=30, minimum_area=40, kernel_size=3,
                association_fraction=.25, duplicate_iou=.8, maximum_foreground_fraction=.45)


def validate_config(config):
    if not isinstance(config, dict) or set(config) != set(DEFAULTS):
        raise ValueError('Unknown/missing foreground parameters')
    for k in ('difference_threshold','minimum_area','kernel_size'):
        if type(config[k]) is not int or config[k] <= 0:
            raise ValueError('Expected positive integer '+k)
    if config['difference_threshold'] > 255 or config['kernel_size'] % 2 != 1:
        raise ValueError('Invalid difference threshold or kernel')
    for k in ('association_fraction','duplicate_iou','maximum_foreground_fraction'):
        if type(config[k]) not in (float,int) or not math.isfinite(config[k]) or not 0 < config[k] <= 1:
            raise ValueError('Invalid fraction '+k)
    return copy.deepcopy(config)


def iou(a,b):
    x1,y1=max(a['xmin'],b['xmin']),max(a['ymin'],b['ymin'])
    x2,y2=min(a['xmax'],b['xmax']),min(a['ymax'],b['ymax'])
    intersection=max(0,x2-x1)*max(0,y2-y1)
    area=lambda r:(r['xmax']-r['xmin'])*(r['ymax']-r['ymin'])
    return intersection/(area(a)+area(b)-intersection)


class ForegroundAssociator:
    def __init__(self, background_bgr, *, reference_id, capture_geometry_id, roi=None, config=None):
        _frame(background_bgr)
        if any(not isinstance(v,str) or not v.strip() for v in (reference_id,capture_geometry_id)):
            raise ValueError('Traceable background and capture geometry IDs required')
        self.background=background_bgr.copy()
        self.background_sha256=hashlib.sha256(self.background.tobytes()).hexdigest()
        self.reference_id=reference_id
        self.geometry_id=capture_geometry_id
        h,w=background_bgr.shape[:2]
        roi=(0,0,w,h) if roi is None else roi
        if (not isinstance(roi,(list,tuple)) or len(roi)!=4 or any(type(v) is not int for v in roi)
                or not 0<=roi[0]<roi[2]<=w or not 0<=roi[1]<roi[3]<=h):
            raise ValueError('ROI must be an integer half-open original-frame box')
        self.roi=tuple(roi)
        self.config=validate_config(DEFAULTS if config is None else config)

    def process(self, frame_bgr, observations, *, capture_geometry_id):
        _frame(frame_bgr)
        if frame_bgr.shape!=self.background.shape or capture_geometry_id!=self.geometry_id:
            raise ValueError('Background resolution or capture geometry mismatch')
        h,w=frame_bgr.shape[:2]
        objects=observations.get('objects')
        if (not isinstance(objects,list) or observations.get('status')!='REVIEW'
                or observations.get('frame_size')!=dict(width=w,height=h)):
            raise ValueError('Expected original-frame P06 detector observations')
        rows=[]
        for index,obj in enumerate(objects):
            if type(obj.get('detection_id')) is not int or obj['detection_id']!=index or type(obj.get('class_id')) is not int:
                raise ValueError('Invalid detection identity')
            rows.append([obj[k] for k in ('xmin','ymin','xmax','ymax','score','class_id')])
        checked=_observations(rows,w,h)
        counts=observations.get('counts')
        if (not isinstance(counts,dict) or set(counts)!=set('012345')
                or any(type(v) is not int or v<0 for v in counts.values())
                or type(observations.get('non_catalogue_count')) is not int
                or checked['counts']!=counts or checked['non_catalogue_count']!=observations['non_catalogue_count']):
            raise ValueError('Objects and counts disagree')
        x1,y1,x2,y2=self.roi
        diff=cv2.absdiff(frame_bgr[y1:y2,x1:x2],self.background[y1:y2,x1:x2]).max(axis=2)
        mask=(diff>=self.config['difference_threshold']).astype(np.uint8)
        kernel=np.ones((self.config['kernel_size'],)*2,np.uint8)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
        fraction=float(mask.mean())
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
        regions=[]
        for label in range(1,count):
            x,y,rw,rh,area=map(int,stats[label])
            if area<self.config['minimum_area']:continue
            regions.append(dict(region_id=len(regions),component_label=label,
                                xmin=x+x1,ymin=y+y1,xmax=x+x1+rw,ymax=y+y1+rh,
                                area_pixels=area,touches_roi_edge=x==0 or y==0 or x+rw==x2-x1 or y+rh==y2-y1))
        bad_background=fraction>self.config['maximum_foreground_fraction']
        # Bipartite connectivity retains many-to-one and one-to-many ambiguity.
        nodes=[('region',r['region_id']) for r in regions]+[('detection',i) for i in range(len(objects))]
        graph={node:set() for node in nodes}
        associations=[]
        if not bad_background:
            for obj in objects:
                left=max(x1,math.floor(obj['xmin']));top=max(y1,math.floor(obj['ymin']))
                right=min(x2,math.ceil(obj['xmax']));bottom=min(y2,math.ceil(obj['ymax']))
                if left>=right or top>=bottom:continue
                piece=labels[top-y1:bottom-y1,left-x1:right-x1]
                for region in regions:
                    overlap=int((piece==region['component_label']).sum())
                    fraction_overlap=overlap/min(region['area_pixels'],piece.size)
                    if fraction_overlap>=self.config['association_fraction']:
                        a,b=('region',region['region_id']),('detection',obj['detection_id'])
                        graph[a].add(b);graph[b].add(a)
                        associations.append(dict(region_id=a[1],detection_id=b[1],overlap_pixels=overlap,
                                                 association_fraction=fraction_overlap))
        groups=[];visited=set();proposed=dict.fromkeys('012345',0);suppressed=[]
        for node in nodes:
            if node in visited:continue
            pending=[node];component=set()
            while pending:
                current=pending.pop()
                if current in component:continue
                component.add(current);pending.extend(graph[current]-component)
            visited.update(component)
            rids=sorted(i for kind,i in component if kind=='region')
            dids=sorted(i for kind,i in component if kind=='detection')
            group=dict(candidate_id=len(groups),region_ids=rids,detection_ids=dids,class_id=None,
                       retained_detection_id=None,suppressed_detection_ids=[],status='REVIEW')
            if bad_background:
                group['reason']='BACKGROUND_OR_SCENE_CHANGED'
            elif not dids:
                group['reason']='UNMATCHED_FOREGROUND'
            elif not rids:
                group['reason']='DETECTION_WITHOUT_FOREGROUND_SUPPORT'
            elif len(rids)>1:
                group['reason']='ONE_DETECTION_OR_CHAIN_LINKS_MULTIPLE_REGIONS'
            elif regions[rids[0]]['touches_roi_edge']:
                group['reason']='FOREGROUND_TOUCHES_ROI_EDGE'
            else:
                same_class=len({objects[i]['class_id'] for i in dids})==1
                clique=all(iou(objects[a],objects[b])>=self.config['duplicate_iou']
                           for pos,a in enumerate(dids) for b in dids[pos+1:])
                if not clique:
                    group['reason']='MULTIPLE_OBJECTS_OR_MERGED_FOREGROUND'
                elif not same_class:
                    group['reason']='CROSS_CLASS_DUPLICATE_AMBIGUOUS'
                else:
                    best=max(dids,key=lambda i:(objects[i]['score'],-i))
                    removed=[i for i in dids if i!=best]
                    group.update(class_id=objects[best]['class_id'],retained_detection_id=best,
                                 suppressed_detection_ids=removed,reason='ASSOCIATED' if not removed else 'SAME_CLASS_DUPLICATES_GROUPED')
                    proposed[str(group['class_id'])]+=1
                    suppressed.extend(removed)
            groups.append(group)
        return dict(status='REVIEW',background_reference_id=self.reference_id,capture_geometry_id=self.geometry_id,
                    background_pixel_sha256=self.background_sha256,frame_size=dict(width=w,height=h),
                    roi=list(self.roi),parameters=copy.deepcopy(self.config),foreground_fraction=fraction,
                    background_usable=not bad_background,regions=regions,associations=associations,candidates=groups,
                    raw_counts=copy.deepcopy(counts),proposed_counts=proposed,
                    counts_complete=not bad_background and all(g['class_id'] is not None for g in groups),
                    suppressed_detection_ids=sorted(suppressed),detector_observations=copy.deepcopy(observations),
                    scope='foreground candidates and proposed counts; final classification and decision pending')
