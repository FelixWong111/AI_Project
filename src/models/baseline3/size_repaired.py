"""Full frozen YOLO + repaired geometry + foreground + work-order candidate.

Independent v2 entry point keeps the frozen v1 implementation reproducible.
"""
import copy
from pathlib import Path
import cv2
import numpy as np
from .inference import ROOT,_frame,_observations
from .size_candidate import SizeDevelopmentBackend
from .geometry_repair import measure_object
from src.work_orders import validate_counts


def assess(measurement,class_id,bands,nominal_spec=None):
    """Unreliable/borderline measurements cannot become positive ID0 evidence."""
    required=bands[str(class_id)];dims=measurement['dimensions'];quality=measurement['diagnostics']
    if measurement['status']!='MEASURED_CANDIDATE' or not quality.get('quality_valid_for_rejection'):
        return 'UNMEASURABLE',[],{}
    uncertainty=float(quality['measurement_uncertainty_mm'])
    outside=[];borderline=[];intervals={}
    for key,band in required.items():
        if key not in dims:return 'UNMEASURABLE',[key],{}
        value=dims[key]['mm_estimate'];interval=[value-uncertainty,value+uncertainty]
        if key=='across_flats' and quality.get('opposite_flats_mm'):
            # Different projected opposite-flat separations are evidence of
            # tilt/shadow ambiguity, not three exact physical AF measurements.
            flats=quality['opposite_flats_mm']
            interval=[min(flats)-uncertainty,max(flats)+uncertainty]
        if key in ('outer_diameter','inner_diameter'):
            error=max(uncertainty,float(quality.get(key.split('_')[0]+'_ellipse_rms_fraction',0))*value)
            interval=[value-error,value+error]
        intervals[key]=dict(local_repeatability_interval_mm=interval,
            absolute_physical_error_bound_mm=None,provisional=True)
        if value<band['min_mm'] or value>band['max_mm']:
            if interval[1]<band['min_mm'] or interval[0]>band['max_mm']:outside.append(key)
            else:borderline.append(key)
    conflicts=[]
    if nominal_spec is not None:
        for key in list(outside):
            nominal=nominal_spec['dimensions'].get(key)
            if nominal and abs(dims[key]['mm_estimate']-nominal['reference_mm'])<=nominal['provisional_allowance_mm']:
                outside.remove(key);conflicts.append(key)
    if outside:return 'OUTSIDE_DEVELOPMENT_BAND',outside,intervals
    # An empirical envelope of one specimen is not a manufacturing tolerance.
    # Contradictory existing nominal evidence prevents an ID0 assertion; it
    # does not enlarge the PASS range or claim teacher/caliper confirmation.
    if conflicts:return 'SPEC_REFERENCE_CONFLICT',conflicts,intervals
    if borderline:return 'BORDERLINE_MEASUREMENT',borderline,intervals
    return 'WITHIN_DEVELOPMENT_BAND',[],intervals


def foreground_conservation(provider,frame,raw,supports):
    """Resolve threshold fragments using measured shapes, retaining residuals.

    Original P06 candidate groups remain visible. Fragmented faces and bridges
    between distinct measured objects do not themselves imply extra objects.
    Every unexplained foreground component above the unchanged area limit still
    prevents PASS. No unknown region is invented as a catalogue/ID0 object.
    """
    original=provider.process(frame,raw)
    if 'illumination_coefficients' not in original:return original
    width,height=provider.width,provider.height;sx=width/frame.shape[1];sy=height/frame.shape[0]
    gray=cv2.cvtColor(cv2.resize(frame,(width,height),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY).astype(np.float32)
    illumination=np.einsum('ijk,k->ij',provider.basis,np.asarray(original['illumination_coefficients']))
    corrected=np.clip(gray-illumination,0,255).astype(np.uint8)
    mask=(cv2.absdiff(corrected,provider.gray.astype(np.uint8))>=provider.config['parameters']['difference_threshold']).astype(np.uint8)
    kernel=np.ones((provider.config['parameters']['kernel_size'],)*2,np.uint8)
    mask=cv2.morphologyEx(cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel),cv2.MORPH_CLOSE,kernel)
    union=np.zeros_like(mask);support_records=[];bad=[];shape_masks=[]
    for obj in raw['objects']:
        shape=np.zeros_like(mask);contours=supports.get(obj['detection_id'],[])
        for contour in contours:
            points=np.asarray(contour,float)*[sx,sy]
            if len(points)>=3:cv2.fillPoly(shape,[cv2.convexHull(np.rint(points).astype(np.int32))],1)
        if not contours:
            bad.append(f"object {obj['detection_id']}: NO_TRUSTED_GEOMETRIC_SUPPORT")
        area=int(shape.sum());observed=int((shape&mask).sum())
        if area and observed/max(1,area)<.03:bad.append(f"object {obj['detection_id']}: NO_FOREGROUND_SUPPORT")
        # 3 reduced pixels = 12 original pixels, only anti-alias/shadow-border support.
        expanded=cv2.dilate(shape,np.ones((7,7),np.uint8));union|=expanded
        support_records.append(dict(detection_id=obj['detection_id'],shape_pixels=area,foreground_pixels=observed))
        shape_masks.append(shape)
    overlapping=[]
    for i,a in enumerate(shape_masks):
        for j,b in enumerate(shape_masks[i+1:],i+1):
            overlap=int((a&b).sum())/max(1,min(int(a.sum()),int(b.sum())))
            if overlap>.35:
                bad.append(f'objects {i}/{j}: OVERLAPPING_MEASURED_SHAPES');overlapping.extend((i,j))
    residual=mask&(1-union);count,labels,stats,_=cv2.connectedComponentsWithStats(residual,8)
    unexplained=[]
    for i in range(1,count):
        x,y,w,h,area=map(int,stats[i])
        if area>=provider.config['parameters']['minimum_area']:
            unexplained.append(dict(xmin=x,ymin=y,xmax=x+w,ymax=y+h,area_pixels=area,reason='UNMATCHED_FOREGROUND'))
    if unexplained:bad.append('UNMATCHED_FOREGROUND')
    if not original.get('background_usable',False):bad.append('BACKGROUND_OR_SCENE_CHANGED')
    return dict(original,counts_complete=not bad,reason='; '.join(dict.fromkeys(bad)),
        original_component_association_complete=original['counts_complete'],
        resolution_method='Conserve foreground outside disjoint, image-measured part shapes; unchanged area/contrast limits',
        measured_shape_support=support_records,residual_regions=unexplained,
        ambiguous_detection_ids=sorted(set(overlapping)),
        unexplained_foreground_pixels=sum(r['area_pixels'] for r in unexplained),
        physical_unknown_validation=False)


class RepairedSizeBackend(SizeDevelopmentBackend):
    def __init__(self,path,**kwargs):
        super().__init__(path,**kwargs)
        self.files.extend([Path(__file__),ROOT/'src/models/baseline3/geometry_repair.py'])
        self.signature=self._signature()
        self.metadata.update(baseline3_mode='size_repaired_development',
            measurement_strategy='Stage2 exterior-supported screw cross-sections; target-owned nut/annulus contours; quality-gated size rejection',
            foreground='P06 threshold + illumination correction + measured-shape foreground conservation',
            size_bands='Unchanged frozen v1 numerical ranges; no widening or new fit',
            model_version=self.metadata['model_version']+'-geometry-repair-v2')

    def infer(self,frame,expected_counts):
        _frame(frame);expected=validate_counts(expected_counts)
        if self._signature()!=self.signature:raise RuntimeError('Repair files changed; reload model')
        raw=self.detector.infer(frame,expected)
        return self.process_observations(frame,raw,expected)

    def process_observations(self,frame,raw,expected_counts,*,measurement_cache=None):
        """Development-only cached replay; normal infer always measures fresh.

        A cache must come from the same original pixels, detector boxes and v2
        geometry. This entry is not a model benchmark or independent evaluation.
        """
        _frame(frame);expected=validate_counts(expected_counts);h,w=frame.shape[:2]
        compatible=[w,h]==self.calibration['frame_size']
        result=copy.deepcopy(raw);supports={}
        inspection=dict(mode='size_repaired_development',acceptance_confirmed=False,calibration_compatible=compatible,
            calibration=dict(calibration_id='ruler10-fixed-collection-size-dev-v1',validated=False,
                px_per_mm=self.calibration['px_per_mm'],instructor_approved=False),
            specifications=copy.deepcopy(self.specifications),measurement_calls=0,measurement_complete=0,
            geometry_execution='FRESH' if measurement_cache is None else 'REPLAY_DEVELOPMENT_MEASUREMENTS',
            failure_reasons=[],review_reasons=[],caliper_mae=None)
        if not compatible:inspection['review_reasons'].append('CALIBRATION_FRAME_MISMATCH / Camo calibration required')
        for obj in result['objects']:
            cid=obj['class_id'];obj['detector_class_id']=cid
            if cid==0:
                obj['size_evidence']=dict(conclusion='DETECTOR_NON_CATALOGUE',provisional=True)
                continue
            if not compatible:
                obj['size_evidence']=dict(conclusion='UNMEASURABLE',reason='CALIBRATION_FRAME_MISMATCH',provisional=True)
                continue
            x1,y1,x2,y2=[obj[k] for k in ('xmin','ymin','xmax','ymax')];pad=max(24,round(max(x2-x1,y2-y1)*.35))
            l,t=max(0,int(x1)-pad),max(0,int(y1)-pad);r,b=min(w,int(np.ceil(x2))+pad),min(h,int(np.ceil(y2))+pad)
            peers=[]
            for other in raw['objects']:
                if other['detection_id']==obj['detection_id']:continue
                a,c,d,e=[other[k] for k in ('xmin','ymin','xmax','ymax')]
                if min(r,d)>max(l,a) and min(b,e)>max(t,c):peers.append([a-l,c-t,d-l,e-t])
            family='1' if cid in (1,2,3) else str(cid)
            params={k:v for k,v in self.selection['selected'][family]['parameters'].items() if k!='targeted'}
            measurement=(measure_object(frame[t:b,l:r],cid,self.calibration['px_per_mm'],params,
                target_box=[x1-l,y1-t,x2-l,y2-t],excluded_boxes=peers)
                if measurement_cache is None else copy.deepcopy(measurement_cache[obj['detection_id']]))
            inspection['measurement_calls']+=1
            inspection['measurement_complete']+=int(measurement['status']=='MEASURED_CANDIDATE')
            if measurement['status']=='MEASURED_CANDIDATE':
                supports[obj['detection_id']]=[(np.asarray(c)+[l,t]).tolist() for c in measurement['contours_xy']]
            measurement.pop('contours_xy',None)
            for dim in measurement['dimensions'].values():dim['line_xy_original']=(np.asarray(dim['line_xy'])+[l,t]).tolist()
            nominal=next((s for s in self.specifications if s['class_id']==cid),None)
            conclusion,details,intervals=assess(measurement,cid,self.selection['bands'],nominal)
            obj['size_evidence']=dict(measurement=measurement,conclusion=conclusion,details=details,provisional=True,
                reason=measurement['reason'],development_acceptance_bands=self.selection['bands'][str(cid)],
                measurement_policy_reason='EXTERIOR_IMAGE_SUPPORT / DEVELOPMENT_ONLY / NOT_INSTRUCTOR_APPROVED',
                quality=dict(measurement['diagnostics'],independent_error_bound_mm=None,
                             ruler_scale_envelope=self.calibration['relative_conservative_scale_envelope']),dimension_evidence=intervals)
            if conclusion=='OUTSIDE_DEVELOPMENT_BAND':
                obj['class_id']=0;obj['class_assignment']='provisional_size_rule_non_catalogue'
                inspection['failure_reasons'].append(f"object {obj['detection_id']}: IMAGE_SUPPORTED_SIZE_OUTSIDE_UNCHANGED_BAND {details}")
            elif conclusion in ('UNMEASURABLE','BORDERLINE_MEASUREMENT','SPEC_REFERENCE_CONFLICT'):
                inspection['review_reasons'].append(f"object {obj['detection_id']}: {measurement['reason'] or conclusion} {details}")
        if compatible:
            inspection['foreground']=foreground_conservation(self.foreground,frame,raw,supports)
            if not inspection['foreground']['counts_complete']:inspection['review_reasons'].append(inspection['foreground']['reason'])
            ambiguous=inspection['foreground'].get('ambiguous_detection_ids',[])
            for obj in result['objects']:
                if obj['detection_id'] in ambiguous and obj.get('class_assignment')=='provisional_size_rule_non_catalogue':
                    obj['class_id']=obj['detector_class_id'];obj.pop('class_assignment')
                    obj['size_evidence'].update(conclusion='AMBIGUOUS_INSTANCE_SUPPORT',
                        reason='Measured shape overlaps another instance; cannot attribute size evidence reliably')
                    prefix=f"object {obj['detection_id']}: IMAGE_SUPPORTED_SIZE_OUTSIDE_UNCHANGED_BAND"
                    inspection['failure_reasons']=[r for r in inspection['failure_reasons'] if not r.startswith(prefix)]
                    inspection['review_reasons'].append(f"object {obj['detection_id']}: AMBIGUOUS_INSTANCE_SUPPORT")
        else:inspection['foreground']=dict(counts_complete=False,reason='UNSUPPORTED_CAPTURE_GEOMETRY')
        result['inspection']=inspection
        return self.rejudge(result,expected)

    def rejudge(self,result,expected_counts):
        result=super().rejudge(result,expected_counts)
        result['baseline3_mode']='size_repaired_development'
        return result


def create_backend(*,model_id,config):
    if model_id!='baseline3' or config.get('baseline3_mode')!='size_repaired_development':
        raise ValueError('Explicit Baseline3 repaired-size mode required; no count-only fallback')
    path=Path(config['size_candidate_config']);path=path if path.is_absolute() else ROOT/path
    return RepairedSizeBackend(path)
