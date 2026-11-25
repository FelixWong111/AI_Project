"""Load traceable validation scopes and enable only matching dimension evidence."""
import hashlib
import json
from pathlib import Path

from .p06_geometry import DEFAULT_PARAMETERS
from .p06_size_evidence import DIMENSIONS,number

GEOMETRY=Path(__file__).with_name('p06_geometry.py')


class MeasurementPolicy:
    def __init__(self,records=()):
        self.records=[]
        for record in records:
            required={'evidence_id','status','geometry_sha256','parameters','calibration_id',
                      'capture_geometry_id','frame_width','frame_height','class_ids',
                      'dimensions','metric_ranges','family_confirmed','px_per_mm'}
            if not isinstance(record,dict) or set(record)!=required:
                raise ValueError('Invalid measurement validation record')
            if (record['status']!='VALIDATED_MEASUREMENT_SCOPE'
                    or record['geometry_sha256']!=hashlib.sha256(GEOMETRY.read_bytes()).hexdigest()
                    or record['parameters']!=DEFAULT_PARAMETERS):
                raise ValueError('Validation record method/version mismatch')
            for key in ('evidence_id','calibration_id','capture_geometry_id'):
                if not isinstance(record[key],str) or not record[key].strip():raise ValueError('Missing evidence identity')
            if any(type(record[k]) is not int or record[k]<=0 for k in ('frame_width','frame_height')):
                raise ValueError('Invalid validated resolution')
            if (not isinstance(record['px_per_mm'],list) or len(record['px_per_mm'])!=2
                    or not all(number(v,True) for v in record['px_per_mm'])):
                raise ValueError('Invalid validated scale')
            ids=record['class_ids']
            if not isinstance(ids,list) or not ids or any(type(i) is not int or i not in DIMENSIONS for i in ids) or len(set(ids))!=len(ids):
                raise ValueError('Invalid validation classes')
            if any(set(DIMENSIONS[i])!=set(DIMENSIONS[ids[0]]) for i in ids):
                raise ValueError('Validation scope mixes geometry families')
            if (not isinstance(record['dimensions'],dict) or not record['dimensions']
                    or set(record['dimensions'])-set(DIMENSIONS[ids[0]])
                    or any(not number(v) for v in record['dimensions'].values())):
                raise ValueError('Invalid validated error bounds')
            ranges=record['metric_ranges']
            if not isinstance(ranges,dict) or not ranges:
                raise ValueError('Automatic metric gates required for validated scope')
            for key,value in ranges.items():
                if (not isinstance(key,str) or not isinstance(value,list) or len(value)!=2
                        or not all(number(v) for v in value) or value[0]>value[1]):
                    raise ValueError('Invalid metric gate')
            if type(record['family_confirmed']) is not bool:raise ValueError('Invalid family validation')
            self.records.append(json.loads(json.dumps(record)))

    def assess(self,measurement,calibration,capture_geometry_id,frame_size):
        if not calibration or calibration['validated'] is not True:
            return {},False,'CALIBRATION_NOT_VALIDATED'
        if measurement.get('status') not in ('MEASURED_CANDIDATE','PARTIAL_CANDIDATE'):
            return {},False,'MEASUREMENT_UNAVAILABLE'
        matching=[]
        for record in self.records:
            if (measurement['class_id'] not in record['class_ids']
                    or record['calibration_id']!=calibration['calibration_id']
                    or record['capture_geometry_id']!=capture_geometry_id
                    or record['px_per_mm']!=list(calibration['px_per_mm'])
                    or (record['frame_width'],record['frame_height'])!=frame_size):
                continue
            diagnostics=measurement.get('diagnostics',{})
            if not all(number(diagnostics.get(k)) and lo<=diagnostics[k]<=hi
                       for k,(lo,hi) in record['metric_ranges'].items()):
                continue
            matching.append(record)
        if len(matching)!=1:
            return {},False,'NO_MATCHING_VALIDATION_SCOPE' if not matching else 'AMBIGUOUS_VALIDATION_SCOPES'
        record=matching[0]
        quality={k:dict(verified=True,max_error_mm=v,evidence_id=record['evidence_id'])
                 for k,v in record['dimensions'].items() if k in measurement['dimensions']}
        return quality,record['family_confirmed'],'VALIDATED_SCOPE_MATCHED'


def load_policy(path,root):
    path=Path(path)
    config=json.loads(path.read_text(encoding='utf-8-sig'))
    if set(config)!={'schema_version','records'} or config['schema_version']!=1 or not isinstance(config['records'],list):
        raise ValueError('Invalid measurement policy registry')
    records=[]
    for ref in config['records']:
        if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:raise ValueError('Invalid validation record reference')
        file=Path(ref['path']);file=file if file.is_absolute() else Path(root)/file
        raw=file.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=ref['sha256']:raise ValueError('Validation evidence SHA256 mismatch')
        records.append(json.loads(raw.decode('utf-8-sig')))
    return MeasurementPolicy(records)
