"""Read-only source audit and deterministic candidate data preparation (no models)."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
from scipy.fft import dctn
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from src.data.development import ROOT, COUNT_FIELDS, digest
from src.data.p06_labelme import CLASSES, to_yolo, from_yolo
from src.work_orders import reference_counts

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.bmp'}
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
SPEC = {'1': 'M4×25', '2': 'M5×10', '3': 'M4×10', '4': 'M4', '5': 'M5'}


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8'))


def save_csv(path, rows, fields=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n', extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def audit_inventory(original):
    hashes,names=defaultdict(list),defaultdict(list)
    for row in original:
        hashes[row['sha256']].append(row['path']);names[Path(row['path']).name.casefold()].append(row['path'])
    return dict(source_files=len(original),byte_duplicate_groups=[v for v in hashes.values() if len(v)>1],
                basename_collisions=[v for v in names.values() if len(v)>1],
                orphan_json=[r['path'] for r in original if r['path'].lower().endswith('.json') and not (ROOT/r['path']).with_suffix('.jpg').exists()])


def register_heif():
    try:
        import pillow_heif
    except ImportError:
        sys.path.insert(0, str(ROOT/'.tools'))
        import pillow_heif
    pillow_heif.register_heif_opener()


def workbook_rows(path):
    """Read literal/cached XLSX cells only. Never rewrite the supplied workbook."""
    with zipfile.ZipFile(path) as z:
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            shared = [''.join(t.itertext()) for t in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        names = sorted(n for n in z.namelist() if re.fullmatch(r'xl/worksheets/sheet\d+\.xml', n))
        result = []
        for name in names:
            doc = ET.fromstring(z.read(name)); cells = {}
            for c in doc.findall('.//s:sheetData/s:row/s:c', NS):
                v, inline = c.find('s:v', NS), c.find('s:is', NS)
                text = v.text if v is not None else ''.join(inline.itertext()) if inline is not None else ''
                cells[c.get('r')] = shared[int(text)] if c.get('t') == 's' else text
            # Expand actual merged ranges only; blank descriptions are never guessed/forward-filled.
            for merge in doc.findall('.//s:mergeCell', NS):
                a, b = merge.get('ref').split(':')
                ca, ra = re.fullmatch(r'([A-Z]+)(\d+)', a).groups()
                cb, rb = re.fullmatch(r'([A-Z]+)(\d+)', b).groups()
                if ca == cb:
                    for n in range(int(ra), int(rb)+1): cells[f'{ca}{n}'] = cells.get(a, '')
            for row in doc.findall('.//s:sheetData/s:row', NS)[1:]:
                n = row.get('r'); d = {c: cells.get(c+n, '') for c in 'ABCDEFGHIJKL'}
                if d['J'].isdigit(): result.append(dict(d, sheet=name, row=n))
        return result


def capture_records(source):
    result, evidence = {}, []
    for p in sorted(source.rglob('*.xlsx')):
        for row in workbook_rows(p):
            prefix, width = ('困难工况', 3) if '困难工况' in p.parts else (p.stem, 2)
            stem = prefix + row['J'].zfill(width)
            candidate = p.parent/(stem+'.jpg')
            if not candidate.exists(): raise ValueError(f'Capture record has no exact filename match: {p}:{row["row"]}')
            key = relative(candidate)
            if key in result: raise ValueError(f'Duplicate capture record: {key}')
            record = dict(source=relative(p), sheet=row['sheet'], row=row['row'], description=row['B'],
                          counts=[int(row[c]) for c in 'CDEFGH'], label=row['I'], image_number=row['J'])
            result[key] = record; evidence.append(dict(image_path=key, **record))
    return result, evidence


def pixel_info(path):
    row = dict(path=relative(path), size_bytes=path.stat().st_size, sha256=digest(path))
    try:
        with Image.open(path) as src:
            orientation = src.getexif().get(274, 1)
            image = ImageOps.exif_transpose(src).convert('RGB'); image.load()
            row.update(width=image.width, height=image.height, exif_orientation=orientation,
                       pixel_sha256=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest())
            gray = image.convert('L')
            a = np.asarray(gray.resize((32,32), Image.Resampling.LANCZOS), dtype=float)
            values = dctn(a, type=2, norm='ortho')[:8,:8].reshape(-1)
            row['phash'] = ''.join('1' if x > np.median(values[1:]) else '0' for x in values)
            small = np.asarray(gray.resize((9,8), Image.Resampling.LANCZOS))
            row['dhash'] = ''.join('1' if x else '0' for x in (small[:,1:]>small[:,:-1]).reshape(-1))
            row['thumbnail'] = np.asarray(gray.resize((64,64), Image.Resampling.LANCZOS)).tolist()
            row['exif'] = {str(k): str(src.getexif().get(k, '')) for k in (271,272,306,33434,33437,37386)}
    except Exception as exc:
        row['decode_error'] = f'{type(exc).__name__}: {exc}'
    return row


def parse_annotation(path, info):
    """Continuous original pixel xyxy; polygons become min/max visible-vertex AABBs."""
    issues, objects = [], []
    if not path.is_file(): return [], ['MISSING_JSON'], {}
    try:
        doc = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(doc, dict) or not isinstance(doc.get('shapes'), list):
            raise ValueError('Expected a LabelMe object with shapes list')
    except (ValueError, OSError) as exc: return [], [f'INVALID_JSON: {exc}'], {}
    if doc.get('imagePath') != path.with_suffix('.jpg').name: issues.append('IMAGE_PATH_MISMATCH')
    if (doc.get('imageWidth'), doc.get('imageHeight')) != (info.get('width'), info.get('height')):
        issues.append('DIMENSION_MISMATCH')
    if info.get('exif_orientation', 1) != 1: issues.append('AMBIGUOUS_EXIF_ORIENTATION')
    if doc.get('flags'): issues.append('UNAGREED_DOCUMENT_FLAGS')
    boxes = []
    for i, shape in enumerate(doc['shapes']):
        try:
            label = shape['label']; kind = shape.get('shape_type')
            if label not in CLASSES: raise ValueError(f'UNKNOWN_CLASS {label!r}')
            if kind not in ('rectangle','polygon'): raise ValueError(f'UNSUPPORTED_SHAPE {kind!r}')
            points = shape['points']
            if not isinstance(points, list) or len(points) < (3 if kind == 'polygon' else 2): raise ValueError('INVALID_POINTS')
            if kind == 'rectangle' and len(points) != 2: raise ValueError('RECTANGLE_POINT_COUNT')
            if any(not isinstance(p, list) or len(p)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in p) for p in points):
                raise ValueError('NONFINITE_OR_INVALID_POINTS')
            x,y = zip(*points); box = [min(x), min(y), max(x), max(y)]
            to_yolo(box, info['width'], info['height'])  # Existing P06 coordinate contract.
            if shape.get('flags') or shape.get('group_id') is not None: raise ValueError('UNAGREED_FLAGS_OR_GROUP')
            if box in boxes: raise ValueError('DUPLICATE_BOX')
            boxes.append(box)
            objects.append(dict(object_id=str(i+1), class_id=CLASSES.index(label), xyxy=box,
                                ignore=0, source_shape=kind, description=shape.get('description','')))
        except (ValueError, KeyError, TypeError) as exc:
            issues.append(f'SHAPE_{i+1}: {exc}')
    return objects, issues, doc


def count_label(counts):
    fixed = reference_counts()
    return 'PASS' if counts[0] == 0 and all(counts[c] == fixed[str(c)] for c in range(1,6)) else 'FAIL'


def scenario(counts):
    targets = [counts[c] for c in range(1,6)]; n0 = counts[0]
    missing, extra = sum(max(0,1-v) for v in targets), sum(max(0,v-1) for v in targets)
    if not sum(targets): return 'G'
    if not n0:
        if not missing and not extra: return 'A'
        if missing and not extra: return 'B'
        if extra and not missing: return 'C'
        return 'F'
    if missing and not extra and n0 == missing: return 'D'
    if not missing and not extra: return 'E'
    return 'F'


def annotate(source, infos, captures, resolutions=None):
    records = []
    for p in sorted(source.rglob('*.jpg')):
        key = relative(p); info = infos[key]; cap = captures.get(key)
        number = re.search(r'(\d+)$', p.stem)
        iid = p.name if '公共数据集' in p.parts else 'H'+number.group(1)+'.jpg'
        objects, issues, doc = parse_annotation(p.with_suffix('.json'), info)
        counts = Counter(o['class_id'] for o in objects if not o['ignore'])
        declared = p.parent.name if '公共数据集' in p.parts else ''
        condition = 'normal' if declared in 'ABCDEFG' and declared else ''
        if not condition and cap:
            condition = 'crowded' if '密集' in cap['description'] else 'edge' if '边缘' in cap['description'] else ''
        if not condition: issues.append('UNKNOWN_CONDITION')
        if 'decode_error' in info: issues.append('DECODE_ERROR')
        if cap:
            actual = [counts[c] for c in (1,2,3,4,5,0)]
            expected = cap['counts']
            # E's header explicitly says 2 denotes an added abnormal specimen in that slot.
            # Keep the literal counts as evidence; compare the declared normal targets separately.
            comparable = [min(v,1) for v in expected[:5]]+[expected[5]] if declared == 'E' else expected
            if actual != comparable: issues.append('CAPTURE_COUNTS_CONFLICT')
            if not objects and any(expected): issues.append('EMPTY_JSON_WITH_NONEMPTY_RECORD')
        elif not objects:
            issues.append('EMPTY_SCENE_UNCONFIRMED')
        calculated = scenario(counts)
        scenario_note=''
        if declared and calculated != declared:
            scenario_note=f'Declared {declared}; count-only pattern {calculated}. Counts cannot prove the identity of a replaced specimen.'
        # F includes a missing normal object plus an unrelated abnormal object, even when its
        # count vector alone also fits D. A-G is metadata, never a replacement for object truth.
        category=declared if declared=='F' else calculated
        record = dict(image_id=iid, source_image=key, source_annotation=relative(p.with_suffix('.json')),
                      scene_id=Path(iid).stem, scene_group_id=Path(iid).stem,
                      condition=condition, scenario_type=category if not issues else 'unknown',scenario_note=scenario_note,
                      declared_scenario=declared, physical_sample_id='', width=info.get('width',''), height=info.get('height',''),
                      annotation_status='REVIEW' if issues else 'COMPLETE', label='' if issues else count_label(counts),
                      annotation_issues=issues, annotation_notes='Human-created JSON; software format checks only; independent visual sign-off pending.',
                      counts={str(c): counts[c] for c in range(6)}, objects=objects, capture_record=cap,
                      non_catalogue_type='', non_catalogue_type_source='',
                      sha256=info['sha256'], pixel_sha256=info.get('pixel_sha256',''), split='quarantine', evaluation_stage='')
        # Only explicit subtype text is retained. A class-0 box itself carries no subtype/spec identity.
        if counts[0] and cap:
            text = cap['description']
            types = [spec for spec, pattern in [('M4x40_screw',r'M4[×x]40'),('M4_small_washer',r'M4小垫片'),('M6_cap_nut',r'M6盖型螺母')] if re.search(pattern,text)]
            # D/E workbook headers define abnormal replacement/addition slots. This is a
            # declared capture subtype, not a subtype inferred from the photographed pixels.
            slots={1:'M4x40_screw',4:'M6_cap_nut',5:'M4_small_washer'}
            if declared in ('D','E'):
                indices=[i for i,v in enumerate(cap['counts'][:5],1) if (v==0 if declared=='D' else v==2)]
                if all(i in slots for i in indices):types=sorted(set(types+[slots[i] for i in indices]))
            record['non_catalogue_type'] = ';'.join(types)
            if types: record['non_catalogue_type_source'] = 'capture_record_declared_not_visually_verified'
        resolution=(resolutions or {}).get(iid)
        if resolution:
            if resolution.get('decision')!='USE_ORIGINAL_JSON' or resolution.get('annotation_sha256')!=digest(p.with_suffix('.json')):
                raise ValueError(f'Human resolution does not bind to unchanged JSON: {iid}')
            resolved=set(resolution['resolved_issues'])
            record['annotation_issues']=[s for s in issues if s not in resolved]
            record['annotation_status']='REVIEW' if record['annotation_issues'] else 'COMPLETE'
            record['label']='' if record['annotation_issues'] else count_label(counts)
            record['scenario_type']=category if not record['annotation_issues'] else 'unknown'
            record['annotation_notes']+=' Count conflict resolved by user: original JSON authoritative, 2026-11-26; source spreadsheet retained.'
            record['human_resolution']=resolution
            if 'CAPTURE_COUNTS_CONFLICT' in resolved:
                record['non_catalogue_type']=''
                record['non_catalogue_type_source']='UNKNOWN: capture composition overridden by confirmed JSON'
        records.append(record)
    names = Counter(r['image_id'] for r in records)
    for r in records:
        if names[r['image_id']] > 1:
            r['image_id'] = hashlib.sha256(r['source_image'].encode()).hexdigest()[:10]+'_'+r['image_id']
            r['scene_id'] = r['scene_group_id'] = Path(r['image_id']).stem
    return records


class Groups:
    def __init__(self, ids): self.parent = {i:i for i in ids}
    def root(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]; i = self.parent[i]
        return i
    def join(self, a, b): self.parent[max(self.root(a),self.root(b))] = min(self.root(a),self.root(b))


def duplicates(records, infos):
    groups = Groups(r['image_id'] for r in records); exact, near = [], []
    for field in ('sha256','pixel_sha256'):
        seen = defaultdict(list)
        for r in records:
            if r[field]: seen[r[field]].append(r)
        for value, members in seen.items():
            if len(members)<2: continue
            exact.append(dict(method=field, digest=value, image_ids=[r['image_id'] for r in members]))
            for r in members[1:]: groups.join(members[0]['image_id'],r['image_id'])
    # Empty-table photographs do not add independently rearranged physical scenes.
    empty = [r for r in records if not r['objects'] and not r['annotation_issues']]
    for r in empty[1:]: groups.join(empty[0]['image_id'],r['image_id'])
    valid = [r for r in records if infos[r['source_image']].get('phash')]
    for i,a in enumerate(valid):
        ia = infos[a['source_image']]
        for b in valid[i+1:]:
            ib = infos[b['source_image']]
            distance = (int(ia['phash'],2)^int(ib['phash'],2)).bit_count()
            if distance > 8: continue
            rmse = float(np.sqrt(np.mean((np.asarray(ia['thumbnail'],float)-np.asarray(ib['thumbnail'],float))**2)))
            near.append(dict(image_a=a['image_id'],image_b=b['image_id'],phash_hamming=distance,
                             gray64_rmse=round(rmse,5),decision='CANDIDATE_ONLY_NOT_PROOF',
                             reason='Background/identical counts do not establish shared scene'))
    for r in records: r['scene_group_id'] = groups.root(r['image_id'])
    return dict(exact_groups=exact, near_candidates=near,
                empty_background_group=[r['image_id'] for r in empty],
                phash_method='32x32 grayscale DCT, upper 8x8, median of 63 AC coefficients; Hamming <=8',
                cropped_derivatives='pHash can find rescaling/recompression; arbitrary crops and moved objects require visual review')


def assign_splits(records, seed=42):
    """Group MILP: exact feasible condition/label quotas; secondary scenario/instance balance."""
    usable = [r for r in records if not r['annotation_issues']]
    groups = defaultdict(list)
    for r in usable: groups[r['scene_group_id']].append(r)
    members = [groups[k] for k in sorted(groups)]; buckets = ('train','validation','stage2_test','stage3_test')
    rng = np.random.default_rng(seed); n = len(members)*4
    equations, lower, upper = [], [], []
    def constraint(coeff, lo, hi): equations.append(coeff); lower.append(lo); upper.append(hi)
    for i, group in enumerate(members): constraint({4*i+j:1 for j in range(4)},1,1)
    quotas = {}
    for condition in ('normal','crowded','edge'):
        for label in ('PASS','FAIL'):
            total = sum(r['condition']==condition and r['label']==label for r in usable)
            wanted = [150,30,15,15] if condition=='normal' else [15,5,0,15]
            # Preserve held-out/validation targets where possible; put small receipt deficits in train.
            wanted[0] += total-sum(wanted)
            if min(wanted)<0: raise ValueError(f'Not enough valid {condition}/{label}; review quotas explicitly')
            for j,quota in enumerate(wanted):
                coeff = {4*i+j:sum(r['condition']==condition and r['label']==label for r in group) for i,group in enumerate(members)}
                constraint(coeff,quota,quota); quotas[f'{condition}/{label}/{buckets[j]}'] = quota
    # Broad scenario representation; no arbitrary subtype inference is needed to choose these quotas.
    for scenario_type in 'BCDEFG':
        for j in range(4):
            coeff = {4*i+j:sum(r['scenario_type']==scenario_type for r in g) for i,g in enumerate(members)}
            if any(coeff.values()): constraint(coeff,1,np.inf)
    matrix = lil_matrix((len(equations),n))
    for row,coeff in enumerate(equations):
        for col,value in coeff.items(): matrix[row,col] = value
    bounds_upper = np.ones(n)
    for i,g in enumerate(members):
        if any(r['condition']!='normal' for r in g): bounds_upper[4*i+2]=0
    # Soft proportional balance of A-G, six instance counts and declared abnormal subtypes.
    soft=[]
    subtype_names=('M4x40_screw','M4_small_washer','M6_cap_nut')
    for condition in ('normal','crowded','edge'):
        subset=[r for r in usable if r['condition']==condition]
        features=[(f'scenario_{s}',lambda r,s=s:int(r['scenario_type']==s)) for s in 'ABCDEFG']
        features += [(f'class_{c}',lambda r,c=c:r['counts'][str(c)]) for c in range(6)]
        features += [(t,lambda r,t=t:int(t in r['non_catalogue_type'].split(';'))) for t in subtype_names]
        for _, feature in features:
            total=sum(feature(r) for r in subset)
            if not total:continue
            for j,bucket in enumerate(buckets):
                target_images=sum(quotas[f'{condition}/{label}/{bucket}'] for label in ('PASS','FAIL'))
                if not target_images:continue
                coeff={4*i+j:sum(feature(r) for r in g if r['condition']==condition) for i,g in enumerate(members)}
                soft.append((coeff,total*target_images/len(subset),1/max(total,1)))
    full=lil_matrix((len(equations)+len(soft),n+2*len(soft)))
    full[:len(equations),:n]=matrix
    objective=list(rng.random(n)*1e-7)
    for k,(coeff,target,weight) in enumerate(soft):
        row=len(equations)+k
        for col,value in coeff.items():full[row,col]=value
        full[row,n+2*k]=1;full[row,n+2*k+1]=-1
        lower.append(target);upper.append(target);objective.extend((weight,weight))
    result = milp(np.array(objective), integrality=np.r_[np.ones(n),np.zeros(2*len(soft))],
                  bounds=Bounds(np.zeros(n+2*len(soft)),np.r_[bounds_upper,np.full(2*len(soft),np.inf)]),
                  constraints=LinearConstraint(full.tocsr(),lower,upper), options={'time_limit':50,'mip_rel_gap':0.005})
    if not result.success: raise ValueError(f'Group allocation infeasible; no fallback random split: {result.message}')
    for i,g in enumerate(members):
        selected=buckets[int(np.argmax(result.x[4*i:4*i+4]))]
        for r in g:
            r['split'] = 'test' if selected.endswith('_test') else selected
            r['evaluation_stage'] = 'stage2' if selected=='stage2_test' else 'stage3' if selected=='stage3_test' else 'development'
    return dict(seed=seed, method='scipy.optimize.milp binary scene-group assignment; hard condition/label quotas and scenario coverage; seeded objective', quotas=quotas, solver=result.message)


def old_inventory(source, output, infos):
    """Data provenance audit only; no prediction, fitting or threshold selection."""
    roots = [ROOT/n for n in ('开发','验证','测试') if (ROOT/n).exists()]
    roots += sorted(ROOT.glob('P03_*'))
    oldpaths = sorted(set(p for root in roots for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES))
    with ThreadPoolExecutor(max_workers=4) as pool: old = list(pool.map(pixel_info,oldpaths))
    save_csv(output/'audit/old_image_hashes.csv',[{k:v for k,v in r.items() if k not in ('thumbnail','exif')} for r in old])
    lookup = defaultdict(list)
    for item in old:
        for field in ('sha256','pixel_sha256'):
            if item.get(field): lookup[(field,item[field])].append(item['path'])
    matches = []
    for key,info in infos.items():
        for field in ('sha256','pixel_sha256'):
            if info.get(field) and (field,info[field]) in lookup:
                matches.append(dict(new_path=key,method=field,old_paths=lookup[(field,info[field])]))
    near=[]
    for key,info in infos.items():
        if not info.get('phash'):continue
        for previous in old:
            if previous.get('phash'):
                distance=(int(info['phash'],2)^int(previous['phash'],2)).bit_count()
                if distance<=6:near.append(dict(new_path=key,old_path=previous['path'],phash_hamming=distance,decision='CANDIDATE_ONLY_NOT_CONFIRMED'))
    return dict(images_scanned=len(old),decode_errors=[{k:v for k,v in r.items() if k in ('path','decode_error')} for r in old if 'decode_error' in r],
                roots=[relative(p) for p in roots], matches=matches,
                near_candidates=near,
                near_public_pairs=sum(Path(r['new_path']).suffix.lower()=='.jpg' for r in near),
                near_specialized_pairs=sum(Path(r['new_path']).suffix.lower()!='.jpg' for r in near),
                scope='All image files in explicit old raw/handoff roots; old Test read for hashing only, never evaluated',
                historical_usage='Current frozen B1/B2 development manifest and B3 training records retained; new collection provenance is user supplied; unrecorded external use UNKNOWN')


def copy_image(source, destination):
    """Independent byte copy: later edits to generated files cannot mutate the originals."""
    import shutil
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source,destination)


def statistics(records):
    result = {}
    subsets = {s:[r for r in records if r['split']==s] for s in ('train','validation','test','quarantine')}
    subsets.update({s:[r for r in records if r['evaluation_stage']==s] for s in ('stage2','stage3')})
    for name,rows in subsets.items():
        result[name] = dict(images=len(rows),scene_ids=len({r['scene_id'] for r in rows}),scene_groups=len({r['scene_group_id'] for r in rows}),
                            labels=dict(Counter(r['label'] or 'UNKNOWN' for r in rows)), conditions=dict(Counter(r['condition'] for r in rows)),
                            scenarios=dict(Counter(r['scenario_type'] for r in rows)),
                            class_instances={str(c):sum(r['counts'][str(c)] for r in rows) for c in range(6)},
                            declared_non_catalogue_types=dict(Counter(r['non_catalogue_type'] or 'UNKNOWN' for r in rows if r['counts']['0'])),
                            declared_subtype_image_presence={t:sum(t in r['non_catalogue_type'].split(';') for r in rows) for t in ('M4x40_screw','M4_small_washer','M6_cap_nut')},
                            independent_scene_note='scene IDs follow collector confirmation; scene groups conservatively collapse repeated empty backgrounds; not independent physical specimens')
    return result


def previews(records, out):
    groups = {f'class_{c}':[r for r in records if r['counts'][str(c)]] for c in range(6)}
    groups.update({k:[r for r in records if r['condition']==k] for k in ('normal','crowded','edge')})
    groups['empty'] = [r for r in records if not r['objects']]
    groups['conflicts'] = [r for r in records if r['annotation_issues']]
    preview_dir=out/'previews'
    active={Path(r['image_id']).stem for r in groups['conflicts']}
    if preview_dir.exists():
        for stale in preview_dir.glob('conflicts_*.jpg'):
            if stale.stem.removeprefix('conflicts_') not in active:stale.unlink()
    # Test is never used for model diagnostics. This is a pre-sealing annotation inspection only.
    selected = []
    for name,rows in groups.items():
        for r in rows[:8] if name=='conflicts' else rows[:1]: selected.append((name,r))
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    font = ImageFont.truetype(str(font_path),18) if font_path.exists() else ImageFont.load_default()
    for name,r in selected:
        with Image.open(ROOT/r['source_image']) as image:
            image=ImageOps.exif_transpose(image).convert('RGB'); image.thumbnail((600,700))
        sx,sy=image.width/r['width'],image.height/r['height']; draw=ImageDraw.Draw(image)
        for o in r['objects']:
            box=[v*(sx if i%2==0 else sy) for i,v in enumerate(o['xyxy'])]
            color='red' if o['class_id']==0 else '#00c060'; draw.rectangle(box,outline=color,width=3)
            text=f"{o['object_id']}: {o['class_id']} {CLASSES[o['class_id']]}"
            text_width=draw.textbbox((0,0),text,font=font)[2]
            draw.text((max(0,min(box[0],image.width-text_width-2)),max(0,box[1]-22)),text,font=font,fill=color,stroke_width=1,stroke_fill='white')
        canvas=Image.new('RGB',(600,760),'white');canvas.paste(image,(0,35));d=ImageDraw.Draw(canvas)
        d.text((5,5),f"{name} | {r['image_id']} | {r['annotation_status']}",font=font,fill='black')
        d.text((5,735),'Annotation preview - not model output',font=font,fill='black')
        target=out/'previews'/f'{name}_{Path(r["image_id"]).stem}.jpg';target.parent.mkdir(parents=True,exist_ok=True);canvas.save(target,quality=90)
    duplicate_path=out/'audit/duplicates.json'
    if duplicate_path.exists():
        pairs=json.loads(duplicate_path.read_text(encoding='utf-8'))['near_candidates']
        by_id={r['image_id']:r for r in records}
        pairs=sorted((p for p in pairs if by_id[p['image_a']]['objects'] or by_id[p['image_b']]['objects']),key=lambda p:(p['phash_hamming'],p['gray64_rmse']))[:10]
        contact=Image.new('RGB',(1600,1100),'white');draw=ImageDraw.Draw(contact)
        for n,pair in enumerate(pairs):
            x=(n%5)*320;y=(n//5)*550
            for k,key in enumerate(('image_a','image_b')):
                with Image.open(ROOT/by_id[pair[key]]['source_image']) as im:
                    im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((316,250));contact.paste(im,(x,y+20+k*265))
                draw.text((x+4,y+k*265),pair[key]+f" h={pair['phash_hamming']}",fill='black')
        contact.save(out/'previews/near_contact.jpg',quality=92)


def auxiliary(source, infos, out):
    dimensions, rulers = [], []
    for p in sorted(source.rglob('*')):
        if p.suffix.lower() not in ('.heic','.heif'): continue
        info=infos[relative(p)]; is_ruler='尺子标定10' in p.parts
        iid=('R_' if is_ruler else 'D_')+p.stem+'.png'
        path=out/'calibration'/('ruler' if is_ruler else 'dimensions')/iid
        # Lossless RGB decode for existing image readers; retain original HEIC and EXIF in source.
        if 'decode_error' not in info:
            path.parent.mkdir(parents=True,exist_ok=True)
            with Image.open(p) as image: ImageOps.exif_transpose(image).convert('RGB').save(path)
        if is_ruler:
            rulers.append(dict(image_id=iid,image_path=relative(path),source_path=relative(p),width=info.get('width',''),height=info.get('height',''),
                               ruler_position='',scale_reference='',px_per_mm='',camera_height_mm='',camera_parameters=json.dumps(info.get('exif',{}),ensure_ascii=False),
                               calibration_status='PENDING_MANUAL_MEASUREMENT',notes=info.get('decode_error','No ruler ticks or physical distances inferred')))
        else:
            cid=int(re.search(r'_ID(\d)_',p.parent.name).group(1)); spec=p.parent.name.split('_')[0]
            dimensions.append(dict(image_id=iid,image_path=relative(path),source_path=relative(p),physical_sample_id='',class_id=cid,
                                   nominal_specification=spec,actual_caliper_measurements_mm='',catalogue_compatible='UNVERIFIED',
                                   nominal_catalogue_membership='false' if cid==0 else 'true',
                                   capture_conditions='',usage='UNASSIGNED_PENDING_PHYSICAL_SAMPLE_AND_HISTORY',width=info.get('width',''),height=info.get('height',''),
                                   notes='Directory-declared specification, not caliper truth. Repeated views do not establish new specimens; M6 cap nut differs in type and size.'))
    save_csv(out/'manifests/dimension_manifest.csv',dimensions)
    save_csv(out/'calibration/ruler_manifest.csv',rulers)
    return dict(dimensions=len(dimensions),rulers=len(rulers),dimension_specs=dict(Counter(r['nominal_specification'] for r in dimensions)),
                decode_failures=sum('decode_error' in i for i in infos.values()),physical_sample_independence='UNKNOWN',calibration_validated=False)


def export(records, out):
    image_rows, object_rows = [], []
    for r in records:
        split = 'val' if r['split']=='validation' else r['split']
        image_path=out/'yolo/images'/split/r['image_id'];copy_image(ROOT/r['source_image'],image_path)
        r['image_path']=relative(image_path);r['annotation_path']=r['source_annotation']
        labels=[]
        for obj in r['objects']:
            box=obj['xyxy']; norm=to_yolo(box,r['width'],r['height'])
            labels.append(str(obj['class_id'])+' '+' '.join(f'{v:.12f}' for v in norm))
            if max(abs(a-b) for a,b in zip(box,from_yolo([float(f'{v:.12f}') for v in norm],r['width'],r['height'],decimal_places=12)))>1e-6:
                raise ValueError(f'YOLO round-trip error: {r["image_id"]}')
            object_rows.append(dict(image_id=r['image_id'],object_id=obj['object_id'],class_id=obj['class_id'],
                                    **dict(zip(('xmin','ymin','xmax','ymax'),box)),ignore=obj['ignore']))
        target=out/'yolo/labels'/split/(Path(r['image_id']).stem+'.txt');target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(('\n'.join(labels)+('\n' if labels else '')).encode('utf-8'))
        row={k:r[k] for k in ('image_id','scene_id','split','width','height','annotation_status','scene_group_id','condition','scenario_type',
                               'evaluation_stage','image_path','annotation_path','non_catalogue_type','physical_sample_id','annotation_notes')}
        row.update({field:r['counts'][str(c)] for c,field in enumerate(COUNT_FIELDS,1)})
        row.update(non_catalogue_count=r['counts']['0'],reference_label_all_one=r['label'],ground_truth_label=r['label'])
        image_rows.append(row)
    save_csv(out/'ground_truth_images.csv',image_rows)
    save_csv(out/'ground_truth_objects.csv',object_rows,['image_id','object_id','class_id','xmin','ymin','xmax','ymax','ignore'])
    development_ids={r['image_id'] for r in records if r['split'] in ('train','validation')}
    save_csv(out/'development/ground_truth_images.csv',[r for r in image_rows if r['image_id'] in development_ids])
    save_csv(out/'development/ground_truth_objects.csv',[r for r in object_rows if r['image_id'] in development_ids],['image_id','object_id','class_id','xmin','ymin','xmax','ymax','ignore'])
    save_json(out/'development/dataset_v2.json',dict(dataset_root=relative(out),status_file=relative(out/'dataset_status.json')))
    notes=[]
    for r in records:
        notes.extend(dict(image_id=r['image_id'],issue=issue,action='Human review required; not eligible for formal use') for issue in r['annotation_issues'])
        if r.get('human_resolution'):
            notes.append(dict(image_id=r['image_id'],issue='RESOLVED_CAPTURE_COUNTS_CONFLICT',action='User confirmed original JSON authoritative on 2026-11-26; both sources retained; see SHA-bound resolution record'))
        if r.get('scenario_note'):notes.append(dict(image_id=r['image_id'],issue='SCENARIO_METADATA_NOTE',action=r['scenario_note']))
        if r['counts']['0'] and not r['non_catalogue_type']:
            notes.append(dict(image_id=r['image_id'],issue='NON_CATALOGUE_SUBTYPE_UNRECORDED',action='Official ID0 retained; exact abnormal specification is unknown, not invented'))
    save_csv(out/'annotation_notes.csv',notes,['image_id','issue','action'])
    save_csv(out/'manifests/dataset_manifest.csv',[
        {k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else v
         for k,v in r.items() if k not in ('objects','capture_record')} for r in records])
    save_csv(out/'manifests/source_mapping.csv',[{k:r[k] for k in ('image_id','source_image','source_annotation','image_path','sha256','pixel_sha256')} for r in records])
    selections={s:[r for r in records if r['split']==s] for s in ('train','validation','test','quarantine')}
    selections.update({s+'_test':[r for r in records if r['evaluation_stage']==s] for s in ('stage2','stage3')})
    selections.update({'stage3_'+c:[r for r in records if r['evaluation_stage']=='stage3' and r['condition']==c] for c in ('normal','crowded','edge')})
    for key,rows in selections.items():
        if key=='quarantine' and not rows:continue
        alias='val' if key=='validation' else key
        path=out/'splits'/f'{alias}.txt';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((''.join(r['image_path']+'\n' for r in rows)).encode('utf-8'))
        save_json(out/'configs'/f'expected_counts_{alias}.json',[dict(image_id=r['image_id'],expected_counts=reference_counts()) for r in rows])
        save_csv(out/'splits'/f'{alias}.csv',[dict(image_id=r['image_id'],image_path=r['image_path'],label=r['label'],annotation_status=r['annotation_status'],scene_group_id=r['scene_group_id']) for r in rows],['image_id','image_path','label','annotation_status','scene_group_id'])
    save_json(out/'expected_counts.json',[dict(image_id=r['image_id'],expected_counts=reference_counts()) for r in records])
    save_json(out/'configs/class_mapping.json',dict(official_to_name=dict(enumerate(CLASSES)),official_to_yolo=dict(enumerate(range(6))),
               yolo_to_official=dict(enumerate(range(6))),source='config/baseline3.json and frozen six-class P06 checkpoint',nominal_specifications=SPEC,
               non_catalogue_specifications=['M4×40 screw','M4 small washer','M6 cap nut']))
    # No test key in training YAML. Relative paths resolve from the repository working directory.
    import yaml
    (out/'yolo/data.yaml').write_bytes(yaml.safe_dump(dict(path=relative(out/'yolo'),train='images/train',val='images/val',nc=6,names=dict(enumerate(CLASSES))),sort_keys=False).encode('utf-8'))
    for stage in ('stage2','stage3'):
        stage_ids={r['image_id'] for r in records if r['evaluation_stage']==stage}
        stage_dir=out/'evaluation'/stage
        save_csv(stage_dir/'ground_truth_images.csv',[r for r in image_rows if r['image_id'] in stage_ids])
        save_csv(stage_dir/'ground_truth_objects.csv',[r for r in object_rows if r['image_id'] in stage_ids],['image_id','object_id','class_id','xmin','ymin','xmax','ymax','ignore'])
        save_json(out/'configs'/f'{stage}_evaluation.json',dict(schema='inspection-evaluation/v2',dataset=relative(stage_dir),evaluation_stage=stage,
                  input_dir=relative(out/'yolo/images/test'),ground_truth_images=relative(stage_dir/'ground_truth_images.csv'),
                  ground_truth_objects=relative(stage_dir/'ground_truth_objects.csv'),counts_file=relative(out/'configs'/f'expected_counts_{stage}_test.json'),
                  split_list=relative(out/'splits'/f'{stage}_test.txt'),dataset_status=relative(out/'dataset_status.json'),formal_evaluation='NOT_RUN'))
    save_json(out/'configs/training.json',dict(schema='mechanical-dataset/v2',data_dir=relative(out/'development'),reference_label_field='reference_label_all_one',
               data_description='New independent mechanical-component arrangements supplied on 2026-11-26; original source 数据集/',
               scene_confirmation=dict(each_image_is_distinct_scene=True,source='2026-11-26 collector/user confirmation; auto IDs, conservative empty background grouping',scope='This new collection only'),
               seed=42,official_held_out_test='NOT_RUN',prepared_dir=relative(out/'development_cache'),report_root='outputs/training_v2'))
    save_json(out/'manifests/records.json',dict(records=records))


def approved_development_path(data_dir, record, path):
    """Use a fixed approved image root, never a record-supplied arbitrary directory."""
    data_dir=Path(data_dir).resolve();marker=data_dir/'dataset_v2.json'
    split=record['split']
    if split not in ('train','validation'):return False
    if not marker.exists():return path.is_relative_to(data_dir/'images'/split)
    root=(ROOT/json.loads(marker.read_text(encoding='utf-8'))['dataset_root']).resolve()
    if root.is_relative_to(ROOT/'datasets'):
        return path.is_relative_to(root/'public'/split/'images')
    folder='val' if split=='validation' else 'train'
    return root.is_relative_to(ROOT/'data') and path.is_relative_to(root/'yolo/images'/folder)


def prepare_development(data_dir, settings):
    """B1/B2 adapter. Read only the dedicated Train/Validation CSVs and image roots."""
    from src.data.development import read_csv
    data_dir=Path(data_dir).resolve();marker=json.loads((data_dir/'dataset_v2.json').read_text(encoding='utf-8'))
    status_path=(ROOT/marker['status_file']).resolve()
    state=json.loads(status_path.read_text(encoding='utf-8'))
    if state['status']!='READY' and settings.get('allow_candidate_annotations') is not True:
        raise ValueError('Dataset is CANDIDATE_NOT_READY: resolve human annotation/duplicate review before formal training')
    selected=[];seen=set();actual=defaultdict(Counter);scene_splits={}
    rows=read_csv(data_dir/'ground_truth_images.csv')
    for row in rows:
        if row['split'] not in ('train','validation'):raise ValueError('Held-out metadata found in development-only manifest')
        if row['annotation_status'] not in ('COMPLETE','VERIFIED','REVIEWED','OK'):raise ValueError('Unresolved annotation in development manifest')
        iid=row['image_id'];path=(ROOT/row['image_path']).resolve()
        if iid in seen or not approved_development_path(data_dir,row,path):raise ValueError('Duplicate ID or path escaped approved development roots')
        seen.add(iid);key=row['scene_group_id']
        if scene_splits.setdefault(key,row['split'])!=row['split']:raise ValueError('Scene group crosses train/validation')
        with Image.open(path) as src:
            image=ImageOps.exif_transpose(src).convert('RGB')
            if image.size!=(int(row['width']),int(row['height'])):raise ValueError('Image dimensions differ')
            pixel_sha=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
        counts=[int(row[f]) for f in COUNT_FIELDS]+[int(row['non_catalogue_count'])]
        if row['reference_label_all_one']!=count_label(Counter({c:v for c,v in zip((1,2,3,4,5,0),counts)})):
            raise ValueError('Fixed-work-order label/count conflict')
        selected.append(dict(image_id=iid,scene_id=row['scene_id'],scene_group=key,split=row['split'],label=row['reference_label_all_one'],
                             counts=counts,path=relative(path),width=int(row['width']),height=int(row['height']),sha256=digest(path),pixel_sha256=pixel_sha))
    for obj in read_csv(data_dir/'ground_truth_objects.csv'):
        if obj['image_id'] not in seen:raise ValueError('Object manifest escaped development IDs')
        if obj['ignore']=='0':actual[obj['image_id']][int(obj['class_id'])]+=1
    for r in selected:
        if [actual[r['image_id']][c] for c in (1,2,3,4,5,0)]!=r['counts']:raise ValueError('Object/image counts conflict')
    for field in ('sha256','pixel_sha256'):
        if len({r[field] for r in selected})!=len(selected):raise ValueError('Exact duplicate in development; select one representative before training')
    selected.sort(key=lambda r:(r['split'],r['image_id']))
    sources=[data_dir/'ground_truth_images.csv',data_dir/'ground_truth_objects.csv',data_dir/'dataset_v2.json',status_path]
    evidence=dict(summary={s:dict(images=sum(r['split']==s for r in selected),**Counter(r['label'] for r in selected if r['split']==s)) for s in ('train','validation')},
                  exclusions=[],expected_counts=reference_counts(),scene_confirmation=settings.get('scene_confirmation',{}),
                  source_files={relative(p):digest(p) for p in sources},dataset_status=state['status'],official_held_out_test='NOT_RUN')
    return selected,evidence


def verify(out, original):
    records=json.loads((out/'manifests/records.json').read_text(encoding='utf-8'))['records']
    checks={}; checks['original_files_unchanged']=all((ROOT/r['path']).is_file() and digest(ROOT/r['path'])==r['sha256'] for r in original)
    source_root=Path(os.path.commonpath([str((ROOT/r['path']).parent) for r in original]))
    checks['original_inventory_unchanged']={relative(p) for p in source_root.rglob('*') if p.is_file()}=={r['path'] for r in original}
    checks['unique_image_ids']=len({r['image_id'] for r in records})==len(records)
    checks['six_class_mapping']=json.loads((out/'configs/class_mapping.json').read_text())['official_to_yolo']=={str(c):c for c in range(6)}
    by_group=defaultdict(set);by_hash=defaultdict(set)
    for r in records:
        if r['split']=='quarantine':continue
        by_group[r['scene_group_id']].add(r['split']+':'+r['evaluation_stage'])
        by_hash[r['pixel_sha256']].add(r['split']+':'+r['evaluation_stage'])
    checks['scene_groups_disjoint']=all(len(v)==1 for v in by_group.values())
    checks['pixels_disjoint']=all(len(v)==1 for v in by_hash.values())
    checks['ground_truth_counts_and_labels']=all((not r['label'] and r['annotation_status']=='REVIEW') or r['label']==count_label(Counter({int(k):v for k,v in r['counts'].items()})) for r in records)
    checks['no_specialized_images_in_public_gt']=all(Path(r['source_image']).suffix.lower()=='.jpg' for r in records)
    checks['stage2_stage3_disjoint']=not ({r['scene_group_id'] for r in records if r['evaluation_stage']=='stage2'} & {r['scene_group_id'] for r in records if r['evaluation_stage']=='stage3'})
    checks['test_not_in_training_yaml']='test:' not in (out/'yolo/data.yaml').read_text()
    from src.evaluator import read_csv, unique, box
    images=unique(read_csv(out/'ground_truth_images.csv',['image_id','annotation_status']), 'image_id','v2 images')
    objects=read_csv(out/'ground_truth_objects.csv',['image_id','object_id','class_id','xmin','ymin','xmax','ymax','ignore'])
    by_image=defaultdict(list)
    for obj in objects:
        by_image[obj['image_id']].append(obj)
        row=images[obj['image_id']];box(obj,int(row['width']),int(row['height']),obj['image_id'])
    for r in records:
        info={'width':r['width'],'height':r['height'],'exif_orientation':1}
        parsed,issues,_=parse_annotation(ROOT/r['source_annotation'],info)
        expected_objects=[(o['class_id'],o['xyxy']) for o in parsed]
        gt_objects=[(int(o['class_id']),[float(o[k]) for k in ('xmin','ymin','xmax','ymax')]) for o in by_image[r['image_id']]]
        if expected_objects!=gt_objects:raise ValueError(f'JSON/GT mismatch: {r["image_id"]}')
        row=images[r['image_id']]
        vector=[int(row[f]) for f in COUNT_FIELDS]+[int(row['non_catalogue_count'])]
        if vector!=[r['counts'][str(c)] for c in (1,2,3,4,5,0)] or row['ground_truth_label']!=r['label']:
            raise ValueError('Image CSV counts or stored label differ from JSON-derived manifest')
        if len({o['object_id'] for o in by_image[r['image_id']]})!=len(by_image[r['image_id']]):raise ValueError('Duplicate object IDs')
        split='val' if r['split']=='validation' else r['split']
        lines=(out/'yolo/labels'/split/(Path(r['image_id']).stem+'.txt')).read_text().splitlines()
        if len(lines)!=len(parsed):raise ValueError('YOLO object count mismatch')
        for text,obj in zip(lines,parsed):
            cls,*values=text.split()
            restored=from_yolo([float(v) for v in values],r['width'],r['height'],decimal_places=12)
            if int(cls)!=obj['class_id'] or max(abs(a-b) for a,b in zip(restored,obj['xyxy']))>1e-6:
                raise ValueError('YOLO/JSON class or coordinate mismatch')
    checks['source_and_yolo_labels_roundtrip']=True
    source_info={r['path']:r for r in json.loads((out/'audit/image_inventory.json').read_text(encoding='utf-8'))}
    from src.data.development import read_csv
    special=read_csv(out/'manifests/dimension_manifest.csv')+read_csv(out/'calibration/ruler_manifest.csv')
    with ThreadPoolExecutor(max_workers=4) as pool:
        decoded=list(pool.map(pixel_info,[ROOT/r['image_path'] for r in special]))
    checks['specialized_png_pixels_match_normalized_heic']=all(a.get('pixel_sha256')==source_info[b['source_path']].get('pixel_sha256') for a,b in zip(decoded,special))
    checks['independent_copy_bytes_match']=all(digest(ROOT/r['image_path'])==r['sha256'] for r in records)
    save_json(out/'audit/acceptance.json',dict(checks=checks,software_checks_passed=all(checks.values()),
              annotation_visual_verification='PENDING',model_training='NOT_RUN',formal_test='NOT_RUN',
              notes='Annotation conversion/load checks do not establish label correctness, calibrated dimensions, specimen independence or model performance.'))
    if not all(checks.values()):raise ValueError(f'Acceptance failed: {checks}')
    return checks


def write_report(out):
    """Derived report only: no performance figures, physical measurements or model predictions."""
    stats=json.loads((out/'manifests/dataset_statistics.json').read_text(encoding='utf-8'))
    dup=json.loads((out/'audit/duplicates.json').read_text(encoding='utf-8'))
    old=json.loads((out/'audit/old_data_comparison.json').read_text(encoding='utf-8'))
    records=json.loads((out/'manifests/records.json').read_text(encoding='utf-8'))['records']
    public=stats['public']
    lines=['# 新数据集审查与候选划分（2026-11-26）','',
      '**状态：CANDIDATE_NOT_READY，未正式冻结；训练和正式Test均为NOT_RUN。**',
      '原始`数据集/`、P03、历史数据、权重、阈值及GUI均保留。数据处理前核对Project-Man第3–6、8、11–12页及team_plan_ai.md；保留Stage 2冻结基线，Stage 3保留集不参与开发。','',
      '## 收件与计划差异','',
      '| 用途 | 计划 | 实收 | JSON |','|---|---:|---:|---:|',
      '| 公共正常工况 | 420 | 420 | 420 |','| crowded | 70 | 70 | 70 |','| edge | 70 | 70 | 70 |',
      '| 尺子 | 10 | 10 | 不要求 |','| 尺寸专项 | 200 | 220 | 不要求 |',
      '| 合计 | 770 | 790 | 560 |','',
      '另有7份Excel。M4小垫片、M6盖型螺母专项分别实收40张，比计划各多10张；没有复制补数。230张HEIC全部可解码，独立导出无损RGB PNG；原始HEIC与EXIF保留。',
      '公共560张与560份LabelMe JSON一一对应；2811个rectangle标注，六类名称均合法，无缺失JSON、解码失败、越界/零面积框或完全重复框。polygon工具规则为顶点min/max水平包围框，本批无polygon。',
      '坐标为原图连续像素xyxy，左上为原点，0≤xmin<xmax≤width、0≤ymin<ymax≤height。框不自动裁剪；无未经依据的ignore，全部ignore=0。','',
      '## 真实标签与人工确认','',
      '固定工单位于config/inspection.json：ID1–5各1，ID0=0。标签从非忽略目标标注计数推导，与文件名前缀和模型预测无关。任何ID0、缺件、多件或空场景均FAIL。',
      'D18：JSON计数(ID1..5,ID0)=[0,1,1,0,1,2]，拍摄表=[0,1,1,1,1,1]。F21：JSON=[1,1,1,0,2,1]，拍摄表=[1,1,1,1,2,1]。用户已明确确认“已核实，两张都以JSON为准”；保留JSON和Excel，确认绑定原JSON SHA256。',
      'E.xlsx表头明确“为2的即为加入异常尺寸件”，因此2不是两个正常件。原始表值完整保存，Ground Truth以JSON为准，未把异常件映射为正常类别。',
      'C38实为同时缺ID4、多ID3（counts与拍摄表一致），分层scenario_type=F，原目录C保留；F08–F12等缺正常件加不同异常件，计数本身不能证明是同类替换，保留F语义与scenario_note。',
      'annotation_status=COMPLETE仅表示人工JSON格式完整及程序检查通过，**不表示已经完成全量视觉真值核验，不设VERIFIED**。已知两处冲突已解决；仍需P03确认全量/抽样质检范围、漏标与细规格真值。','',
      '## 候选划分','',
      'seed=42；二元scene-group分配，硬约束工况与PASS/FAIL数量及A–G覆盖，软目标平衡A–G、六类实例数及有记录的异常类型。配额可因真实缺收/无效标注而调整，禁止复制凑数。',
      '各照片独立摆放依据本次用户/采集者确认，自动scene_id来自全局唯一image_id。困难图按Excel图片编号和工况文字可靠关联，转H001.jpg等，原路径映射保留。不建立跨工况配对。','',
      '| 分区 | 图片 | PASS | FAIL | normal/crowded/edge | scene_id | scene_group |',
      '|---|---:|---:|---:|---|---:|---:|']
    for name in ('train','validation','test','stage2','stage3','quarantine'):
        s=public[name];c=s['conditions'];l=s['labels']
        lines.append(f"| {name} | {s['images']} | {l.get('PASS',0)} | {l.get('FAIL',0)} | {c.get('normal',0)}/{c.get('crowded',0)}/{c.get('edge',0)} | {s['scene_ids']} | {s['scene_groups']} |")
    lines+=['','G01–G07均为空台面，保守同组且只计一个场景支持，未用7张近乎相同空背景虚增独立物理场景。其他scene_id/scene_group按采集者确认，不能据此声称独立实物。','',
      '| 分区 | ID0 | ID1 | ID2 | ID3 | ID4 | ID5 | A/B/C/D/E/F/G |','|---|---:|---:|---:|---:|---:|---:|---|']
    for name in ('train','validation','test','stage2','stage3'):
        s=public[name];lines.append('| '+name+' | '+' | '.join(str(s['class_instances'][str(c)]) for c in range(6))+' | '+'/'.join(str(s['scenarios'].get(c,0)) for c in 'ABCDEFG')+' |')
    lines+=['','异常类型仅记录有明确拍摄表文字或D/E替换/添加栏依据的声明，不把ID0框虚构为已确认具体规格；缺失处UNKNOWN。D18拍摄表组成已被覆盖，额外ID0细类型保持UNKNOWN。各分区异常组合详见dataset_statistics.json；这统计的是含该声明的图片，不是已逐目标验证的异常实例数。','',
      '## 重复与泄漏','',
      f"公共集文件SHA256完全重复组={len(dup['exact_groups'])}（两个哈希通道独立记录，不能相加为图片数）；像素完全重复组=0。pHash≤8候选{len(dup['near_candidates'])}对，仅作候选，不据此合并非空独立摆放。G01–G07保守同组。",
      f"旧数据明确根目录共扫描{old['images_scanned']}张，文件/解码像素交叉相同匹配={len(old['matches'])}；旧新pHash≤6候选={len(old.get('near_candidates',[]))}，均来自一张尺寸专项盖型螺母与旧图片路径，公共560图对旧集感知候选为0。哈希不能证明物理场景或实物独立，任意裁剪与未经记录的外部使用历史仍无法完全排除。",
      '近重复候选和选定视觉预览在audit/duplicates.json、previews/；抽查发现相似背景/相似布局但有零件位置改变，采集者确认独立摆放仍是场景依据；未对127对全部做人工签字。不得把近重复候选数当作泄漏数。',
      '没有跨Train/Val/Test以及Stage 2/3的已确认像素重复或scene_group交叉。旧Test仅为本次授权的数据哈希审查读取，无推理、评估、调参。原开发训练清单与模型冻结证据不变。','',
      '## Stage 2/3隔离','',
      'Stage 2候选测试30张normal，15 PASS/15 FAIL，A–G均覆盖，含错规格替换、正常数量正确但额外ID0；满足组内数量目标。它不是已有模型的测试成绩。',
      'Stage 3保留90张：normal、crowded、edge各30张，各15 PASS/15 FAIL。两阶段图像与scene_group完全不交叉；属于分层独立样本，**不是同场景配对实验**。未运行Stage 3模型、设计改进或按测试错误调参。',
      '尚无证据表明新测试图曾用于本仓库冻结模型训练；外部未记录用途无法由代码证明，封存前请P02/P03确认。旧Stage 2模型/配置/验证结果不覆盖，新版本未READY，不自动冻结。','',
      '## 三模型与P08入口','',
      '- Baseline1/2共同数据：development/（只含Train/Validation CSV），configs/training.json；图片共用yolo/images/train和val，路径为项目相对路径。prepare入口防止未READY数据用于正式训练，--audit-candidate仅允许--prepare-only。',
      '- Baseline3：yolo/data.yaml，现有模型真实为六类YOLO，官方ID与内部索引0–5一一对应，已核对冻结映射；不改变网络。ID0完整保留，空标注生成空TXT；training YAML仅Train/Val，完全不含Test键。旧P06历史训练脚本的分支/环境/数据验收约束仍保留，正式训练新版本需P06明确批准新候选流程。',
      '- P08：configs/stage2_evaluation.json或stage3_evaluation.json；对应evaluation/<stage>/两份过滤GT和独立工单文件。p08_run.py支持--check-data-only，仅检查数据。正式运行要求READY且frozen，根目录全量Test不能隐式作为新阶段评价集。',
      '- predict.py接口未改变，仍只接受图片与不含GT的工单JSON；本次不调用新Test做预测。expected_counts.json（全公共集）不是推荐的阶段评估入口，应使用阶段专用配置。','',
      '## 尺子与尺寸专项','',
      'calibration/ruler/保存10份PNG，ruler_manifest.csv记录源HEIC、分辨率与可用EXIF；尺子位置、刻度点、px/mm、相机高度保持空，需人工标定。',
      'dimension_manifest.csv保存220张的目录声明规格及官方ID。卡尺实测、physical_sample_id、拍摄条件、开发/独立验证用途留空或UNASSIGNED；先确认实物编号和历史，再按实物整体分开发/验证，不能随机把同实物的照片拆成独立实物验证。',
      '没有卡尺表、真实尺度误差界或验证记录，不输出任何伪造尺寸或通过结论。M6盖型螺母同时有类型与规格差异，不把全部拒识收益归于尺寸。','',
      '## 自动验收与待办','',
      'audit/acceptance.json：源文件SHA256保全、唯一ID、类别、原JSON↔GT↔YOLO逐框/数量和≤1e-6px还原、像素与scene_group隔离、工单判定、专项隔离等。audit/interface_acceptance.json：真实模型加载、B1/B2数据读取、实际YOLO数据加载、P08阶段GT读取、seed42复现、历史模型/结果未变；不等于模型性能或标定验收。',
      '仍需：P03签署全量/抽样标注质检范围并核实漏标及细规格；复核感知候选与采集历史；提供实物编号/卡尺实测/尺子刻度与机位。当前文件可做加载及开发诊断，**正式训练前置条件尚未全部满足，状态维持NOT_READY**。','',
      '## 复现命令','', '从项目根目录、.venv311运行（源目录不改；新输出不覆盖历史）：','',
      '```powershell','python scripts/prepare_dataset_v2.py',
      'python scripts/prepare_dataset_v2.py --verify-only',
      'python -m scripts.train_baseline1 --dataset-config data/mechanical_components_v2/configs/training.json --prepare-only --audit-candidate',
      'python -m scripts.train_baseline2 --dataset-config data/mechanical_components_v2/configs/training.json --prepare-only --audit-candidate',
      'python scripts/check_dataset_v2.py',
      'python p08_run.py --evaluation-config data/mechanical_components_v2/configs/stage2_evaluation.json --check-data-only',
      'python p08_run.py --evaluation-config data/mechanical_components_v2/configs/stage3_evaluation.json --check-data-only','```','',
      '源HEIC解码使用本地.tools/pillow_heif（1.9.0），不进入模型运行requirements；若缺失，按脚本提示安装到.tools。数据/图片仅在本机data/，已被.gitignore忽略；精简文本审查在results/data_manifests/mechanical_components_v2/。',
      '全部生成文件见manifests/generated_files.csv；原始清单与SHA256见audit/source_files.json。各文件UTF-8（CSV带BOM方便Excel），清单使用项目相对路径。LLM协助记录见docs/p07/advice_record.md。本次未训练、未修改权重、未生成性能指标、未commit/push。','']
    (out/'README.md').write_bytes(('\n'.join(lines)).encode('utf-8'))
    inventory=[dict(path=relative(p),size_bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='generated_files.csv' and p.suffix!='.cache']
    save_csv(out/'manifests/generated_files.csv',inventory)
