"""P07 live-camera processing latency. Never substitutes files or cached frames."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
from src.camera import FrameProvider
from src.p07_bridge import load_backend, validate_result
from src.work_orders import reference_counts, validate_counts

MODELS = ('baseline1', 'baseline2', 'baseline3')
NORMAL = {'PASS', 'FAIL', 'REVIEW'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def statistics(rows, model, requested, warmup):
    measured = [r for r in rows if r['model'] == model and r['phase'] == 'measure']
    acquired = [r for r in measured if r['processing_status'] in ('OK', 'ERROR', 'UNSUPPORTED')]
    ok = [r for r in acquired if r['processing_status'] == 'OK']
    capture_failures = sum(r['processing_status'] == 'CAPTURE_ERROR' for r in measured)
    timeouts = sum(r['processing_status'] == 'CAPTURE_TIMEOUT' for r in measured)
    values = np.asarray([r['processing_latency_ms'] for r in ok], dtype=float)
    attempts = len(acquired) + capture_failures + timeouts
    result = dict(model=model, requested_frames=requested, acquired_new_frames=len(acquired),
        successful_frames=len(ok), failed_frames=attempts-len(ok), processing_errors=len(acquired)-len(ok),
        capture_failures=capture_failures, capture_timeouts=timeouts,
        coverage=len(ok)/attempts if attempts else 0.,
        processing_coverage=len(ok)/len(acquired) if acquired else 0.,
        warmup_acquired=sum(r['model']==model and r['phase']=='warmup' and
                            r['processing_status'] in ('OK','ERROR','UNSUPPORTED') for r in rows),
        median_ms=None, p95_ms=None, mean_ms=None, min_ms=None, max_ms=None, std_ms=None,
        processing_fps=None, repeated_pixel_hashes=sum(bool(r.get('repeated_pixels')) for r in acquired),
        geometry_measured_frames=sum(r.get('geometry_objects_measured',0)>0 for r in acquired),
        geometry_attempted_frames=sum(r.get('geometry_objects_attempted',0)>0 for r in acquired),
        minimum_20_status='INSUFFICIENT', requested_protocol_complete=False)
    if len(values):
        result.update(median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95,method='linear')),
            mean_ms=float(np.mean(values)),min_ms=float(np.min(values)),max_ms=float(np.max(values)),
            std_ms=float(np.std(values,ddof=0)),processing_fps=float(1000/np.mean(values)))
    live=[r for r in ok if r.get('live_latency_ms') is not None]
    result.update(live_median_ms=None,live_p95_ms=None,observed_sample_fps=None,
                  live_minimum_20_status='NOT_MEASURED',observed_processing_fps=None,
                  theoretical_serial_capacity_fps=result['processing_fps'],peak_rss_bytes=None)
    completed=[r for r in ok if r.get('processing_done_perf') is not None]
    if completed:
        span=max(r['processing_done_perf'] for r in completed)-min(r['available_perf'] for r in completed)
        result['observed_processing_fps']=len(completed)/span if span>0 else None
        memory=[r['rss_bytes'] for r in completed if r.get('rss_bytes') is not None]
        result['peak_rss_bytes']=max(memory) if memory else None
    result['processing_fps']=result['observed_processing_fps']
    if live:
        result['live_median_ms']=float(np.median([r['live_latency_ms'] for r in live]))
        result['live_p95_ms']=float(np.percentile([r['live_latency_ms'] for r in live],95,method='linear'))
        span=max(r['display_done_perf'] for r in live)-min(r['available_perf'] for r in live)
        result['observed_sample_fps']=len(live)/span if span>0 else None
        result['live_minimum_20_status']='SUFFICIENT_READS_CHECK_PHYSICAL_CONDITIONS' if len(live)>=20 else 'INSUFFICIENT'
    hashes={r['pixel_sha256'] for r in ok if r.get('pixel_sha256')}
    if len(ok)>=20:
        result['minimum_20_status']='MET_NEW_READ_PROTOCOL' if len(hashes)>1 else 'NOT_VERIFIED_IDENTICAL_PIXELS'
    result['requested_protocol_complete']=(len(acquired)==requested and result['warmup_acquired']==warmup)
    return result


class CaptureLog:
    """FrameProvider journal adapter; memory only during acquisition/processing."""
    def __init__(self, events):
        self.events=events;self.lock=threading.Lock();self.model='';self.phase='warmup'

    def capture(self, kind, value, backend, image, read_ms, fps, ok):
        with self.lock:
            fid=len(self.events)+1
            self.events.append(dict(frame_id=fid,model=self.model,phase=self.phase,capture_time_utc=utc(),
                backend=backend,read_success=bool(ok),width=image.shape[1] if ok else 0,
                height=image.shape[0] if ok else 0,frame_read_ms=read_ms,capture_fps=fps,selected=False))
            return fid

    def event(self, fid):
        with self.lock:
            event=self.events[fid-1];event['selected']=True
            return dict(event)


def fresh_frame(provider, after, last_id, timeout):
    deadline=time.perf_counter()+timeout
    while time.perf_counter()<deadline:
        packet=provider.latest
        if packet is not None and packet.frame_id>last_id and packet.available_at>after:
            return packet
        if provider.finished:
            break
        time.sleep(.001)
    return None


def process(backend, packet, expected, model):
    """Drain previous GPU work outside timing; completion sync is inside timing."""
    backend.synchronize()
    start=time.perf_counter()
    try:
        result=validate_result(backend.infer(packet.image,expected),model,packet.image.shape)
        backend.synchronize()
        return result,(time.perf_counter()-start)*1000
    except Exception as exc:
        return dict(status='ERROR',message=f'{type(exc).__name__}: {exc}'),(time.perf_counter()-start)*1000


def require_size_scope(model,result):
    if model=='baseline3' and result.get('inspection',{}).get('calibration_compatible') is False:
        raise ValueError('Baseline3 full-size latency NOT_VERIFIED: capture geometry does not match calibration; do not benchmark a skipped size branch')


def display_result(packet, result, expected, model, info):
    """Same P07 rendering; return only after imshow and event-update calls."""
    import cv2
    from src.p07_ui import present
    canvas=present(packet,result,expected,model,'real',info)
    cv2.imshow('P07 live benchmark',canvas)
    return cv2.waitKey(1)&255 not in (ord('q'),27)


def environment():
    dependencies={}
    for name in ('numpy','torch','torchvision','opencv-python','ultralytics','scikit-learn'):
        try:dependencies[name]=version(name)
        except PackageNotFoundError:dependencies[name]=None
    cpu=platform.processor()
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            cpu=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
    except (OSError,ImportError):pass
    try:
        r=subprocess.run(['powershell','-NoProfile','-Command',
            '(Get-CimInstance Win32_Processor).Name'],capture_output=True,timeout=10)
        if r.returncode==0:cpu=r.stdout.decode(errors='replace').strip() or cpu
    except (OSError,subprocess.TimeoutExpired):pass
    import torch
    gpus=[dict(index=i,name=torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
    def git(*args):
        r=subprocess.run(['git',*args],cwd=ROOT,capture_output=True)
        return r.stdout.decode('utf-8',errors='replace').strip() if r.returncode==0 else 'UNAVAILABLE'
    return dict(cpu=cpu,gpu=gpus,python=sys.version,executable=sys.executable,
                dependencies=dependencies,platform=platform.platform(),git_commit=git('rev-parse','HEAD'),
                git_worktree_before_run=git('status','--short'),cuda_runtime=torch.version.cuda)


def export(out, rows, events, config):
    summaries=[statistics(rows,m,config['frames'],config['warmup']) for m in config['models']]
    def table(name, fields, data):
        with (out/name).open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(data)
    fields=['model','phase','frame_id','capture_time_utc','processing_latency_ms','final_status',
            'processing_status','failure_reason','decision_reason','width','height','queue_age_ms',
            'pixel_sha256','repeated_pixels','geometry_objects_measured','geometry_objects_attempted',
            'live_latency_ms','available_perf','display_done_perf','processing_done_perf','rss_bytes','gpu_allocated_bytes',
            'preprocess_ms','model_inference_ms','postprocess_ms','timing_segments_status']
    table('frame_latency.csv',fields,[{k:r.get(k) for k in fields} for r in rows])
    table('summary.csv',list(summaries[0]),summaries)
    event_fields=['frame_id','model','phase','capture_time_utc','backend','read_success','width','height',
                  'frame_read_ms','capture_fps','selected']
    table('capture_events.csv',event_fields,events)
    (out/'run_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# Live Processing Latency Benchmark','',
        'Purpose: compare the configured P07 baselines on newly read Camo camera frames.',
        f"Run status: **{config['run_status']}**. Camera {config['camera']}. Configured warm-up / measured frames per model: {config['warmup']} / {config['frames']}.",
        f"CPU: {config['environment']['cpu']}. GPU: {config['environment']['gpu']}.",
        f"Software: {config['environment']['dependencies']}.",'',
        'Models run sequentially. Warm-up samples are excluded. Actual devices, resolutions, model identities, active configuration and calibration are in run_config.json.',
        'Timing: perf_counter starts after selecting a fresh read and draining prior GPU work; ends after infer, postprocessing, final business decision, result validation and GPU completion synchronization. Loading, warm-up, camera waiting, hashing, progress printing and file export are excluded.',
        'This is NOT exposure-to-display or end-to-end latency. Phone exposure, Camo transmission and GUI rendering are excluded.',
        'NumPy median; P95 percentile method=linear; population standard deviation ddof=0. Processing FPS=1000/mean processing milliseconds, not capture/display throughput.',
        'PASS, FAIL and REVIEW are completed processing samples. ERROR and unsupported work orders are failures. Coverage=completed samples/(acquired samples + camera-read failures + wait timeouts); processing coverage uses acquired samples only. Failed-processing durations remain in frame_latency.csv but do not enter successful-processing statistics.','',
        '| Model | New reads | Completed | Failed attempts | Coverage | Median ms | P95 ms | Mean ms | Processing FPS | >=20 evidence |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    def fmt(v):return 'N/A' if v is None else f'{v:.3f}'
    for s in summaries:
        lines.append(f"| {s['model']} | {s['acquired_new_frames']} | {s['successful_frames']} | {s['failed_frames']} | {s['coverage']:.1%} | {fmt(s['median_ms'])} | {fmt(s['p95_ms'])} | {fmt(s['mean_ms'])} | {fmt(s['processing_fps'])} | {s['minimum_20_status']} |")
    lines += ['', 'Actual capture resolutions and devices: '+str({m:{k:v for k,v in info.items() if k in ('actual_resolutions','device','error')}
                    for m,info in config['model_runs'].items()}),
              'Per-model decision status support: '+str({m:dict(Counter(r.get('final_status','CAPTURE_FAILURE') for r in rows
                    if r['model']==m and r['phase']=='measure')) for m in config['models']})]
    lines+=['','## Failures and limitations',
        'See per-frame errors, decision reasons and complete capture-attempt log. Skipped successful camera reads are logged but are not benchmark samples.',
        'Unique FrameProvider read IDs and a post-previous-processing arrival barrier prevent application-cache reuse. They do not prove unique phone exposures: OpenCV/Camo can replay frozen pixels. Repeated pixel hashes are reported without discarding samples; an entirely identical stream cannot establish fresh-image evidence.',
        'Baseline3 must be size_verified. Geometry may be skipped by its real calibration/measurement guards; geometry_measured_frames and calibration diagnostics disclose this. Such timing is the current configured pipeline, NOT proven full physical-metrology workload or functional compliance.',
        'At least 20 successfully processed new reads are necessary. NOT_VERIFIED_IDENTICAL_PIXELS, interrupted/incomplete runs or unsuitable physical conditions require further evidence. No accuracy or official held-out Test evaluation is performed.',
        f"Run errors: {config['errors']}"]
    if config.get('display'):
        lines+=['','Display-enabled timing starts at FrameProvider packet.available_at and ends after the actual P07 render, imshow and waitKey calls, including GPU completion synchronization. It excludes exposure/transport before frame availability and physical monitor refresh.',
                'Live Median/P95 and observed_sample_fps are in summary.csv. The latter uses the actual selected-sample time span, not reciprocal model latency. Capture FPS is recorded by FrameProvider in capture_events.csv.']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summaries


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera',type=int,default=0)
    parser.add_argument('--frames',type=int,default=100)
    parser.add_argument('--warmup',type=int,default=20)
    parser.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    parser.add_argument('--config',type=Path,default=ROOT/'config/p07.json')
    parser.add_argument('--frame-timeout',type=float,default=15.)
    parser.add_argument('--display',action='store_true',default=True,help='Measure through P07 display-update calls (default)')
    parser.add_argument('--no-display',dest='display',action='store_false',help='Diagnostic processing-only run; not official live latency')
    args=parser.parse_args(argv)
    if args.frames<1 or args.warmup<0 or args.camera<0 or not np.isfinite(args.frame_timeout) or args.frame_timeout<=0 or len(set(args.models))!=len(args.models):
        parser.error('Require positive frames/timeout, nonnegative camera/warmup and unique models')
    active=json.loads(args.config.read_text(encoding='utf-8-sig'))
    if 'baseline3' in args.models:
        from src.model_registry import resolve
        resolved,_=resolve('baseline3',active)
        if resolved.get('model_config',{}).get('baseline3_mode') not in ('size_verified','size_repaired_development'):
            parser.error('Baseline3 benchmark requires full size verification; no YOLO-only mode')
    expected=validate_counts(active.get('expected_counts',reference_counts()))
    config=dict(environment=environment(),camera=args.camera,frames=args.frames,warmup=args.warmup,
        models=[m for m in MODELS if m in args.models],expected_counts=expected,
        config_path=str(args.config.resolve()),config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        active_config=active,frame_timeout=args.frame_timeout,run_status='RUNNING',errors=[],model_runs={},
        timing_clock='time.perf_counter',p95_method='numpy.percentile(method=linear)',std_ddof=0,
        freshness='new FrameProvider read ID, available_at after previous iteration; no rejudge/cache reuse',
        timing_boundary='after fresh frame selection + pre-sync, through infer/decision/validation + completion sync; excludes capture/exposure/transport/GUI/load/logging/export',
        official_test='NOT_RUN',display=args.display,
        live_timing_boundary='FrameProvider available_at -> GPU complete + P07 render + imshow/waitKey return; physical monitor refresh/exposure/transport before availability excluded' if args.display else 'NOT_MEASURED')
    out=ROOT/'results/live_benchmark'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    print('Output:',out,flush=True)
    rows=[];events=[];observer=CaptureLog(events);provider=None;display_opened=False
    try:
        if args.display:
            import cv2
            cv2.namedWindow('P07 live benchmark',cv2.WINDOW_NORMAL)
            display_opened=True
        for model in config['models']:
            observer.model=model;observer.phase='warmup'
            cursor=len(events)
            info=config['model_runs'][model]={}
            try:
                backend=load_backend('real',model,active)
                if model=='baseline3' and backend.metadata.get('baseline3_mode') not in ('size_verified','size_repaired_development'):
                    raise RuntimeError('Loaded Baseline3 is not the integrated size mode')
                info['metadata']=backend.metadata
                info['device']=backend.metadata.get('hardware',{}).get('device','cpu')
                info['actual_resolutions']=[];info['actual_backends']=[]
                print(f"{model}: device={info['device']}, mode={backend.metadata.get('baseline3_mode','image_level')}",flush=True)
                provider=FrameProvider('camera',args.camera,active,observer).start()
                last_id=0;previous_hash=None
                for phase,count in [('warmup',args.warmup),('measure',args.frames)]:
                    print(f'{model}: {phase} {count} fresh camera reads',flush=True)
                    observer.phase=phase;barrier=time.perf_counter()
                    for index in range(count):
                        packet=fresh_frame(provider,barrier,last_id,args.frame_timeout)
                        if packet is None:
                            rows.append(dict(model=model,phase=phase,processing_status='CAPTURE_TIMEOUT',
                                failure_reason=provider.error or 'No fresh camera frame before timeout'))
                            raise TimeoutError(provider.error or 'No fresh camera frame')
                        last_id=packet.frame_id;event=observer.event(last_id)
                        if int(packet.image.max())<=1:
                            rows.append(dict(model=model,phase=phase,frame_id=last_id,processing_status='CAPTURE_ERROR',
                                failure_reason='Camera delivered a black frame; physical source is NOT_VERIFIED'))
                            raise RuntimeError('Black camera input; connect/unlock Camo and show the physical rig before benchmarking')
                        age=(time.perf_counter()-packet.available_at)*1000
                        interrupted=False
                        try:result,latency=process(backend,packet,expected,model)
                        except KeyboardInterrupt:
                            result,latency=dict(status='ERROR',message='Interrupted during processing'),None
                            interrupted=True
                        except Exception as exc:result,latency=dict(status='ERROR',message=str(exc)),None
                        processing_done=time.perf_counter()
                        normal=result['status'] in NORMAL
                        live_ms=display_done=None;keep_running=True
                        if args.display and not interrupted:
                            try:
                                keep_running=display_result(packet,result,expected,model,
                                    dict(source='camera',camera_index=args.camera,detecting=True,device=info['device'],
                                         model_mode=backend.metadata.get('baseline3_mode'),version=backend.metadata.get('model_version','')))
                                display_done=time.perf_counter();live_ms=(display_done-packet.available_at)*1000
                            except Exception as exc:
                                result=dict(status='ERROR',message='Display update: '+str(exc));normal=False
                        pixel_hash=hashlib.sha256(packet.image.tobytes()).hexdigest()
                        measured=sum(bool(o.get('size_evidence',{}).get('measurement',{}).get('dimensions')) for o in result.get('objects',[]))
                        rows.append(dict(model=model,phase=phase,frame_id=last_id,capture_time_utc=event['capture_time_utc'],
                            processing_latency_ms=latency,final_status=result['status'],processing_status='OK' if normal else
                            ('UNSUPPORTED' if result['status']=='UNSUPPORTED_COUNTS' else 'ERROR'),
                            failure_reason='' if normal else result.get('message',result['status']),
                            decision_reason=result.get('message',''),width=packet.image.shape[1],height=packet.image.shape[0],
                            queue_age_ms=age,pixel_sha256=pixel_hash,repeated_pixels=pixel_hash==previous_hash,
                            geometry_objects_measured=measured,live_latency_ms=live_ms,
                            available_perf=packet.available_at,display_done_perf=display_done))
                        import psutil,torch
                        rows[-1].update(processing_done_perf=processing_done,rss_bytes=psutil.Process().memory_info().rss,
                            gpu_allocated_bytes=torch.cuda.memory_allocated() if torch.cuda.is_available() else None,
                            preprocess_ms=None,model_inference_ms=None,postprocess_ms=None,
                            timing_segments_status='NOT_SEPARATELY_INSTRUMENTED: total full pipeline is measured without modifying frozen algorithms')
                        rows[-1]['geometry_objects_attempted']=sum('measurement' in o.get('size_evidence',{}) for o in result.get('objects',[]))
                        try:require_size_scope(model,result)
                        except ValueError as exc:
                            rows[-1].update(processing_status='ERROR',failure_reason=str(exc))
                            info['full_size_status']='NOT_VERIFIED_CALIBRATION_MISMATCH'
                            raise
                        previous_hash=pixel_hash
                        resolution=[packet.image.shape[1],packet.image.shape[0]]
                        if resolution not in info['actual_resolutions']:info['actual_resolutions'].append(resolution)
                        if event['backend'] not in info['actual_backends']:info['actual_backends'].append(event['backend'])
                        if interrupted or not keep_running:raise KeyboardInterrupt
                        if phase=='measure' and (index+1)%10==0:
                            print(f"{model}: {index+1}/{count}, {latency if latency is not None else 'ERROR'} ms, {result['status']}",flush=True)
                        barrier=time.perf_counter()  # Next sample must arrive after all previous processing/log work.
            except Exception as exc:
                info['error']=f'{type(exc).__name__}: {exc}';config['errors'].append(model+': '+info['error'])
                rows.append(dict(model=model,phase='load_or_run',processing_status='RUN_ERROR',failure_reason=info['error']))
                print(model,info['error'],flush=True)
            finally:
                shutdown_error=None
                if provider is not None:
                    try:provider.close()
                    except Exception as exc:
                        shutdown_error=exc;config['errors'].append('Camera shutdown: '+str(exc))
                    finally:provider=None
                for event in events[cursor:]:
                    if not event['read_success']:
                        rows.append(dict(model=model,phase=event['phase'],frame_id=event['frame_id'],
                            capture_time_utc=event['capture_time_utc'],processing_status='CAPTURE_ERROR',
                            failure_reason='Camera read/open did not deliver an image'))
                if 'backend' in locals():del backend
                if shutdown_error:raise shutdown_error
            s=statistics(rows,model,args.frames,args.warmup)
            if model=='baseline3' and s['geometry_attempted_frames']==0 and 'error' not in info:
                info['full_size_status']='NOT_VERIFIED_NO_MEASUREMENT_CALLS'
                config['errors'].append('Baseline3: no geometry attempts in measurement frames; full-size branch performance NOT_VERIFIED')
            print(model,json.dumps(s),flush=True)
        config['run_status']='COMPLETED' if not config['errors'] else 'COMPLETED_WITH_ERRORS'
    except KeyboardInterrupt:
        config['run_status']='INTERRUPTED';config['errors'].append('KeyboardInterrupt')
    except Exception as exc:
        config['run_status']='ABORTED';config['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        if display_opened:
            import cv2
            try:cv2.destroyWindow('P07 live benchmark')
            except Exception as exc:config['errors'].append('Display shutdown: '+str(exc))
        config['finished_utc']=utc()
        summaries=export(out,rows,events,config)
        print('Saved:',out,flush=True)
    return 0 if config['run_status']=='COMPLETED' and all(s['successful_frames']>=20 and
        s['requested_protocol_complete'] and s['minimum_20_status']=='MET_NEW_READ_PROTOCOL' for s in summaries) else 1


if __name__=='__main__':
    raise SystemExit(main())
