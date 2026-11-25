"""Synthetic foreground association examples, no real datasets or detector."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.p06_foreground import ForegroundAssociator
from src.p06_inference import _observations


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'output/p06_foreground_stage2')
    dest=parser.parse_args().output.resolve()
    if dest.exists() or not dest.is_relative_to(ROOT/'output'):
        parser.error('Choose a new output directory')
    bg=np.full((240,400,3),240,np.uint8)
    frame=bg.copy()
    cv2.rectangle(frame,(60,70),(110,150),(25,)*3,-1)
    cv2.circle(frame,(280,110),25,(50,)*3,-1)
    associator=ForegroundAssociator(bg,reference_id='SYNTHETIC_BLANK',capture_geometry_id='SYNTHETIC_FIXTURE')
    examples={}
    cases=dict(missed_object=[[58.,68.,113.,153.,.9,2]],
               same_class_duplicate=[[58.,68.,113.,153.,.9,2],[59.,69.,112.,152.,.7,2]],
               cross_class_duplicate=[[58.,68.,113.,153.,.9,2],[59.,69.,112.,152.,.7,3]],
               empty=[])
    dest.mkdir(parents=True)
    for name,rows in cases.items():
        image=bg if name=='empty' else frame
        result=associator.process(image,_observations(rows,400,240),capture_geometry_id='SYNTHETIC_FIXTURE')
        examples[name]=result
        drawn=image.copy()
        for region in result['regions']:
            cv2.rectangle(drawn,(region['xmin'],region['ymin']),(region['xmax']-1,region['ymax']-1),(0,170,0),2)
        for obj in result['detector_observations']['objects']:
            cv2.rectangle(drawn,(int(obj['xmin']),int(obj['ymin'])),(int(obj['xmax']),int(obj['ymax'])),(180,40,190),1)
        cv2.putText(drawn,'SYNTHETIC '+name,(5,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
        if not cv2.imwrite(str(dest/(name+'.png')),drawn):raise OSError('Failed to save preview')
    (dest/'examples.json').write_text(json.dumps(dict(evidence='SYNTHETIC_FUNCTIONAL_EXAMPLES',examples=examples),indent=2),encoding='utf-8')
    print(dest)


if __name__=='__main__':main()
