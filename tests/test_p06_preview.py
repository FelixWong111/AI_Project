"""EXIF preview regressions use synthetic temporary inputs only."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from src.p06_data import DataError, inspect_dataset
from tools.p06_preview_annotations import COLORS, draw_preview, load_preview, generate


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()

    def fixture(self, orientation=None, empty=False):
        image = Image.new('RGB', (120,80), 'white')
        image.putpixel((10,20),(255,0,0))
        image.putpixel((90,60),(0,0,255))
        kwargs = {}
        if orientation is not None:
            exif=Image.Exif(); exif[274]=orientation
            kwargs['exif']=exif
        image.save(self.source/'pass01.png', **kwargs)
        dims=(80,120) if orientation==6 else (120,80)
        doc=dict(imagePath='pass01.png',imageWidth=dims[0],imageHeight=dims[1],flags={},
                 shapes=[] if empty else [dict(label='non_catalogue',shape_type='rectangle',
                                               points=[[15,70],[50,105]],flags={})])
        if orientation!=6 and not empty:
            doc['shapes'][0]['points']=[[15,20],[50,55]]
        path=self.source/'pass01.json'
        path.write_text(json.dumps(doc))
        return path

    def test_orientation6_size_pixel_mapping_and_direct_box(self):
        path=self.fixture(6)
        image,record,objects=load_preview(path)
        self.assertEqual(image.size,(80,120))
        self.assertEqual((record['original_width'],record['original_height']),(120,80))
        # Independently known clockwise rotation: (x,y) -> (H-1-y,x).
        self.assertEqual(image.getpixel((59,10)),(255,0,0))
        self.assertEqual(image.getpixel((19,90)),(0,0,255))
        self.assertEqual(objects[0]['xyxy'],[15,70,50,105])
        rendered=draw_preview(image,objects)
        self.assertEqual(rendered.size,(80,120))
        self.assertEqual(rendered.getpixel((15,90)),COLORS[0])
        self.assertEqual(rendered.getpixel((60,90)),image.getpixel((60,90)))
        # Existing strict conversion checker must still reject EXIF=6.
        self.assertIn('EXIF_ORIENTATION',[e['type'] for e in inspect_dataset(self.source)['errors']])

    def test_no_exif(self):
        image,record,objects=load_preview(self.fixture())
        self.assertEqual(image.size,(120,80))
        self.assertEqual(record['exif_orientation'],1)
        self.assertEqual(objects[0]['xyxy'],[15,20,50,55])

    def test_dimension_mismatch_refused(self):
        path=self.fixture(6)
        doc=json.loads(path.read_text()); doc['imageWidth']=120; doc['imageHeight']=80
        path.write_text(json.dumps(doc))
        with self.assertRaisesRegex(DataError,'display'):
            load_preview(path)

    def test_confirmed_empty_has_no_id0_or_drawn_pixels(self):
        path=self.fixture(6,empty=True)
        with self.assertRaises(DataError): load_preview(path)
        image,record,objects=load_preview(path,{'pass01.png'})
        self.assertEqual(objects,[])
        self.assertEqual(record['preview_status'],'CONFIRMED_EMPTY')
        self.assertEqual(record['count_0'],0)
        self.assertEqual(draw_preview(image,objects).tobytes(),image.tobytes())

    def test_outputs_preserve_sources_and_refuse_overwrite(self):
        self.fixture(6)
        before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.iterdir()}
        out=self.root/'out'
        summary=generate(self.source,out)
        self.assertEqual(summary['generated'],1)
        with Image.open(out/'individual/pass01.png') as image:
            self.assertEqual(image.size,(80,120))
            self.assertEqual(image.getexif().get(274,1),1)
        self.assertTrue((out/'contact_sheets/PASS-01.jpg').exists())
        self.assertEqual(before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.iterdir()})
        with self.assertRaises(FileExistsError): generate(self.source,out)
        with self.assertRaises(DataError): generate(self.source,self.source/'out')

    def test_one_mismatch_is_reported_without_guessing(self):
        path=self.fixture(6)
        doc=json.loads(path.read_text()); doc['imageHeight']=80
        path.write_text(json.dumps(doc))
        out=self.root/'out'
        summary=generate(self.source,out)
        self.assertEqual(summary['generated'],0)
        self.assertEqual(summary['failed'],1)
        self.assertFalse((out/'individual/pass01.png').exists())
        self.assertIn('DIMENSION_MISMATCH',(out/'orientation_check.md').read_text(encoding='utf-8'))


if __name__=='__main__':
    unittest.main()
