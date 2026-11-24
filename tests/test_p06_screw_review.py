import unittest

from tools.p06_screw_review import KEYS, KNOWN, PHASES, choose, compare_marks


class ScrewReviewTests(unittest.TestCase):
    def test_selection_is_stratified_disjoint_and_repeatable(self):
        rows = []
        for cid in (1, 2, 3):
            for i in range(30):
                oid = KNOWN[i] if cid == 2 and i < 2 else f'{cid}_{i}'
                rows.append(dict(object_id=oid, image_id=f'{i}.jpg', class_id=cid, scale_ok=True,
                                 **{p: {'dimensions': bool(i % 3)} for p in PHASES}))
        a = choose(rows)
        self.assertEqual(a, choose(rows[::-1]))
        self.assertEqual(len({r['image_id'] for r in a}), 24)
        for cid in (1,2,3):
            self.assertEqual(sum(r['class_id']==cid and r['role']=='review' for r in a), 2)
        self.assertTrue(set(KNOWN) <= {r['object_id'] for r in a if r['role']=='development'})

    def setUp(self):
        self.queue = dict(queue_id='q', scale_px_per_mm=[2,4], tasks=[
            dict(object_id='s', role='development', crop_xyxy=[100,200,220,300])])
        line = dict(line_xy=[[110,201],[110,213]], mm_estimate=3)
        self.rows = [dict(object_id='s', **{p:{'dimensions':{KEYS[0]:line}} for p in PHASES})]
        self.data = dict(version=1, coordinates='original_display_xy', queue_id='q',
                         marks={'s':{KEYS[0]:[[110,200],[110,216]], KEYS[1]:None}})

    def test_lines_use_original_coordinates_and_separate_xy_scale(self):
        a,b = compare_marks(self.queue,self.rows,self.data)
        self.assertEqual(a['before']['length_difference_px'],-4)
        self.assertEqual(a['before']['length_difference_candidate_mm'],-1)
        self.assertEqual(a['before']['shoulder_offset_px'],1)
        self.assertEqual(a['before']['tip_offset_px'],3)
        self.assertEqual(b['human'],'unclear')

    def test_missing_and_half_lines_are_pending(self):
        self.data['marks']['s'] = {KEYS[0]:[[101,202]]}
        self.assertEqual([r['human'] for r in compare_marks(self.queue,self.rows,self.data)],['pending']*2)

    def test_invalid_origin_and_foreign_queue_rejected(self):
        for value in ([[0,0],[1,1]], [[100,200],[100,200]], [[100,200],[220,201]]):
            self.data['marks']['s'][KEYS[0]]=value
            with self.assertRaises(ValueError):
                compare_marks(self.queue,self.rows,self.data)
        self.data['queue_id']='other'
        with self.assertRaisesRegex(ValueError,'different queue'):
            compare_marks(self.queue,self.rows,self.data)
