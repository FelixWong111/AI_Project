"""Verify class-aware one-to-one diagnostic matching on synthetic boxes."""
import unittest
from tools.p06_validate_train_limited import matched


class LimitedMatchingTests(unittest.TestCase):
    def box(self,cid=2,score=.8,detection_id=0,**fields):
        return dict(xmin=0,ymin=0,xmax=20,ymax=20,class_id=cid,score=score,detection_id=detection_id,**fields)

    def test_duplicate_predictions_cannot_both_match_one_truth(self):
        gt=[self.box(object_id='g')]
        self.assertEqual(matched([self.box(detection_id=1),self.box(score=.9)],gt),[(0,'g')])

    def test_class_confusion_is_unmatched(self):
        self.assertEqual(matched([self.box(cid=3)],[self.box(object_id='g')]),[])

    def test_half_iou_boundary_and_empty_inputs(self):
        gt=[dict(self.box(object_id='g'),xmax=40)]
        self.assertEqual(matched([self.box()],gt),[(0,'g')])
        self.assertEqual(matched([],gt),[])
        self.assertEqual(matched([self.box()],[]),[])
