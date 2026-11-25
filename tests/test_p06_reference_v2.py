"""Count/group diagnostics only; no real data or model inference."""
import unittest
from tools.p06_reference_v2 import group_records,count_summary,verify_support
from src.data.p06_labelme import DataError
from types import SimpleNamespace


class ReferenceTests(unittest.TestCase):
    def test_groups_exclude_train_and_partition_validation(self):
        records=[dict(split='train',provenance='new275'),dict(split='val',provenance='old51'),dict(split='val',provenance='new275')]
        g=group_records(records)
        self.assertEqual([len(g[k]) for k in ('all65','old10','new55')],[2,1,1])

    def test_id0_empty_and_count_metrics(self):
        rows=[dict(image_id='empty',gt_counts=[0]*6,class_counts=[1,0,0,0,0,0]),
              dict(image_id='foreign',gt_counts=[2,1,0,0,0,0],class_counts=[1,1,0,0,0,0])]
        s=count_summary(rows)
        self.assertEqual(s['five_counts_exact'],2); self.assertEqual(s['six_counts_exact'],0)
        self.assertEqual(s['id0_false_positive_instances_on_id0_free_images'],1)
        self.assertEqual(s['count_mae'],[1,0,0,0,0,0])
        self.assertEqual(s['gt_instances'],3)

    def test_empty_group_has_no_fabricated_mae(self):
        self.assertEqual(count_summary([])['count_mae'],[None]*6)

    def test_official_support_uses_callback_seen_not_model_attribute(self):
        metrics=SimpleNamespace(nt_per_class=[1,2,0,0,0,0])
        verify_support(metrics,[1,2,0,0,0,0],2,2)
        for support,seen in (([1,1,0,0,0,0],2),([1,2,0,0,0,0],1),([1,2,0,0,0,0],None)):
            with self.assertRaises(DataError): verify_support(metrics,support,seen,2)


if __name__=='__main__': unittest.main()
