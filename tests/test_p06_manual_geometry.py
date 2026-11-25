import unittest
from tools.p06_analyse_manual_geometry import coverage, statistics


class ManualGeometryTests(unittest.TestCase):
    def test_independent_dimensions_and_partial_review_counts(self):
        rows=[]
        for cid in (1,2,3,5):
            a,b=('underhead_length','thread_major_diameter') if cid!=5 else ('outer_diameter','inner_diameter')
            for dimensions,partial in (({a:{}},{}),({b:{}},{}),({a:{},b:{}},{}),({}, {b:{}})):
                rows.append(dict(class_id=cid,before=dict(dimensions=dimensions,diagnostics={'partial_dimensions':partial})))
        for counts in coverage(rows).values():
            self.assertEqual(counts['total'],4)
            result=counts['before']
            self.assertEqual(result['both'],1)
            self.assertEqual(result['partial_review_only'],1)
            self.assertEqual(sorted(v for k,v in result.items() if k not in ('both','partial_review_only')),[2,2])

    def test_signed_bias_and_absolute_error_are_distinct(self):
        result=statistics([-2,2])
        self.assertEqual(result['bias'],0)
        self.assertEqual(result['mean_absolute'],2)
        self.assertEqual(statistics([]),{'n':0})
