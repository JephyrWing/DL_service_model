"""Class order used by the CNN-MLP output."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit_bee_status as app


class ClassLabelTests(unittest.TestCase):
    def test_korean_labels_follow_model_class_order(self):
        self.assertEqual(app.CLASS_NAMES, (
            '기존 여왕벌 존재', '여왕벌 없음', '새 여왕벌 거부됨', '새 여왕벌 수용됨',
        ))


if __name__ == '__main__':
    unittest.main(verbosity=2)
