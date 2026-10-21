"""Offline checks for the paper's archive validation and calibration analysis."""

from pathlib import Path
import tempfile
import unittest

from scripts.reproduce_paper_figures import (
    calibration_summary, causal_confidence_pairs, load_and_validate,
)


class TestPaperCalibration(unittest.TestCase):
    def test_confidence_boundaries_are_counted_once(self):
        bins = calibration_summary([(0.0, 1), (0.1, 0), (0.5, 1), (1.0, 0)])
        self.assertEqual(sum(row['n'] for row in bins), 4)
        self.assertEqual([row['n'] for row in bins], [1, 1, 0, 0, 0, 1, 0, 0, 0, 1])
        self.assertEqual(bins[-1]['mean_confidence'], 1.0)
        self.assertTrue(bins[-1]['upper_inclusive'])

    def test_reports_use_only_solve_outputs_available_on_an_earlier_turn(self):
        report = {'query_count': 8, 'a': 0.8, 'b': 0.7, 'c': 0.6, 'f': 0.9}
        text = 'Confidence: a=80%, b=70%, c=60%, f=90%'
        run = {
            'metacognitive': {'confidence_reports': [report.copy() for _ in range(4)]},
            'turns': [{'turn': turn, 'role': 'assistant', 'content': text} for turn in (1, 2, 3, 5)],
            'solves': [
                {'turn': 2, 'solve_number': 1, 'queries_at_solve': 8,
                 'coeff_errors': {'a': 0.2, 'b': 0.05, 'c': 0.01}},
                {'turn': 4, 'solve_number': 2, 'queries_at_solve': 8,
                 'coeff_errors': {'a': 0.05, 'b': 0.05, 'c': 0.01}},
            ],
        }
        pairs, support = causal_confidence_pairs(run)
        self.assertEqual(support['reports_excluded_no_preceding_solve'], 2)
        self.assertEqual(support['reports_paired'], 2)
        self.assertEqual(pairs, [(0.8, 0), (0.7, 1), (0.6, 1),
                                 (0.8, 1), (0.7, 1), (0.6, 1)])

    def test_archive_report_values_must_match_transcript(self):
        run = {
            'metacognitive': {'confidence_reports': [
                {'query_count': 8, 'a': 0.8, 'b': 0.7, 'c': 0.6, 'f': 0.9}]},
            'turns': [{'turn': 1, 'role': 'assistant',
                       'content': 'Confidence: a=99%, b=70%, c=60%, f=90%'}],
            'solves': [],
        }
        with self.assertRaisesRegex(ValueError, 'disagree with the transcript'):
            causal_confidence_pairs(run)


class TestPinnedArchive(unittest.TestCase):
    def test_wrong_archive_fails_before_any_zip_member_is_read(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'untrusted.zip'
            archive.write_bytes(b'not the pinned archive')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                load_and_validate(archive, {'archive_sha256': '0' * 64})


if __name__ == '__main__':
    unittest.main()
