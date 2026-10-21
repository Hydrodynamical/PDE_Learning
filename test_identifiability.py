"""Proof and numeric sanity checks for the paper's identifiability audit."""
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from numpy.polynomial import Polynomial

from basis import CoefficientFunction, LegendreBasis
from pde import EllipticPDE, solve_elliptic
from scripts.audit_identifiability import (
    design_row, exact_rank, joint_diagnostics, load_inputs, polynomial_operator_matrix,
    spectra, structural_certificate,
)


class StructuralAmbiguityTests(unittest.TestCase):
    def test_exact_nullity_in_all_archived_classes_and_larger_classes(self):
        for n in range(2, 9):
            with self.subTest(n=n):
                matrix = polynomial_operator_matrix(n)
                self.assertEqual(exact_rank(matrix), n + 2)
                self.assertEqual(3 * n - exact_rank(matrix), 2 * n - 2)

    def test_integer_matrix_matches_independent_polynomial_differentiation(self):
        """Check the certificate against calculus, not a second rank routine."""
        u = Polynomial([0, 1, -1])
        for n in (2, 3, 4, 8):
            matrix = polynomial_operator_matrix(n)
            for block in range(3):
                for j in range(n):
                    p = Polynomial([0] * j + [1])
                    image = (-(p * u.deriv()).deriv() if block == 0 else
                             p * u.deriv() if block == 1 else p * u)
                    expected = np.pad(image.coef, (0, n + 2 - len(image.coef)))
                    np.testing.assert_array_equal(matrix[:, block * n + j], expected)

    def test_ellipticity_and_same_source_for_concrete_pair(self):
        # epsilon=-1/8 keeps a>0 and c>=0; both operators give f=2.
        x = np.linspace(0, 1, 1001)
        epsilon = -.125
        a = 1 + epsilon
        b, c = epsilon * (4*x - 2), -8 * epsilon
        self.assertGreater(a, 0)
        self.assertGreaterEqual(c, 0)
        np.testing.assert_allclose(2*a + b*(1 - 2*x) + c*x*(1-x), 2,
                                   rtol=0, atol=1e-15)
        certificate = structural_certificate()
        self.assertEqual([r["exact_nullity"] for r in certificate["certificates"][:3]], [2, 4, 6])


class NumericalAuditTests(unittest.TestCase):
    def test_unpinned_archive_is_rejected_before_computation(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "invalid.zip"
            archive.write_bytes(b"unverified input")
            with self.assertRaisesRegex(ValueError, "Archive SHA-256"):
                load_inputs(archive, {"archive_sha256": "0" * 64})

    def test_optional_extracted_copy_must_match_pinned_member_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "sample.zip"
            with zipfile.ZipFile(archive, "w") as source:
                source.writestr("run_example.json", b'{"example":true}\n')
            extracted = root / "runs"
            extracted.mkdir()
            (extracted / "run_example.json").write_bytes(b'{"example":true}\n')
            with patch("scripts.audit_identifiability.load_and_validate", return_value=[{"example": True}]):
                runs, hashes = load_inputs(archive, {}, extracted)
                self.assertEqual(runs, [("run_example.json", {"example": True})])
                self.assertEqual(len(hashes["run_example.json"]), 64)
                (extracted / "run_example.json").write_bytes(b'{"example":false}\n')
                with self.assertRaisesRegex(ValueError, "names/bytes"):
                    load_inputs(archive, {}, extracted)

    def test_joint_rank_exposes_dependency_missed_by_separate_blocks(self):
        full_rank_block = np.array([[1., 0.], [0., 1.], [1., 1.]])
        joint = np.hstack([full_rank_block] * 3)
        self.assertEqual(np.linalg.matrix_rank(full_rank_block), 2)
        summary = joint_diagnostics(joint, np.ones(3), np.zeros(6))
        self.assertEqual(summary["raw"]["rank_at_numpy_default_tolerance"], 2)
        self.assertEqual(len(summary["raw"]["singular_values"]), 6)

    def test_column_normalization_does_not_claim_exact_structural_rank(self):
        result = spectra(np.diag([1., 1e-11]))
        self.assertEqual(result["rank_at_numpy_default_tolerance"], 2)
        self.assertEqual(result["rank_at_relative_tolerance"]["1e-08"], 1)

    def test_reconstructed_row_matches_analytic_forcing_integral(self):
        basis = LegendreBasis(2)
        coefficient = lambda values: CoefficientFunction(basis, values)
        pde = EllipticPDE(a=coefficient([1, 0]), b=coefficient([0, 0]),
                          c=coefficient([0, 0]), f=coefficient([2, 0]))
        solution = solve_elliptic(pde, n_grid=101)
        matrix, source, rhs = design_row(solution, basis, "x*(1-x)")
        self.assertAlmostEqual(rhs, 1/3, places=13)
        self.assertAlmostEqual(source[0], 1/6, places=13)
        # The expression parser differentiates numerically at step 1e-5.
        self.assertAlmostEqual(matrix[0], rhs, delta=1e-10)


if __name__ == "__main__":
    unittest.main()
