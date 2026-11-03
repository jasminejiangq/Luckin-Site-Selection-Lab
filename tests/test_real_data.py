"""Integration checks for the downloaded, cleaned Manhattan data bundle."""

import json
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import BASE, PROC, RAW
from evaluate import evaluate, greedy
import params


class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_manifest = json.loads((RAW / "source_manifest.json").read_text(encoding="utf-8"))
        cls.processed_manifest = json.loads(
            (PROC / "processed_manifest.json").read_text(encoding="utf-8")
        )
        cls.instance = params.build(PROC, BASE)

    def test_processed_row_counts_match_manifest(self):
        expected = self.processed_manifest["rows"]
        for name, count in expected.items():
            frame = pd.read_csv(PROC / f"{name}.csv")
            self.assertEqual(len(frame), count, name)
            self.assertFalse(frame.empty, name)

    def test_tract_and_store_ids_are_unique_and_coordinates_are_valid(self):
        bounds = (40.65, 40.90, -74.08, -73.85)
        for name, id_col in (
            ("tracts", "tract_id"),
            ("candidates", "cand_id"),
            ("luckin", "store_id"),
            ("competitors", "store_id"),
        ):
            frame = pd.read_csv(PROC / f"{name}.csv")
            self.assertTrue(frame[id_col].notna().all(), name)
            self.assertTrue(frame[id_col].is_unique, name)
            self.assertTrue(frame["lat"].between(bounds[0], bounds[1]).all(), name)
            self.assertTrue(frame["lon"].between(bounds[2], bounds[3]).all(), name)

    def test_known_missing_data_are_only_unmeasured_candidate_rents(self):
        for name in ("tracts", "luckin", "competitors", "walk_dist"):
            frame = pd.read_csv(PROC / f"{name}.csv")
            self.assertEqual(int(frame.isna().sum().sum()), 0, name)
        candidates = pd.read_csv(PROC / "candidates.csv")
        self.assertTrue(candidates["rent_psf_month"].isna().all())
        self.assertTrue(candidates["eligible"].isin([0, 1]).all())

    def test_population_jobs_and_monthly_subway_coverage(self):
        tracts = pd.read_csv(PROC / "tracts.csv")
        self.assertEqual(tracts["tract_id"].nunique(), len(tracts))
        self.assertTrue((tracts[["residents", "jobs", "transit"]] >= 0).all().all())
        subway = pd.read_csv(RAW / "subway_202608.csv")
        self.assertTrue((subway["observed_days"] == 31).all())
        self.assertEqual(len(subway), self.source_manifest["cleaned_row_counts"]["subway_station_complexes"])

    def test_no_openings_and_greedy_plan_are_feasible_and_reconcile(self):
        no_openings = evaluate(self.instance, [0] * self.instance.nJ)
        self.assertAlmostEqual(no_openings["Z"], 0.0, places=6)
        self.assertTrue(no_openings["feasible"], no_openings["checks"])
        plan = greedy(self.instance)
        result = evaluate(self.instance, plan)
        self.assertTrue(result["feasible"], result["checks"])
        self.assertAlmostEqual(
            result["gross_new_orders"],
            result["lost_incumbent_orders"]
            + result["from_competitors"]
            + result["from_no_purchase"],
            places=5,
        )

    def test_baseline_orders_follow_the_scenario_scale_assumption(self):
        self.assertAlmostEqual(
            (self.instance.w * self.instance.gamma).sum(),
            self.instance.nL * BASE.vbar,
            places=6,
        )

    def test_reported_model_dimensions_include_tracts_without_candidates(self):
        sizes = params.summary(self.instance)
        self.assertEqual(
            sizes["share vars (M+|I|)"],
            self.instance.M + self.instance.nI,
        )
        self.assertEqual(
            sizes["choice constraints (3M+|I|)"],
            3 * self.instance.M + self.instance.nI,
        )


if __name__ == "__main__":
    unittest.main()
