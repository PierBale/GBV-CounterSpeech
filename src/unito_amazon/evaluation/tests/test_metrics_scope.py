"""Metric entry-point filtering checks; all metric computation is mocked."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import metrics


class MetricScopeTests(unittest.TestCase):
    def test_only_multiconan_with_reference_reaches_computation(self):
        records = [
            {"id": "edos", "source": "edos", "reference_counter_speech": ""},
            {"id": "missing", "source": "multiconan", "reference_counter_speech": None},
            {"id": "valid", "source": "multiconan", "reference_counter_speech": "Gold"},
        ]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(metrics, "read_jsonl", return_value=records),
            patch.object(metrics, "AUTOMATIC_METRICS_OUTPUT_DIR", Path(directory)),
            patch.object(metrics, "compute_automatic_metrics", return_value=[{**records[2], "automatic_metrics": {}}]) as compute,
            patch.object(metrics, "summarize_automatic_metrics", return_value={}),
        ):
            metrics.evaluate_file(Path("synthetic.jsonl"))
        self.assertEqual(compute.call_args.args[0], [records[2]])

    def test_edos_only_never_loads_metric_models(self):
        with (
            patch.object(metrics, "read_jsonl", return_value=[{"source": "edos"}]),
            patch.object(metrics, "compute_automatic_metrics") as compute,
        ):
            metrics.evaluate_file(Path("synthetic.jsonl"))
        compute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
