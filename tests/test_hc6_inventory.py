from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from finley.config import DatasetConfig
from finley.data.hc6 import scan_dataset, summarize_inventory


class HC6InventoryTests(unittest.TestCase):
    def test_scan_dataset_filters_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "extracted"
            bon = root / "Bon"
            bon.mkdir(parents=True)
            (bon / "session1.res").write_text("a", encoding="utf-8")
            (bon / "session1.clu").write_text("b", encoding="utf-8")

            config = DatasetConfig(
                root=root,
                animals=["Bon"],
                allowed_extensions=[".res"],
                ignore_hidden=True,
            )

            records = scan_dataset(config)

            self.assertEqual([record["relative_path"] for record in records], ["Bon/session1.res"])
            self.assertEqual([record["extension"] for record in records], [".res"])
            self.assertEqual([record["modality"] for record in records], ["unknown"])

    def test_summarize_inventory_handles_empty_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "extracted"
            root.mkdir(parents=True)
            config = DatasetConfig(root=root, animals=[], allowed_extensions=[], ignore_hidden=True)

            records = scan_dataset(config)
            summary = summarize_inventory(records)

            self.assertEqual(summary["row_count"], 0)
            self.assertEqual(summary["animals"], [])

    def test_scan_dataset_accepts_direct_animal_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "Bon"
            root.mkdir(parents=True)
            (root / "session1.res").write_text("a", encoding="utf-8")

            config = DatasetConfig(root=root, animals=["Bon"], allowed_extensions=[], ignore_hidden=True)
            records = scan_dataset(config)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["animal"], "Bon")
            self.assertEqual(records[0]["relative_path"], "session1.res")

    def test_scan_dataset_extracts_hc6_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "Bon"
            eeg = root / "EEG"
            eeg.mkdir(parents=True)
            (root / "bonspikes03.mat").write_text("a", encoding="utf-8")
            (root / "boncellinfo.mat").write_text("b", encoding="utf-8")
            (eeg / "boneeg03-1-02.mat").write_text("c", encoding="utf-8")

            config = DatasetConfig(root=root, animals=["Bon"], allowed_extensions=[".mat"], ignore_hidden=True)
            records = scan_dataset(config)

            by_path = {record["relative_path"]: record for record in records}
            self.assertEqual(by_path["bonspikes03.mat"]["modality"], "spikes")
            self.assertEqual(by_path["bonspikes03.mat"]["session"], 3)
            self.assertEqual(by_path["boncellinfo.mat"]["modality"], "cellinfo")
            self.assertIsNone(by_path["boncellinfo.mat"]["session"])
            self.assertEqual(by_path["EEG/boneeg03-1-02.mat"]["modality"], "eeg")
            self.assertEqual(by_path["EEG/boneeg03-1-02.mat"]["session"], 3)


if __name__ == "__main__":
    unittest.main()
