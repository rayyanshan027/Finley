from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from finley.config import DatasetConfig
from finley.data.session import build_session_paths, list_available_sessions, resolve_animal_root


class HC6SessionPathTests(unittest.TestCase):
    def test_resolve_animal_root_from_extracted_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "extracted"
            bon = root / "Bon"
            bon.mkdir(parents=True)

            config = DatasetConfig(root=root, animals=["Bon"], allowed_extensions=[], ignore_hidden=True)
            resolved = resolve_animal_root(config, "Bon")

            self.assertEqual(resolved, bon)

    def test_build_session_paths_for_direct_animal_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bon = Path(temp_dir) / "Bon"
            bon.mkdir(parents=True)

            config = DatasetConfig(root=bon, animals=["Bon"], allowed_extensions=[], ignore_hidden=True)
            paths = build_session_paths(config, "Bon", 3)

            self.assertEqual(paths.spikes_path, bon / "bonspikes03.mat")
            self.assertEqual(paths.pos_path, bon / "bonpos03.mat")
            self.assertEqual(paths.task_path, bon / "bontask03.mat")
            self.assertEqual(paths.rawpos_path, bon / "bonrawpos03.mat")

    def test_build_session_paths_matches_case_insensitive_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "extracted"
            cor = root / "Cor"
            cor.mkdir(parents=True)
            (cor / "Corspikes03.mat").write_text("", encoding="utf-8")
            (cor / "Corpos03.mat").write_text("", encoding="utf-8")
            (cor / "Cortask03.mat").write_text("", encoding="utf-8")
            (cor / "Corrawpos03.mat").write_text("", encoding="utf-8")

            config = DatasetConfig(root=root, animals=["Cor"], allowed_extensions=[], ignore_hidden=True)
            paths = build_session_paths(config, "Cor", 3)

            self.assertEqual(paths.spikes_path, cor / "Corspikes03.mat")
            self.assertEqual(paths.pos_path, cor / "Corpos03.mat")
            self.assertEqual(paths.task_path, cor / "Cortask03.mat")
            self.assertEqual(paths.rawpos_path, cor / "Corrawpos03.mat")

    def test_list_available_sessions_from_spike_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "extracted"
            bon = root / "Bon"
            bon.mkdir(parents=True)
            (bon / "bonspikes03.mat").write_text("", encoding="utf-8")
            (bon / "bonspikes10.mat").write_text("", encoding="utf-8")
            (bon / "bonpos03.mat").write_text("", encoding="utf-8")

            config = DatasetConfig(root=root, animals=["Bon"], allowed_extensions=[], ignore_hidden=True)
            self.assertEqual(list_available_sessions(config, "Bon"), [3, 10])


if __name__ == "__main__":
    unittest.main()
