"""Verify task names without importing rendering or training dependencies."""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent
EXPECTED = {
    "self_where": "landmark-relative position reasoning",
    "env_where": "self-relative position reasoning",
}


def display_mapping(path, name):
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(node.value)
    raise AssertionError("Task display mapping not found: " + str(path))


class TaskTitleTests(unittest.TestCase):
    def test_export_and_training_names_match_the_reference_frame(self):
        for relative, variable in (
            ("export_gauu_real_world_web.py", "TASK_TITLES"),
            ("export_gauu_real_world_results.py", "TASK_TITLES"),
            ("uav_dualcog_optimize/common.py", "TASK_TYPE_DISPLAY"),
        ):
            mapping = display_mapping(ROOT / relative, variable)
            for task_id, title in EXPECTED.items():
                with self.subTest(module=relative, task=task_id):
                    self.assertEqual(mapping[task_id].lower(), title)


if __name__ == "__main__":
    unittest.main()
