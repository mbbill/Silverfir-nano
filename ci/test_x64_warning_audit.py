from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class SavedCompilerWarningAuditTests(unittest.TestCase):
    def test_saved_logs_fail_closed_on_diagnostics_or_missing_evidence(self) -> None:
        for content, passes in [
            ('    Finished `release` profile\n', True),
            ('warning: unused import: `Example`\n', False),
            ('error[E0308]: mismatched types\n', False),
            ('', False),
            (None, False),
        ]:
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                log = Path(directory) / 'compiler.log'
                if content is not None:
                    log.write_text(content)
                result = subprocess.run(
                    [sys.executable, '-m', 'ci.x64_warning_audit', str(log)],
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode == 0, passes, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
