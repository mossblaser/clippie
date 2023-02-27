import subprocess

from pathlib import Path

import clippie


def test_mypy():
    assert subprocess.run(
        [
            "mypy",
            str(Path(clippie.__file__).parent),
            str(Path(__file__).parent),
        ],
    )
