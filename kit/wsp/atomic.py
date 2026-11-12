"""Atomic JSON writes for state files (edit.json, report.json,
versions.json, latest.json, .watch-seen.json, merge.json).

The new content goes to a temp file in the same directory (so the
rename stays on one filesystem), is flushed and fsynced, then moved
into place with ``os.replace``. A crash mid-write leaves the
previous file intact and no temp file behind. Stdlib only.
"""

import json
import os
import tempfile


def atomic_write_json(path, obj, indent=2):
    """Write ``obj`` as JSON to ``path`` atomically.

    Raises whatever ``json.dump`` raises, after removing the temp
    file — the previous file is untouched in that case.
    """
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(
        dir=directory, prefix="." + os.path.basename(path) + ".tmp-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
