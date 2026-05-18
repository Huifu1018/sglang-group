"""Re-apply sglang-group legacy patches inside spawned SGLang children."""

from __future__ import annotations

import os


if os.getenv("SGLANG_GROUP_LEGACY_NGRAM_PATCH") == "1":
    from sglang_group.sglang.compat import patch_legacy_ngram_worker

    patch_legacy_ngram_worker()
