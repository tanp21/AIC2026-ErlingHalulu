"""The video-id format every route, ZIP parser and label document has to agree on.

``<group>_V<nnn>`` -- ``L21_V001``, ``S01_V012``, ``N029_V002``. Batch 1 groups carry two
digits and the batch 2 traffic-camera groups three, so the group number is ``\\d{2,3}``; the
separator on disk is always an underscore, and a hyphen is accepted too because the
organisers' own detector exports and map-keyframes write it that way (``dres_payload``
normalises those ids before they reach a path).

One definition rather than one per module: the copies that said ``\\d{2}`` are what made every
``/thumb``, ``/video``, ``/shots`` and ``/frame-map`` request for an N video answer 422, which
is a broken keyframe, a player that never opens, and a candidate the checker refuses -- with
the index itself perfectly fine.
"""
from __future__ import annotations

import re

VIDEO_ID_PATTERN = r"^[A-Z]\d{2,3}[_-]V\d{3}$"
VIDEO_ID = re.compile(VIDEO_ID_PATTERN)
