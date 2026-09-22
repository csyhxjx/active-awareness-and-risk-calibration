"""Stdlib-only evidence contracts, usable in both pinned environments."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

ROUTES = ('left_route', 'center_route', 'right_route')
LIMIT = 0.25


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite_tree(value):
    if isinstance(value, float):
        require(math.isfinite(value), 'non-finite evidence')
    elif isinstance(value, dict):
        for v in value.values():
            finite_tree(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            finite_tree(v)


def validate_export(root):
    root = Path(root).resolve()
    m = json.loads((root / 'export_manifest.json').read_text())
    finite_tree(m)
    require(m['source_mujoco_version'] == '2.3.7', 'source engine')
    require(file_hash(root / 'scene.xml') == m['xml_sha256'], 'XML hash')
    for name, sha in m['assets'].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root), 'asset path escape')
        require(file_hash(path) == sha, 'asset hash: ' + name)
    xml = ET.parse(root / 'scene.xml').getroot()
    for e in xml.iter():
        if e.get('file'):
            require(e.get('file') in m['assets'], 'unbound XML asset')
    require(set(m['nominals']) == set(ROUTES), 'nominal route set')
    for route, n in m['nominals'].items():
        require(n['route'] == route and n['reached'] is True, 'nominal route/reached')
        require(len(n['qpos']) == n['control_steps'] + 1 >= 2, 'nominal length')
        require(len(n['eef']) == len(n['qpos']), 'EEF length')
        require(digest(n['qpos']) == n['qpos_sha256'], 'qpos hash')
        base = {k: v for k, v in n.items() if k not in ('trajectory_sha256', 'qpos_sha256')}
        require(digest(base) == n['trajectory_sha256'], 'trajectory hash')
    return m


def expected_grid():
    return [(r, w, round(s * (.040 + .002 * i), 3))
            for r in ROUTES for w in (.025, .040, .055)
            for s in (-1, 1) for i in range(46)]


def stratum(c):
    if c <= 0:
        return 'blocked'
    if c - .004 >= .008:
        return 'safe'
    if abs(c - .004) <= .003:
        return 'boundary'
    return 'gap'


def decisions(c):
    require(c is not None and math.isfinite(c), 'exact clearance required')
    return {'penetrating': c < 0, 'feasible': c >= .004}


def refinement(low, high):
    return {'error_m': abs(low - high),
            'collision_equal': decisions(low)['penetrating'] == decisions(high)['penetrating'],
            'feasibility_equal': decisions(low)['feasible'] == decisions(high)['feasible']}


def validate_result(r, request, m, *, enhanced=False):
    finite_tree(r)
    for key in ('route', 'width_m', 'offset_m', 'intervals'):
        require(r[key] == request[key], 'result identity: ' + key)
    require(r['mujoco_version'] == '3.13.0' and r['native_ccd'] is True, 'backend provenance')
    steps = r['steps']
    require(len(steps) == len(request['qpos']), 'result step count')
    for i, s in enumerate(steps):
        require(s['step'] == i, 'step identity')
        require(0 <= s['sample'] <= request['intervals'], 'sample index')
        require(s['fraction'] == s['sample'] / request['intervals'], 'sample fraction')
        require(s['robot_geom'] in m['robot_geom_names'], 'robot geom identity')
        require(s['obstacle_geom'] in m['obstacle_geom_names'][r['route']], 'obstacle identity')
        require(s['distance_m'] <= LIMIT, 'distance exceeds query radius')
    minimum = min(steps, key=lambda s: s['distance_m'])
    require(r['minimum'] == minimum, 'minimum mismatch')
    censored = minimum['distance_m'] == LIMIT
    if enhanced:
        require(r['right_censored'] == censored, 'censor flag')
        require(r['qpos_sha256'] == digest(request['qpos']), 'result qpos hash')
        require(r['minimum_clearance_m'] == (None if censored else minimum['distance_m']), 'clearance mismatch')
        require(r['clearance_lower_bound_m'] == (LIMIT if censored else None), 'censor lower bound')
        for s in steps:
            if s['distance_m'] == LIMIT:
                require(s['fromto'] is None, 'censored segment')
    else:
        require(r['minimum_clearance_m'] == minimum['distance_m'], 'legacy clearance mismatch')


def request_for(record, m, intervals=8):
    return {k: record[k] for k in ('route', 'width_m', 'offset_m')} | {
        'qpos': m['nominals'][record['route']]['qpos'], 'intervals': intervals}


def validate_output(raw, requests, m, *, enhanced=True):
    rows = [json.loads(line) for line in raw.splitlines()]
    require(len(rows) == len(requests), 'worker output count/truncation')
    for r, q in zip(rows, requests):
        validate_result(r, q, m, enhanced=enhanced)
    return rows


def audit_grid(root, m):
    root = Path(root)
    expected = expected_grid()
    require({p.name for p in root.glob('case_*.json')} == {f'case_{i:03d}.json' for i in range(828)}, 'grid file set')
    records = []
    for i, (route, width, offset) in enumerate(expected):
        r = json.loads((root / f'case_{i:03d}.json').read_text())
        require(r['case_index'] == i, 'case index')
        require((r['route'], r['width_m'], r['offset_m']) == (route, width, offset), 'grid parameter identity')
        require(r['nominal_trajectory_sha256'] == m['nominals'][route]['trajectory_sha256'], 'grid trajectory hash')
        validate_result(r['result'], request_for(r, m), m)
        records.append(r)
    return records
