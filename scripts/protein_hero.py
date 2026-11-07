#!/usr/bin/env python3
"""Animated GitHub profile hero: a DNA helix translating into a protein of the day.

Picks a structure from the PDB (seeded by date), flattens its 3D fold into a
2D cartoon (helix ribbons, strand arrows, loop tubes) and writes an animated SVG.

    python scripts/protein_hero.py                          # today's pick
    python scripts/protein_hero.py --date 2026-01-01
    python scripts/protein_hero.py --pdb 1UBQ
    python scripts/protein_hero.py --out assets/hero.svg --readme README.md   # what the Action runs
"""
import argparse
import datetime as dt
import hashlib
import html
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import biotite.database.rcsb as rcsb
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx

# ---- profile text -----------------------------------------------------------
NAME = "Sudhish"
SUBTITLE = "Biotech @ IITH · comp bio · AI for molecules"
TAGLINE = "ATCG → AUG → fold"

# ---- canvas -----------------------------------------------------------------
W, H = 1200, 380
BOX = (600, 52, 1160, 300)  # protein area: x0, y0, x1, y1
BG = "#0d1117"
PALETTE = ["#22d3ee", "#4ade80", "#a78bfa", "#f472b6"]  # N -> C
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace"

# ---- cartoon geometry (angstroms) ---------------------------------------------
K = 5            # spline samples per residue
STRIP = 2        # spline sub-segments per painted strip
HW_HELIX = 1.3   # ribbon half-widths
HW_STRAND = 1.05
HW_ARROW = 1.9
HW_COIL = 0.32
T_MIN = 0.22     # minimum half-thickness so edge-on ribbons stay visible

# ---- animation ----------------------------------------------------------------
GROW = 4.0       # seconds for the chain to appear N -> C
BUCKETS = 40

# ---- selection filters ----------------------------------------------------------
MIN_RES, MAX_RES = 80, 320
MAX_RESOLUTION = 2.2
MIN_SSE_FRACTION = 0.45


# =============================================================================
# Picking a structure
# =============================================================================
def build_query():
    Q = rcsb.FieldQuery
    return rcsb.CompositeQuery([
        Q("rcsb_entry_info.polymer_entity_count_protein", equals=1),
        Q("rcsb_entry_info.polymer_entity_count_nucleic_acid", equals=0),
        Q("rcsb_entry_info.deposited_polymer_entity_instance_count", equals=1),
        Q("rcsb_entry_info.deposited_polymer_monomer_count", range_closed=(MIN_RES, MAX_RES)),
        Q("exptl.method", exact_match="X-RAY DIFFRACTION"),
        Q("rcsb_entry_info.resolution_combined", less_or_equal=MAX_RESOLUTION),
    ], "and")


def candidates(date, n=12):
    """Yield PDB IDs deterministically from the date."""
    query = build_query()
    total = rcsb.count(query)
    seed = int(hashlib.sha256(date.isoformat().encode()).hexdigest(), 16)
    for k in range(n):
        start = (seed + k * 7919) % total
        ids = rcsb.search(query, range=(start, start + 1),
                          sort_by="rcsb_accession_info.initial_release_date")
        if ids:
            yield ids[0]


def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "github-profile-hero"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


ACRONYMS = {"DNA", "RNA", "ATP", "ADP", "GTP", "GDP", "NAD", "NADH", "NADP", "NADPH",
            "FAD", "FMN", "COA", "II", "III", "IV", "VI", "VII", "VIII", "IX", "HIV",
            "SARS", "MHC", "TRNA", "MRNA", "RRNA", "PDZ", "SH2", "SH3", "UDP", "AMP"}


def tidy_name(name):
    if not name.isupper():
        first, _, rest = name.partition(" ")
        return (first.capitalize() if first.islower() else first) + (" " + rest if rest else "")
    words = [w if w.strip("(),-") in ACRONYMS or any(c.isdigit() for c in w) else w.lower()
             for w in name.split()]
    out = " ".join(words)
    return out[:1].upper() + out[1:]


def metadata(pdb_id):
    entry = fetch_json(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}")
    entity_id = entry["rcsb_entry_container_identifiers"]["polymer_entity_ids"][0]
    ent = fetch_json(f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}")
    orgs = ent.get("rcsb_entity_source_organism") or [{}]
    method = entry["exptl"][0]["method"]
    res = (entry["rcsb_entry_info"].get("resolution_combined") or [None])[0]
    return {
        "id": pdb_id.upper(),
        "name": tidy_name(ent["rcsb_polymer_entity"].get("pdbx_description") or entry["struct"]["title"]),
        "organism": orgs[0].get("scientific_name") or "unknown source",
        "method": "X-ray" if method == "X-RAY DIFFRACTION" else method.title(),
        "resolution": res,
        "title": entry["struct"]["title"],
        "url": f"https://www.rcsb.org/structure/{pdb_id.upper()}",
    }


# =============================================================================
# Structure -> per-residue backbone data
# =============================================================================
# biotite < 2 returns "a"/"b"/"c"; biotite >= 2 returns 1 (helix) / 2 (strand) / 0
SSE_CODES = {"a": "a", "b": "b", 1: "a", 2: "b"}


def load_residues(pdb_id):
    f = pdbx.BinaryCIFFile.read(rcsb.fetch(pdb_id, "bcif"))
    atoms = pdbx.get_structure(f, model=1)
    atoms = atoms[struc.filter_amino_acids(atoms) & ~atoms.hetero]
    chain = atoms[atoms.chain_id == atoms.chain_id[0]]
    sse = struc.annotate_sse(chain)
    starts = struc.get_residue_starts(chain)

    ca, co, ss = [], [], []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(chain)
        names = list(chain.atom_name[s:e])
        if "CA" not in names:
            continue
        coords = chain.coord[s:e]
        ca.append(coords[names.index("CA")])
        if "C" in names and "O" in names:
            co.append(coords[names.index("O")] - coords[names.index("C")])
        else:
            co.append(np.full(3, np.nan))
        ss.append(SSE_CODES.get(sse[k], "c"))
    return np.array(ca, float), np.array(co, float), clean_sse(ss)


def clean_sse(ss):
    """Drop helices shorter than 4 and strands shorter than 3 residues."""
    ss = list(ss)
    i = 0
    while i < len(ss):
        j = i
        while j < len(ss) and ss[j] == ss[i]:
            j += 1
        if (ss[i] == "a" and j - i < 4) or (ss[i] == "b" and j - i < 3):
            ss[i:j] = ["c"] * (j - i)
        i = j
    return ss


def fragments(ca):
    """Split the chain where residues are missing (CA-CA gap)."""
    breaks = np.where(np.linalg.norm(np.diff(ca, axis=0), axis=1) > 4.3)[0]
    bounds = [0, *(breaks + 1), len(ca)]
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b - a >= 2]


def orient(ca, co):
    """Rotate so the widest view faces the camera (PCA), keeping handedness."""
    center = ca.mean(axis=0)
    _, _, vt = np.linalg.svd(ca - center, full_matrices=False)
    if np.linalg.det(vt) < 0:
        vt[2] *= -1
    if (ca[0] - center) @ vt[0] > 0:  # spin 180° about y so the N-terminus faces the DNA
        vt[[0, 2]] *= -1
    return (ca - center) @ vt.T, co @ vt.T


# =============================================================================
# Spline sampling
# =============================================================================
def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)


def catmull(p0, p1, p2, p3, t):
    t2, t3 = t * t, t * t * t
    pos = 0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
    tan = 0.5 * ((p2 - p0) + 2 * (2 * p0 - 5 * p1 + 4 * p2 - p3) * t
                 + 3 * (-p0 + 3 * p1 - 3 * p2 + p3) * t2)
    return pos, tan


def residue_halfwidth(ss, i, n):
    """Half-width and ribbon-ness (1 = flat ribbon, 0 = round tube) at residue i."""
    if ss[i] == "a":
        return HW_HELIX, 1.0
    if ss[i] == "b":
        return HW_STRAND, 1.0
    return HW_COIL, 0.0


def strand_end(ss, i):
    """Index of the last residue of the strand containing i."""
    while i + 1 < len(ss) and ss[i + 1] == "b":
        i += 1
    return i


def sample(ca, co, ss):
    n = len(ca)
    # strands: smooth out the pleat
    ctrl = ca.copy()
    for i in range(1, n - 1):
        if ss[i - 1] == ss[i] == ss[i + 1] == "b":
            ctrl[i] = (ca[i - 1] + 2 * ca[i] + ca[i + 1]) / 4

    # ribbon width direction: carbonyl vector, sign-smoothed along the chain
    curv = np.zeros_like(ca)
    curv[1:-1] = ca[1:-1] - (ca[:-2] + ca[2:]) / 2  # points away from helix axis
    wdir = np.where(np.isnan(co), curv, co)
    wdir = unit(wdir)
    for i in range(1, n):
        if np.dot(wdir[i], wdir[i - 1]) < 0:
            wdir[i] = -wdir[i]
    radial = unit(curv)

    frags = []
    for a, b in fragments(ca):
        pts = []
        idx = list(range(a, b))
        for j, i in enumerate(idx[:-1]):
            p0 = ctrl[idx[max(j - 1, 0)]]
            p1, p2 = ctrl[i], ctrl[i + 1]
            p3 = ctrl[idx[min(j + 2, len(idx) - 1)]]
            last = j == len(idx) - 2
            for s in range(K + (1 if last else 0)):
                t = s / K
                pos, tan = catmull(p0, p1, p2, p3, t)
                resf = i + t
                hw0, rib0 = residue_halfwidth(ss, i, n)
                hw1, rib1 = residue_halfwidth(ss, i + 1, n)
                hw, rib = hw0 + (hw1 - hw0) * t, rib0 + (rib1 - rib0) * t
                if ss[i] == "b" and ss[i + 1] == "b":
                    end = strand_end(ss, i)
                    if resf >= end - 1:  # arrow head spans the last residue
                        hw = HW_ARROW * (end - resf)
                elif ss[i] == "b":  # leaving a strand: tube grows out of the arrow tip
                    hw, rib = HW_COIL * t, 0.0
                w = wdir[i] * (1 - t) + wdir[i + 1] * t
                tan_u = unit(tan)
                w = unit(w - np.dot(w, tan_u) * tan_u)
                r = radial[i] * (1 - t) + radial[i + 1] * t
                pts.append(dict(pos=pos, tan=tan_u, w=w, radial=r, hw=hw, rib=rib,
                                resf=resf, ss=ss[i] if t < 0.5 else ss[i + 1]))
        frags.append(pts)
    return frags


# =============================================================================
# Colour
# =============================================================================
def hex_rgb(h):
    return np.array([int(h[k:k + 2], 16) for k in (1, 3, 5)], float) / 255


def rgb_hex(c):
    c = np.clip(c, 0, 1)
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in c)


PAL = [hex_rgb(h) for h in PALETTE]
BG_RGB = hex_rgb(BG)


def gradient(f):
    f = min(max(f, 0.0), 1.0) * (len(PAL) - 1)
    k = min(int(f), len(PAL) - 2)
    return PAL[k] + (PAL[k + 1] - PAL[k]) * (f - k)


def shade(base, light, fog):
    c = base * light if light <= 1 else base + (1 - base) * (light - 1)
    return c + (BG_RGB - c) * fog


# =============================================================================
# SVG
# =============================================================================
def fmt(p):
    return f"{p[0]:.1f},{p[1]:.1f}"


def cartoon(frags, n_res):
    allpos = np.array([p["pos"] for fr in frags for p in fr])
    lo, hi = allpos.min(axis=0) - 2, allpos.max(axis=0) + 2
    x0, y0, x1, y1 = BOX
    scale = min((x1 - x0) / (hi[0] - lo[0]), (y1 - y0) / (hi[1] - lo[1]))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    mid = (lo + hi) / 2
    zlo, zhi = lo[2], hi[2]

    def to2d(v):
        return np.array([cx + (v[0] - mid[0]) * scale, cy - (v[1] - mid[1]) * scale])

    strips, backbone, lengths = [], [], []
    for fr in frags:
        left, right, centers = [], [], []
        for p in fr:
            c2 = to2d(p["pos"])
            t2 = np.array([p["tan"][0], -p["tan"][1]])
            t2 = t2 / (np.linalg.norm(t2) or 1)
            perp = np.array([-t2[1], t2[0]])
            w2 = np.array([p["w"][0], -p["w"][1]]) * p["hw"] * scale
            off = p["rib"] * w2 + (1 - p["rib"]) * p["hw"] * scale * perp
            along = np.dot(off, perp)
            tmin = T_MIN * scale
            if abs(along) < tmin:
                off = off + ((tmin if along >= 0 else -tmin) - along) * perp
            left.append(c2 + off)
            right.append(c2 - off)
            centers.append(c2)
        backbone.append(centers)

        for j in range(0, len(fr) - 1, STRIP):
            k = min(j + STRIP, len(fr) - 1)
            m = fr[(j + k) // 2]
            z = np.mean([fr[q]["pos"][2] for q in range(j, k + 1)])
            normal = unit(np.cross(m["tan"], m["w"]))
            if m["ss"] == "a":
                if np.dot(normal, m["radial"]) < 0:
                    normal = -normal
                nz = normal[2]
                light_ss = 0.6 + 0.55 * nz if nz >= 0 else 0.5 + 0.2 * -nz
            else:
                light_ss = 0.55 + 0.6 * abs(normal[2])
            light = m["rib"] * light_ss + (1 - m["rib"]) * 0.95
            fog = (1 - (z - zlo) / ((zhi - zlo) or 1)) * 0.5
            frac = m["resf"] / max(n_res - 1, 1)
            color = rgb_hex(shade(gradient(frac), light, fog))
            ring = left[j:k + 1] + right[j:k + 1][::-1]
            d_fill = "M" + fmt(ring[0]) + "L" + " ".join(fmt(q) for q in ring[1:]) + "Z"
            d_edge = ("M" + fmt(left[j]) + "L" + " ".join(fmt(q) for q in left[j + 1:k + 1])
                      + "M" + fmt(right[j]) + "L" + " ".join(fmt(q) for q in right[j + 1:k + 1]))
            bucket = min(BUCKETS - 1, int(frac * BUCKETS))
            strips.append((z, bucket, color, d_fill, d_edge))

    strips.sort(key=lambda s: s[0])
    body = "".join(
        f'<path class="s d{b}" fill="{c}" stroke="{c}" stroke-width=".6" d="{df}"/>'
        f'<path class="s d{b} e" d="{de}"/>'
        for _, b, c, df, de in strips)

    # 2D backbone trace for the glow and the travelling "ribosome" dot
    d_trace, resf_at, cum = [], [], []
    for fr, centers in zip(frags, backbone):
        d_trace.append("M" + fmt(centers[0]) + "L" + " ".join(fmt(q) for q in centers[1:]))
        for q, (p, c) in enumerate(zip(fr, centers)):
            step = float(np.linalg.norm(c - centers[q - 1])) if q else 0.0  # gaps add no length
            cum.append((cum[-1] if cum else 0.0) + step)
            resf_at.append(p["resf"])
    cum = np.array(cum)
    total = cum[-1] or 1
    resf_at = np.array(resf_at)
    key_t, key_p = [], []
    for f in np.linspace(0, 1, 21):
        q = int(np.argmin(np.abs(resf_at - f * (n_res - 1))))
        key_t.append(f)
        key_p.append(max(cum[q] / total, key_p[-1] if key_p else 0))
    key_p[-1] = 1.0
    left = min(q[0] for centers in backbone for q in centers)
    dock = (left - 18, backbone[0][0][1])  # where codons arrive: protein's left edge, N-term height
    return body, "".join(d_trace), key_t, key_p, dock


def dna_helix(x0, cy, rungs=17, gap=24, amp=46, period=3.2):
    ease_in, ease_out = "0.12 0 0.39 0", "0.61 1 0.88 1"
    ys = lambda a, b: f"{cy + a};{cy};{cy + b};{cy};{cy + a}"
    y_spl = ";".join([ease_in, ease_out, ease_in, ease_out])
    r_spl = ";".join([ease_out, ease_in, ease_out, ease_in])

    def anim(attr, values, splines, begin):
        return (f'<animate attributeName="{attr}" values="{values}" keyTimes="0;.25;.5;.75;1" '
                f'calcMode="spline" keySplines="{splines}" dur="{period}s" begin="{begin:.2f}s" '
                f'repeatCount="indefinite"/>')

    out = []
    for i in range(rungs):
        x, b = x0 + i * gap, -i * period / 16
        out.append(
            f'<line x1="{x}" x2="{x}" stroke="#a78bfa" stroke-opacity=".45" stroke-width="2">'
            f'{anim("y1", ys(-amp, amp), y_spl, b)}{anim("y2", ys(amp, -amp), y_spl, b)}</line>'
            f'<circle cx="{x}" fill="#22d3ee">{anim("cy", ys(-amp, amp), y_spl, b)}'
            f'{anim("r", "4.5;6;4.5;3;4.5", r_spl, b)}{anim("opacity", ".85;1;.85;.45;.85", r_spl, b)}</circle>'
            f'<circle cx="{x}" fill="#4ade80">{anim("cy", ys(amp, -amp), y_spl, b)}'
            f'{anim("r", "4.5;3;4.5;6;4.5", r_spl, b)}{anim("opacity", ".85;.45;.85;1;.85", r_spl, b)}</circle>')
    return "".join(out), x0 + (rungs - 1) * gap


def codon_stream(x_from, cy, target):
    tx, ty = target
    path = f"M{x_from},{cy} C{x_from + 70},{cy} {tx - 90:.1f},{ty:.1f} {tx:.1f},{ty:.1f}"
    codons = ["AUG", "GCU", "UUC", "CGA", "AAG", "UGG", "CAC", "GAU"]
    dur = 4.0
    items = "".join(
        f'<text font-size="12" fill="#c4b5fd" text-anchor="middle" dy="4" opacity="0">{c}'
        f'<animateMotion path="{path}" dur="{dur}s" begin="{-k * dur / len(codons):.2f}s" repeatCount="indefinite"/>'
        f'<animate attributeName="opacity" values="0;.85;.85;0" keyTimes="0;.15;.8;1" dur="{dur}s" '
        f'begin="{-k * dur / len(codons):.2f}s" repeatCount="indefinite"/></text>'
        for k, c in enumerate(codons))
    return (f'<path d="{path}" fill="none" stroke="#a78bfa" stroke-opacity=".18" stroke-width="1.5" '
            f'stroke-dasharray="3 5"/>{items}')


def typed(text, x, y, begin, size=13, fill="#c9d1d9", char_w=None):
    char_w = char_w or size * 0.61
    n = len(text)
    widths = ";".join(f"{k * char_w:.1f}" for k in range(n + 1))
    times = ";".join(f"{k / (n + 1):.3f}" for k in range(n + 1))
    cid = "caption-clip"
    return (f'<clipPath id="{cid}"><rect x="{x}" y="{y - size - 2}" height="{size + 8}" width="0">'
            f'<animate attributeName="width" values="{widths}" keyTimes="{times}" calcMode="discrete" '
            f'dur="{n * 0.035:.2f}s" begin="{begin}s" fill="freeze"/></rect></clipPath>'
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" clip-path="url(#{cid})">{esc(text)}</text>')


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render(meta, frags, n_res, date):
    body, trace, key_t, key_p, dock = cartoon(frags, n_res)
    helix, helix_end = dna_helix(56, 236)
    delays = "".join(f".d{b}{{animation-delay:{b * GROW / BUCKETS:.2f}s}}" for b in range(BUCKETS))
    kt = ";".join(f"{t:.3f}" for t in key_t)
    kp = ";".join(f"{p:.4f}" for p in key_p)

    res = f" {meta['resolution']:.1f} Å" if meta["resolution"] else ""
    head = f"PDB {meta['id']} · "
    tail = f" · {meta['organism']} · {meta['method']}{res}"
    room = 64 - len(head) - len(tail)
    name = meta["name"] if len(meta["name"]) <= room else meta["name"][:room - 1].rstrip() + "…"
    caption = head + name + tail

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t d">
<title id="t">{esc(NAME)}: DNA to protein</title>
<desc id="d">A turning DNA double helix sends codons to today's protein from the PDB, {esc(meta['id'])} ({esc(meta['name'])}), which draws itself from N to C terminus.</desc>
<style>
.s{{opacity:0;animation:g .45s ease-out forwards}}
@keyframes g{{to{{opacity:1}}}}
{delays}
.e{{fill:none;stroke:#05080d;stroke-opacity:.8;stroke-width:1.1;stroke-linejoin:round;stroke-linecap:round}}
.glow{{opacity:0;animation:g 1.5s ease-out {GROW:.1f}s forwards}}
.bob{{animation:b 7s ease-in-out {GROW:.1f}s infinite}}
@keyframes b{{50%{{transform:translateY(-5px)}}}}
.blink{{animation:k 1s steps(1) infinite}}
@keyframes k{{50%{{opacity:0}}}}
@media (prefers-reduced-motion:reduce){{.s,.glow{{animation:none;opacity:1}}.bob,.blink{{animation:none}}}}
</style>
<defs><pattern id="dots" width="24" height="24" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1" fill="#1b222c"/></pattern></defs>
<rect width="{W}" height="{H}" rx="16" fill="{BG}"/>
<rect width="{W}" height="{H}" rx="16" fill="url(#dots)"/>
<rect x=".5" y=".5" width="{W - 1}" height="{H - 1}" rx="16" fill="none" stroke="#30363d"/>
<g font-family="{MONO}">
<text x="56" y="84" font-size="40" font-weight="700" fill="#e6edf3">Hi, I'm <tspan fill="#22d3ee">{esc(NAME)}</tspan></text>
<text x="58" y="118" font-size="16" fill="#8b949e">{esc(SUBTITLE)}</text>
{helix}
{codon_stream(helix_end + 22, 236, dock)}
<g class="bob">
<path class="glow" d="{trace}" fill="none" stroke="#22d3ee" stroke-opacity=".035" stroke-width="16" stroke-linejoin="round" stroke-linecap="round"/>
{body}
<g><circle r="9" fill="#ffffff" opacity=".18"/><circle r="3.5" fill="#ffffff"/>
<animateMotion path="{trace}" dur="{GROW}s" keyTimes="{kt}" keyPoints="{kp}" calcMode="linear" fill="freeze"/>
<animate attributeName="opacity" values="1;0" begin="{GROW}s" dur=".4s" fill="freeze"/></g>
<g opacity="0"><circle r="8" fill="#f0abfc" opacity=".2"/><circle r="3" fill="#f0abfc"/>
<animateMotion path="{trace}" dur="8s" begin="{GROW + 1.5}s" repeatCount="indefinite"/>
<animate attributeName="opacity" values="0;.9;.9;0" keyTimes="0;.05;.9;1" dur="8s" begin="{GROW + 1.5}s" repeatCount="indefinite"/></g>
</g>
<text x="{W - 40}" y="34" font-size="12" fill="#6e7681" text-anchor="end">fold of the day · {date.isoformat()}</text>
{typed(caption, BOX[0], 340, GROW, fill="#c9d1d9")}
<text x="58" y="340" font-size="15" fill="#4ade80">&gt; <tspan fill="#c9d1d9">{esc(TAGLINE)}</tspan><tspan class="blink" fill="#4ade80"> ▋</tspan></text>
</g>
</svg>
'''


def update_readme(path, meta, svg_path):
    """Rewrite the block between the FOLD markers so the banner links to today's entry."""
    res = f" {meta['resolution']:.1f} Å" if meta["resolution"] else ""
    e = html.escape
    alt = f"DNA helix translating into today's protein: PDB {meta['id']}, {meta['name']}"
    block = (
        f'<a href="{meta["url"]}"><img src="{svg_path}" width="100%" alt="{e(alt)}"></a>\n'
        f'<p align="center"><sub>fold of the day · <a href="{meta["url"]}"><b>{meta["id"]}</b> · {e(meta["name"])}</a>'
        f' · <i>{e(meta["organism"])}</i> · {e(meta["method"])}{res}</sub></p>'
    )
    text = path.read_text(encoding="utf-8")
    new, n = re.subn(r"(<!-- FOLD:START -->).*?(<!-- FOLD:END -->)",
                     lambda m: f"{m.group(1)}\n{block}\n{m.group(2)}", text, flags=re.S)
    if not n:
        sys.exit(f"{path}: FOLD markers not found")
    path.write_text(new, encoding="utf-8")


# =============================================================================
def good_enough(ss):
    sse = sum(s != "c" for s in ss) / len(ss)
    return sse >= MIN_SSE_FRACTION and "a" in ss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", help="use this PDB ID instead of the daily pick")
    ap.add_argument("--date", type=dt.date.fromisoformat,
                    default=dt.datetime.now(dt.timezone.utc).date())
    ap.add_argument("--out", type=Path, default=Path("out/hero.svg"))
    ap.add_argument("--readme", type=Path, help="README whose FOLD block should link to the pick")
    args = ap.parse_args()

    picks = [args.pdb] if args.pdb else candidates(args.date)
    for pdb_id in picks:
        try:
            ca, co, ss = load_residues(pdb_id)
        except Exception as exc:  # malformed or unusual entries: try the next one
            print(f"skip {pdb_id}: {exc}", file=sys.stderr)
            continue
        if not args.pdb and (not MIN_RES <= len(ca) <= MAX_RES or not good_enough(ss)):
            print(f"skip {pdb_id}: {len(ca)} residues, sse {''.join(ss)[:40]}...", file=sys.stderr)
            continue
        break
    else:
        sys.exit("no suitable structure found")

    meta = metadata(pdb_id)
    ca, co = orient(ca, co)
    frags = sample(ca, co, ss)
    svg = render(meta, frags, len(ca), args.date)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(svg, encoding="utf-8")
    meta.update(residues=len(ca), date=args.date.isoformat(),
                helix=ss.count("a"), strand=ss.count("b"))
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if args.readme:
        update_readme(args.readme, meta, args.out.relative_to(args.readme.parent).as_posix())
    print(f"{meta['id']} {meta['name']} ({len(ca)} res) -> {args.out} "
          f"[{args.out.stat().st_size / 1024:.0f} KB]")


if __name__ == "__main__":
    main()
