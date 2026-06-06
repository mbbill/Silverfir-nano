#!/usr/bin/env python3
"""Linter for the design tree. Every check cites a rule stated in README.md;
this file is the executable form of that grammar, never a second spec.

Usage:  python3 design/lint.py [--ledger]
Exit 0 = clean. Violations -> stderr, exit 1.
"""
import os, re, subprocess, sys

DESIGN = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.join(DESIGN, "design-tree")
ERRORS = []

def err(rule, path, msg):
    ERRORS.append(f"[{rule}] {os.path.relpath(path, DESIGN)}: {msg}")

# ---------- collect ----------
node_files, fact_files = [], []
for dirpath, dirnames, filenames in os.walk(TREE):
    for f in filenames:
        if not f.endswith(".md"):
            continue
        p = os.path.join(dirpath, f)
        (fact_files if dirpath.endswith(".fact") else node_files).append(p)

def stem(p):
    return os.path.basename(p)[:-3]

# logical path: strip .alt / .fact segments
def logical(p):
    rel = os.path.relpath(p, TREE)[:-3]
    parts = [seg[:-4] if seg.endswith(".alt") else seg for seg in rel.split(os.sep)]
    return "/".join(parts)

stems = {}
for p in node_files:
    stems.setdefault(stem(p), []).append(p)

# ---------- R-root: exactly one top-level node ----------
top = [p for p in node_files if os.path.dirname(p) == TREE]
if len(top) != 1:
    err("R-root", TREE, f"expected exactly 1 top-level node, found {len(top)}: "
        + ", ".join(os.path.basename(t) for t in top))

# ---------- R-orphan / R-empty: structure dirs ----------
for dirpath, dirnames, filenames in os.walk(TREE):
    for d in dirnames:
        full = os.path.join(dirpath, d)
        base = d[:-5] if d.endswith(".fact") else (d[:-4] if d.endswith(".alt") else d)
        sibling = os.path.join(dirpath, base + ".md")
        if not os.path.exists(sibling):
            err("R-orphan", full, f"directory has no sibling {base}.md")
        if (d.endswith(".alt") or d.endswith(".fact")) and not any(
                f.endswith(".md") for f in os.listdir(full)):
            err("R-empty", full, "empty structure directory")

# ---------- node file grammar ----------
ENTRY_HEAD = re.compile(
    r"^- (\d{4}-\d{2}-\d{2})( \(([0-9a-f]{8})\))? ([a-z][a-z -]*?)(:| \[\[)")
PROV = re.compile(r"\((diff|author|inferred → Q\d+)\)\.?\s*$")
MOVE_VERB = re.compile(
    r"^- \S+( \([0-9a-f]{8}\))? (replaced by \[\[[^\]]+\]\]:|replaced \[\[[^\]]+\]\]:"
    r"|dropped:|removed:|revived:)")
LINK = re.compile(r"\[\[([^\]]+)\]\]")

def blocks(text):
    """split a section body into '- ' entry blocks separated by blank lines"""
    out, cur = [], []
    for line in text.splitlines():
        if line.startswith("- ") and cur:
            out.append("\n".join(cur)); cur = [line]
        elif line.strip() == "":
            if cur: out.append("\n".join(cur)); cur = []
        else:
            cur.append(line) if cur or line.startswith("- ") else None
            if not cur and line.startswith("- "): cur = [line]
    if cur: out.append("\n".join(cur))
    return [b for b in out if b.strip()]

def norm_why(b):
    """strip entry head + provenance, collapse whitespace -> comparable why"""
    b = re.sub(r"^- [^:]*?(\[\[[^\]]+\]\])?:", "", b, count=1)
    b = PROV.sub("", b.strip())
    return re.sub(r"\s+", " ", b).strip().rstrip(".")

parsed = {}  # path -> dict(items, facts, moves)
for p in node_files:
    text = open(p).read()
    # R-title: first non-empty line must be an item
    first = next((l for l in text.splitlines() if l.strip()), "")
    if first.startswith("#"):
        err("R-title", p, "file starts with a heading; items come first, no titles")
    # R-sections: only ## Facts / ## Moves headings, in that order
    heads = re.findall(r"^(#+ .*)$", text, flags=re.M)
    allowed = ["## Facts", "## Moves"]
    seq = [h for h in heads]
    if [h for h in seq if h not in allowed]:
        err("R-sections", p, f"unexpected heading(s): {[h for h in seq if h not in allowed]}")
    if seq != [h for h in allowed if h in seq]:
        err("R-sections", p, f"sections out of order: {seq}")
    # split
    def section(name):
        m = re.search(rf"^## {name}$(.*?)(?=^## |\Z)", text, flags=re.M | re.S)
        return m.group(1) if m else ""
    items_part = re.split(r"^## ", text, flags=re.M)[0]
    facts, moves = section("Facts"), section("Moves")
    parsed[p] = dict(items=items_part, facts=blocks(facts), moves=blocks(moves))
    # R-items: every items paragraph starts with '- '
    for para in re.split(r"\n\s*\n", items_part.strip()):
        if para and not para.startswith("- "):
            err("R-items", p, f"non-item content in items section: {para.splitlines()[0][:60]!r}")
    # R-entry: facts/moves entries dated+labeled, provenance-tagged
    for kind, blist in (("Facts", parsed[p]["facts"]), ("Moves", parsed[p]["moves"])):
        for b in blist:
            if not ENTRY_HEAD.match(b):
                err("R-entry", p, f"{kind} entry malformed head: {b.splitlines()[0][:70]!r}")
            if not PROV.search(b):
                err("R-prov", p, f"{kind} entry missing provenance tag: {b.splitlines()[0][:70]!r}")
            if kind == "Moves" and not MOVE_VERB.match(b):
                err("R-verb", p, f"Moves entry has no boundary verb: {b.splitlines()[0][:70]!r}")
    # R-meta: tree never references its own construction
    for word in ("ledger", "batch report", "design tree", "extraction run", "deferred until"):
        if word in text.lower():
            err("R-meta", p, f"workflow-metadata vocabulary in tree: {word.strip()!r}")

# ---------- R-link: every [[link]] resolves ----------
def resolve(ref, frm):
    name = ref.split("/")[-1]
    # allow explicit fact links like [[x.fact/slug]]
    if ".fact/" in ref:
        cand = [f for f in fact_files if f.endswith(ref.split("/")[-1] + ".md")
                or logical(f).endswith(ref)]
        return cand[0] if cand else None
    cands = stems.get(name, [])
    if len(cands) == 1:
        return cands[0]
    near = [c for c in cands if os.path.dirname(c).startswith(os.path.dirname(frm))
            or os.path.dirname(frm).startswith(os.path.dirname(c).replace(".alt", ""))]
    return near[0] if len(near) == 1 else (cands[0] if cands else None)

for p in node_files:
    for ref in LINK.findall(open(p).read()):
        if resolve(ref, p) is None:
            err("R-link", p, f"unresolvable link [[{ref}]]")

# ---------- R-pair: replaced <-> replaced by, verbatim why ----------
for p in node_files:
    for b in parsed[p]["moves"]:
        m = re.match(r"^- \S+ \(([0-9a-f]{8})\) replaced \[\[([^\]]+)\]\]:", b)
        if not m:
            continue
        h, loser_ref = m.groups()
        loser = resolve(loser_ref, p)
        if loser is None:
            continue  # R-link already fired
        twins = [tb for tb in parsed.get(loser, {}).get("moves", [])
                 if f"({h}) replaced by [[" in tb]
        if not twins:
            err("R-pair", p, f"replaced [[{loser_ref}]] ({h}) has no 'replaced by' twin in loser")
        elif norm_why(b) != norm_why(twins[0]):
            err("R-pair", p, f"why differs from twin in {os.path.basename(loser)} ({h})")

# ---------- R-frozen: .alt members end superseded; main nodes do not ----------
for p in node_files:
    in_alt = ".alt" + os.sep in p or p.split(os.sep)[-2].endswith(".alt")
    mv = parsed[p]["moves"]
    last = mv[-1] if mv else ""
    if in_alt and not re.search(r"(replaced by \[\[|removed:)", last):
        err("R-frozen", p, ".alt member's Moves must end in 'replaced by'/'removed'")
    if not in_alt and re.search(r"replaced by \[\[", last) and "revived" not in last:
        err("R-frozen", p, "main-tree node ends 'replaced by' (should it be in .alt/?)")

# ---------- R-factfile: graduated fact files are heading-free prose ----------
for p in fact_files:
    if re.search(r"^#+ ", open(p).read(), flags=re.M):
        err("R-factfile", p, "fact file contains headings")

# ---------- R-append: Moves/Facts append-only vs last accepted commit ----------
def git(*args):
    return subprocess.run(["git", *args], cwd=DESIGN, capture_output=True, text=True)

if git("rev-parse", "HEAD").returncode == 0:
    # A pivot (README "A pivot is file motion") relocates the incumbent's
    # frozen Facts/Moves into the challenger's .alt/ in the same change, so a
    # committed entry that leaves one file must reappear verbatim in another —
    # that is relocation, not removal. Append-only therefore means "no entry
    # lost from the tree", checked tree-globally; only entries that vanish
    # everywhere are edited/removed.
    tree_entries = set()
    for p in node_files:
        tree_entries.update(re.sub(r"\s+", " ", b)
                            for b in parsed[p]["facts"] + parsed[p]["moves"])
    for p in node_files:
        rel = os.path.relpath(p, os.path.dirname(DESIGN))
        old = git("show", f"HEAD:{rel}")
        if old.returncode != 0:
            continue  # new file
        old_entries = set()
        for sec in re.findall(r"^## (?:Facts|Moves)$(.*?)(?=^## |\Z)",
                              old.stdout, flags=re.M | re.S):
            old_entries.update(re.sub(r"\s+", " ", b) for b in blocks(sec))
        gone = old_entries - tree_entries
        if gone:
            err("R-append", p, f"{len(gone)} committed Facts/Moves entr{'y' if len(gone)==1 else 'ies'} edited or removed")

# ---------- optional: ledger checks ----------
if "--ledger" in sys.argv:
    lp = os.path.join(DESIGN, "extraction", "ledger.tsv")
    rows = [l.rstrip("\n").split("\t") for l in open(lp)][1:]
    want, logicals = 1, {logical(p) for p in node_files}
    for r in rows:
        if len(r) != 7:
            err("L-cols", lp, f"row has {len(r)} columns: {r[:2]}"); continue
        seq, pid, cls, verdict, ref, depth, batch = r
        if re.fullmatch(r"\d+", seq):
            if int(seq) != want:
                err("L-seq", lp, f"seq {seq}, expected {want}")
            want = int(seq) + 1
        elif not re.fullmatch(r"\d+b", seq):
            err("L-seq", lp, f"bad seq {seq!r}")
        if verdict not in ("HIT", "REPAIR", "FORCED", "SKIP", "-"):
            err("L-verdict", lp, f"seq {seq}: bad verdict {verdict!r}")
        if verdict in ("HIT", "REPAIR") and not ref.strip():
            err("L-ref", lp, f"seq {seq}: {verdict} without ref")
        if depth == "M":
            err("L-depth", lp, f"seq {seq}: message-only depth")
        if verdict == "HIT":
            named = re.findall(r"[a-z0-9-]+(?:/[a-z0-9-]+)+", ref)
            words = set(re.findall(r"[a-z0-9-]+", ref))
            stems_ok = words & {l.split("/")[-1] for l in logicals}
            if named and not any(n in logicals for n in named) and not stems_ok:
                err("L-hitref", lp, f"seq {seq}: no named node resolves: {named}")

# ---------- report ----------
if ERRORS:
    print(f"{len(ERRORS)} violation(s):", file=sys.stderr)
    for e in ERRORS:
        print("  " + e, file=sys.stderr)
    sys.exit(1)
print(f"lint clean: {len(node_files)} nodes, {len(fact_files)} fact files")
