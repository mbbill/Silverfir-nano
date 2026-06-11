#!/usr/bin/env python3
"""Linter for the design tree. Every check cites a rule stated in README.md
or EXTRACTION.md; this file is the executable form of those rules, never a
second spec.

Usage:  python3 design/lint.py [--ledger] [--skeleton] [--window NN] [--parallel]
  --skeleton   tree checks minus R-thin (a Stage-1 skeleton is pre-prune)
  --window NN  check only extraction/win-NN.{records.jsonl,ledger.tsv}
  --parallel   check all window files + cross-window seq + placements + R-derive
Exit 0 = clean. Violations -> stderr, exit 1.
"""
import json, os, re, subprocess, sys

DESIGN = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.join(DESIGN, "design-tree")
EXTR = os.path.join(DESIGN, "extraction")
ERRORS = []

SKELETON = "--skeleton" in sys.argv
PARALLEL = "--parallel" in sys.argv
WINDOW = sys.argv[sys.argv.index("--window") + 1] if "--window" in sys.argv else None

def err(rule, path, msg):
    ERRORS.append(f"[{rule}] {os.path.relpath(path, DESIGN)}: {msg}")

def report_and_exit(label):
    if ERRORS:
        print(f"{len(ERRORS)} violation(s):", file=sys.stderr)
        for e in ERRORS:
            print("  " + e, file=sys.stderr)
        sys.exit(1)
    print(f"lint clean: {label}")
    sys.exit(0)

# ---------- parallel-mode window files (EXTRACTION.md "Parallel mode") ----------
REC_PROV = re.compile(r"^(diff|author|inferred (→|->) W\d{2}-q\d+)$")
REC_VERDICTS = ("RECORD", "COVERED", "FORCED", "SKIP", "-")
REC = {}           # record id -> {"hash":..., "type":..., "window":...}
LEDGER_SEQS = []   # int seqs across all checked window ledgers

def ok_concept(c):
    return (isinstance(c, dict) and isinstance(c.get("name"), str)
            and c["name"].strip() and isinstance(c.get("anchors"), list)
            and c["anchors"] and all(isinstance(a, str) and a.strip()
                                     for a in c["anchors"]))

def check_window(nn):
    rp = os.path.join(EXTR, f"win-{nn}.records.jsonl")
    lp = os.path.join(EXTR, f"win-{nn}.ledger.tsv")
    rec_seqs = set()
    if not os.path.exists(rp):
        err("P-files", rp, "missing records file")
    else:
        for i, line in enumerate(open(rp), 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                err("P-json", rp, f"line {i}: invalid JSON"); continue
            rid, typ = r.get("id", ""), r.get("type", "")
            tag = rid or f"line {i}"
            if not re.fullmatch(rf"W{nn}-[rq]\d+", rid):
                err("P-id", rp, f"line {i}: id {rid!r} not W{nn}-r<k>/W{nn}-q<k>")
            elif rid in REC:
                err("P-id", rp, f"line {i}: duplicate id {rid}")
            if not isinstance(r.get("seq"), int):
                err("P-seq", rp, f"{tag}: seq must be an integer")
            else:
                rec_seqs.add(r["seq"])
            if not re.fullmatch(r"[0-9a-f]{8}", r.get("hash", "") or ""):
                err("P-hash", rp, f"{tag}: hash must be 8 hex chars")
            if typ != "question" and not re.fullmatch(
                    r"\d{4}-\d{2}-\d{2}", r.get("date", "") or ""):
                err("P-date", rp, f"{tag}: bad/missing date")
            if typ != "question" and not REC_PROV.match(r.get("provenance", "") or ""):
                err("P-prov", rp, f"{tag}: bad/missing provenance")
            if typ == "transition":
                verb = r.get("verb")
                if verb not in ("replaced", "dropped", "removed"):
                    err("P-verb", rp, f"{tag}: bad verb {verb!r}")
                if not ok_concept(r.get("old")):
                    err("P-concept", rp, f"{tag}: bad/missing old concept")
                if verb == "replaced" and not ok_concept(r.get("new")):
                    err("P-concept", rp, f"{tag}: replaced needs a new concept")
                if verb in ("dropped", "removed") and r.get("new") is not None:
                    err("P-concept", rp, f"{tag}: {verb} must have new=null")
                if not (isinstance(r.get("why"), str) and r["why"].strip()):
                    err("P-why", rp, f"{tag}: missing why")
                fi = r.get("frozen_items")
                if verb in ("replaced", "removed") and not (
                        isinstance(fi, list) and fi and
                        all(isinstance(x, str) and x.strip() for x in fi)):
                    err("P-frozen", rp, f"{tag}: {verb} needs frozen_items")
            elif typ == "fact":
                if not (isinstance(r.get("kind"), str) and r["kind"].strip()):
                    err("P-kind", rp, f"{tag}: missing kind")
                if not ok_concept(r.get("concept")):
                    err("P-concept", rp, f"{tag}: bad/missing concept")
                if not (isinstance(r.get("text"), str) and r["text"].strip()):
                    err("P-text", rp, f"{tag}: missing text")
            elif typ == "birth":
                if not ok_concept(r.get("concept")):
                    err("P-concept", rp, f"{tag}: bad/missing concept")
            elif typ == "question":
                if not rid.startswith(f"W{nn}-q"):
                    err("P-id", rp, f"{tag}: question ids use -q")
                for f in ("context", "question", "blocks"):
                    if not (isinstance(r.get(f), str) and r[f].strip()):
                        err("P-question", rp, f"{tag}: missing {f}")
            else:
                err("P-type", rp, f"{tag}: bad type {typ!r}")
            if rid:
                REC[rid] = {"hash": r.get("hash"), "type": typ, "window": nn,
                            "why": r.get("why") if typ == "transition" else None}
    win_ids = {i for i, v in REC.items() if v["window"] == nn}
    if not os.path.exists(lp):
        err("P-files", lp, "missing window ledger")
        return
    rows = [l.rstrip("\n").split("\t") for l in open(lp)]
    if not rows or rows[0] != ["seq", "piece-id", "class", "verdict", "ref", "depth", "batch"]:
        err("P-header", lp, "bad/missing header row")
    want = None
    ints = set()
    for r in rows[1:]:
        if len(r) != 7:
            err("P-cols", lp, f"row has {len(r)} columns: {r[:2]}"); continue
        seq, pid, cls, verdict, ref, depth, batch = r
        if re.fullmatch(r"\d+", seq):
            if want is not None and int(seq) != want:
                err("P-seq", lp, f"seq {seq}, expected {want}")
            want = int(seq) + 1
            ints.add(int(seq)); LEDGER_SEQS.append(int(seq))
        elif not re.fullmatch(r"\d+b", seq):
            err("P-seq", lp, f"bad seq {seq!r}")
        if cls not in ("add", "change", "fix", "forced", "statement", "note"):
            err("P-class", lp, f"seq {seq}: bad class {cls!r}")
        if verdict not in REC_VERDICTS:
            err("P-verdict", lp, f"seq {seq}: bad verdict {verdict!r}")
        if verdict in ("RECORD", "COVERED") and not ref.strip():
            err("P-ref", lp, f"seq {seq}: {verdict} without ref")
        if verdict == "RECORD":
            cited = re.findall(r"W\d{2}-[rq]\d+", ref)
            if not cited:
                err("P-ref", lp, f"seq {seq}: RECORD ref cites no record ids")
            for c in cited:
                if c not in win_ids:
                    err("P-ref", lp, f"seq {seq}: ref cites unknown record {c}")
        if depth == "M":
            err("P-depth", lp, f"seq {seq}: message-only depth")
        if batch != f"W{nn}":
            err("P-batch", lp, f"seq {seq}: batch {batch!r}, expected W{nn}")
    stray = rec_seqs - ints
    if stray:
        err("P-seq", rp, f"record seqs not in this window's ledger: {sorted(stray)}")

if WINDOW:
    check_window(WINDOW)
    report_and_exit(f"window {WINDOW}")

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
    """strip entry head + provenance, collapse whitespace and backticks
    (rendering, not content) -> comparable why"""
    b = re.sub(r"^- [^:]*?(\[\[[^\]]+\]\])?:", "", b, count=1)
    b = PROV.sub("", b.strip())
    return re.sub(r"\s+", " ", b.replace("`", "")).strip().rstrip(".")

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
        # R-tail: items state what holds, never why (no rationale tails)
        flat = re.sub(r"\s+", " ", para)
        m = re.search(r"(^|[ ,;(])(so|so that|because|thus|hence|since|therefore)[ ,]",
                      flat, re.I)
        if para.startswith("- ") and m and m.group(2).lower() not in ("since",) :
            err("R-tail", p, f"item has a rationale tail ({m.group(1).strip()!r}); "
                f"move the why to Facts/Moves: {flat[:70]!r}")
    # R-entry: facts/moves entries dated+labeled, provenance-tagged
    for kind, blist in (("Facts", parsed[p]["facts"]), ("Moves", parsed[p]["moves"])):
        for b in blist:
            if not ENTRY_HEAD.match(b):
                err("R-entry", p, f"{kind} entry malformed head: {b.splitlines()[0][:70]!r}")
            if not PROV.search(b):
                err("R-prov", p, f"{kind} entry missing provenance tag: {b.splitlines()[0][:70]!r}")
            if kind == "Moves" and not MOVE_VERB.match(b):
                err("R-verb", p, f"Moves entry has no boundary verb: {b.splitlines()[0][:70]!r}")
            # R-join: one entry = one claim; a chaining connective smuggles a
            # second (often inferred) clause under one provenance tag.
            if re.search(r"\b(which is why|that is why|the reason|hence|therefore)\b",
                         re.sub(r"\s+", " ", b), re.I):
                err("R-join", p, f"{kind} entry chains two claims (atomize it): "
                    f"{b.splitlines()[0][:70]!r}")

    # R-redundant: a 'rationale' fact must not share a commit with a Moves
    # entry on the same node — the move already carries that re-decision's why.
    move_hashes = {m.group(3) for b in parsed[p]["moves"]
                   if (m := ENTRY_HEAD.match(b))}
    for b in parsed[p]["facts"]:
        m = ENTRY_HEAD.match(b)
        if m and m.group(4).strip() == "rationale" and m.group(3) in move_hashes:
            err("R-redundant", p, f"rationale fact shares commit {m.group(3)} with a "
                f"move on this node; the move already records the why")
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
    # backtick-quoted spans are verbatim code (e.g. a TOML `[[table]]`
    # faithfully quoted in a why), never links
    for ref in LINK.findall(re.sub(r"`[^`]*`", "", open(p).read())):
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

# ---------- R-thin: node with no decision-content is a module-map entry ----------
# (--skeleton skips this one check: a Stage-1 skeleton is pre-prune by design)
for p in ([] if SKELETON else node_files):
    base = p[:-3]
    has_alt = os.path.isdir(base + ".alt") and any(
        f.endswith(".md") for f in os.listdir(base + ".alt")) if os.path.isdir(base + ".alt") else False
    has_children = os.path.isdir(base) and any(
        f.endswith(".md") for f in os.listdir(base)) if os.path.isdir(base) else False
    has_facts = bool(parsed[p]["facts"])
    has_moves = bool(parsed[p]["moves"])
    if not (has_alt or has_children or has_facts or has_moves):
        err("R-thin", p, "node has no .alt/, no Facts, no Moves, and no children — "
            "it asserts a component without recording a decision (module-map node); "
            "fold it into its parent, or record the decision (alternative, rationale fact)")

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

# ---------- parallel mode: all windows + placements + derivation ----------
if PARALLEL:
    wins = sorted(re.match(r"win-(\d{2})\.", f).group(1)
                  for f in os.listdir(EXTR) if re.match(r"win-\d{2}\.records\.jsonl$", f))
    if not wins:
        err("P-files", EXTR, "no win-*.records.jsonl files found")
    for nn in wins:
        check_window(nn)
    # cross-window: ledger seqs cover 1..N contiguously, no duplicates
    s = sorted(LEDGER_SEQS)
    if len(s) != len(set(s)):
        dup = sorted({x for x in s if s.count(x) > 1})
        err("P-seq", EXTR, f"duplicate seqs across windows: {dup}")
    elif s and (s[0] != 1 or s != list(range(1, len(s) + 1))):
        err("P-seq", EXTR, f"window ledgers not contiguous from 1: {s[0]}..{s[-1]}, {len(s)} rows")
    # placements: every record lands exactly once (post-reduce only)
    pp = os.path.join(EXTR, "placements.tsv")
    paths_by_hash = {}
    if os.path.exists(pp):
        placed = {}
        for i, line in enumerate(open(pp), 1):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                err("P-place", pp, f"line {i}: expected 'id<TAB>placement'"); continue
            rid, val = parts
            if rid not in REC:
                err("P-place", pp, f"line {i}: unknown record id {rid}"); continue
            if rid in placed:
                err("P-place", pp, f"line {i}: duplicate placement for {rid}"); continue
            placed[rid] = val
            if val.startswith("discarded:"):
                if not val[len("discarded:"):].strip():
                    err("P-place", pp, f"{rid}: discarded without a reason")
                continue
            if val.startswith("absorbed:"):
                t = val[len("absorbed:"):].strip()
                if REC[rid]["type"] != "transition":
                    err("P-place", pp, f"{rid}: absorbed is only for transition records")
                if ".alt" not in t:
                    err("P-place", pp, f"{rid}: absorbed target must be a superseded "
                        f"generation (.alt member): {t}")
                if not os.path.exists(os.path.join(TREE, t)):
                    err("P-place", pp, f"{rid}: target does not exist: {t}")
                elif REC[rid]["hash"]:
                    paths_by_hash.setdefault(REC[rid]["hash"], set()).add(t)
                continue
            for t in (t.strip() for t in val.split(",")):
                if t == "questions.md":
                    continue
                if not os.path.exists(os.path.join(TREE, t)):
                    err("P-place", pp, f"{rid}: target does not exist: {t}")
                else:
                    h = REC[rid]["hash"]
                    if h:
                        paths_by_hash.setdefault(h, set()).add(t)
        missing = set(REC) - set(placed)
        if missing:
            err("P-place", pp, f"{len(missing)} record(s) never placed: {sorted(missing)[:10]}")
        # R-fold: a placed (non-discarded) transition record must materialize as
        # a Moves entry carrying its hash at one of its placed nodes — placement
        # alone does not prove the re-decision was folded into the tree.
        for rid, val in placed.items():
            if REC[rid]["type"] != "transition" or val.startswith("discarded:"):
                continue
            if val.startswith("absorbed:"):
                t = val[len("absorbed:"):].strip()
                full = os.path.join(TREE, t)
                if not any(REC[rid]["hash"] in re.findall(r"\(([0-9a-f]{8})\)", b)
                           for b in parsed.get(full, {}).get("facts", [])):
                    err("R-fold", pp, f"{rid}: absorbed transition ({REC[rid]['hash']}) "
                        f"has no hashed Facts entry at {t}")
                continue
            folded = False
            for t in (t.strip() for t in val.split(",")):
                full = os.path.join(TREE, t)
                for b in parsed.get(full, {}).get("moves", []):
                    if REC[rid]["hash"] in re.findall(r"\(([0-9a-f]{8})\)", b):
                        folded = True
            if not folded:
                err("R-fold", pp, f"{rid}: placed transition ({REC[rid]['hash']}) has no "
                    f"Moves entry at any placed node — re-decision dropped by the reduce")
        # R-derive: every hashed Facts/Moves entry in the tree comes from a
        # record placed at that node ("the reduce never invents").
        for p in node_files:
            rel = os.path.relpath(p, TREE)
            for b in parsed[p]["facts"] + parsed[p]["moves"]:
                m = ENTRY_HEAD.match(b)
                if not m or not m.group(3):
                    continue
                if rel not in paths_by_hash.get(m.group(3), set()):
                    err("R-derive", p, f"entry ({m.group(3)}) has no record placed "
                        f"here: {b.splitlines()[0][:60]!r}")
        # R-why: a replaced/replaced-by move why with (diff) provenance must be
        # the verbatim why of a transition record at that hash — the reduce
        # never authors a why ("move pairs are generated from each
        # transition's single why, verbatim by construction").
        whys_by_hash = {}
        for v in REC.values():
            if v["type"] == "transition" and v["hash"] and v["why"]:
                whys_by_hash.setdefault(v["hash"], set()).add(
                    re.sub(r"\s+", " ", v["why"].replace("`", "")).strip().rstrip("."))
        for p in node_files:
            for b in parsed[p]["moves"]:
                m = ENTRY_HEAD.match(b)
                if not m or not m.group(3):
                    continue
                if not re.search(r"replaced( by)? \[\[", b.splitlines()[0] + " " +
                                 (b.splitlines()[1] if len(b.splitlines()) > 1 else "")):
                    continue
                pm = PROV.search(b)
                if not pm or pm.group(1) != "diff":
                    continue
                h = m.group(3)
                if h in whys_by_hash and norm_why(b) not in whys_by_hash[h]:
                    err("R-why", p, f"move why ({h}) is not the verbatim why of any "
                        f"transition record at that hash — reduce-authored why: "
                        f"{norm_why(b)[:70]!r}")

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
        if verdict not in ("HIT", "REPAIR", "FORCED", "SKIP", "-",
                           "RECORD", "COVERED"):  # last two: parallel mode
            err("L-verdict", lp, f"seq {seq}: bad verdict {verdict!r}")
        if verdict in ("HIT", "REPAIR", "RECORD", "COVERED") and not ref.strip():
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
