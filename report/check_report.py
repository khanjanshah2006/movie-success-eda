"""Static checks on the report, standing in for a compile when no TeX is installed.

    python report/check_report.py report

Checks that every stats macro, figure, table, \\ref and \\cite resolves, that
braces, environments and $ signs balance, that there is no non-ASCII text
(pdflatex), and that no causal verbs slipped into the prose.
"""
import re
import sys
from pathlib import Path

rep = Path(sys.argv[1])
tex = (rep / "template2026.tex").read_text(encoding="utf-8")
stats = (rep / "stats" / "stats.tex").read_text(encoding="utf-8")
tables = {p.stem: p.read_text(encoding="utf-8") for p in (rep / "tables").glob("*.tex")}
figs = {p.stem for p in (rep.parent / "figures").glob("*.pdf")}
problems = []

# strip comments for analysis
def strip_comments(s):
    return "\n".join(re.sub(r"(?<!\\)%.*", "", ln) for ln in s.splitlines())

body = strip_comments(tex)

# 1. stats macros used vs defined
defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", stats))
used = set(re.findall(r"\\((?:Audit|Clean|Uni|Multi|Feat|Model)[A-Za-z]+)", body))
for m in sorted(used - defined):
    problems.append(f"undefined stats macro: \\{m}")

# 2. figures
for f in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
    f = f.strip('"')
    if f.startswith("#"):  # macro parameter inside \member's definition
        continue
    if f.endswith(".jpg"):
        if not (rep / f).exists():
            problems.append(f"missing image {f}")
    elif f not in figs:
        problems.append(f"missing figure {f}")
unused_figs = figs - set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body))
for f in sorted(unused_figs):
    problems.append(f"figure produced but not in report: {f}")

# 3. tables
for t in re.findall(r"\\tablein\{([^}]+)\}", body):
    if t not in tables:
        problems.append(f"missing table {t}")
for t in sorted(set(tables) - set(re.findall(r"\\tablein\{([^}]+)\}", body))):
    problems.append(f"table produced but not in report: {t}")

# 4. refs and cites
labels = set(re.findall(r"\\label\{([^}]+)\}", body))
for r in re.findall(r"\\ref\{([^}]+)\}", body):
    if r not in labels:
        problems.append(f"undefined ref {r}")
bib = set(re.findall(r"\\bibitem\{([^}]+)\}", body))
for c in re.findall(r"\\cite\{([^}]+)\}", body):
    for k in c.split(","):
        if k.strip() not in bib:
            problems.append(f"undefined cite {k}")

# 5. braces and environments
def brace_balance(name, s):
    s = re.sub(r"\\[{}%$&#_]", "", strip_comments(s))
    depth = 0
    for i, ch in enumerate(s):
        depth += ch == "{"
        depth -= ch == "}"
        if depth < 0:
            problems.append(f"{name}: unmatched '}}' near: {s[max(0, i-60):i+10]!r}")
            return
    if depth:
        problems.append(f"{name}: {depth} unclosed '{{'")

brace_balance("template2026.tex", tex)
brace_balance("stats.tex", stats)
for k, v in tables.items():
    brace_balance(k, v)
    for ln in v.splitlines():
        if len(re.findall(r"(?<!\\)\$", ln)) % 2:
            problems.append(f"{k}: odd number of $ in line: {ln}")

stack = []
for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", body):
    kind, env = m.groups()
    if kind == "begin":
        stack.append(env)
    elif not stack or stack.pop() != env:
        problems.append(f"environment mismatch at \\end{{{env}}}")
if stack:
    problems.append(f"unclosed environments: {stack}")

# 6. inline-math dollar balance per paragraph in the main text
for para in re.split(r"\n\s*\n", body):
    p = re.sub(r"\\\$", "", para)
    if len(re.findall(r"\$", p)) % 2:
        problems.append(f"odd $ count in paragraph: {para.strip()[:90]!r}")

# 7. non-ASCII (pdflatex without inputenc tricks)
for name, s in [("template2026.tex", tex), ("stats.tex", stats), *tables.items()]:
    for i, ln in enumerate(s.splitlines(), 1):
        bad = [ch for ch in ln if ord(ch) > 127]
        if bad:
            problems.append(f"{name}:{i}: non-ASCII {''.join(sorted(set(bad)))!r}")

# 8. causal language
for w in ["drives", "driven by", "causes", "caused", "leads to", "led to", "impact", "boost", "effect of"]:
    for m in re.finditer(rf"\b{w}\b", body, flags=re.I):
        ctx = body[max(0, m.start()-50):m.end()+30].replace("\n", " ")
        problems.append(f"causal wording '{w}': ...{ctx}...")

print(f"macros used {len(used)}, defined {len(defined)}; labels {len(labels)}; figures {len(figs)}; tables {len(tables)}")
print("\n".join(problems) if problems else "NO PROBLEMS FOUND")
