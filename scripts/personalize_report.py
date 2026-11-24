"""Make a copy of the report for another student: replace name and roll number everywhere (body, tables, headers, footers, text boxes)."""
import sys

import docx

SRC, DST, OLD_NAME, OLD_ROLL, NEW_NAME, NEW_ROLL = sys.argv[1:7]
d = docx.Document(SRC)
parts = [d.part] + [s.part for sec in d.sections for s in (sec.header, sec.footer, sec.first_page_header, sec.first_page_footer) if s.is_linked_to_previous is False or True]
seen, n = set(), 0
for part in parts:
    if id(part) in seen:
        continue
    seen.add(id(part))
    # merge text of runs per paragraph when the name is split across runs
    for p in part.element.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
        ts = list(p.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"))
        full = "".join(t.text or "" for t in ts)
        if OLD_NAME not in full and OLD_ROLL not in full:
            continue
        new = full.replace(OLD_NAME, NEW_NAME).replace(OLD_ROLL, NEW_ROLL)
        n += full.count(OLD_NAME) + full.count(OLD_ROLL)
        # put text back run by run where possible; otherwise collapse into first run
        if all(OLD_NAME not in "".join([]) for _ in [0]):
            pos = 0
            rebuilt = []
            for t in ts:
                rebuilt.append(t.text or "")
            # simple path: replace inside individual runs
            ok = True
            for t in ts:
                if t.text and (OLD_NAME in t.text or OLD_ROLL in t.text):
                    t.text = t.text.replace(OLD_NAME, NEW_NAME).replace(OLD_ROLL, NEW_ROLL)
            if "".join(t.text or "" for t in ts) != new:
                ts[0].text = new
                for t in ts[1:]:
                    t.text = ""
d.save(DST)
check = docx.Document(DST)
left = sum(1 for p in check.element.iter() if p.tag.endswith("}t") and p.text and (OLD_NAME in p.text or OLD_ROLL in p.text))
print(f"{NEW_NAME}: replaced {n}, leftover {left}")
