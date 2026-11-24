"""Make inserted figure captions match the original style: bold 'Fig. x.y:' + italic title."""
import copy
import sys

import docx

d = docx.Document(sys.argv[1])
ref = next(p for p in d.paragraphs if p.text.strip().startswith("Fig. 4.3:") and len(p.runs) >= 2)
label_r, title_r = ref.runs[0]._r, ref.runs[1]._r
n = 0
for p in d.paragraphs:
    t = p.text.strip()
    if t.startswith("Fig. ") and len(p.runs) == 1 and ":" in t:
        label, title = t.split(":", 1)
        for r in list(p.runs):
            r._r.getparent().remove(r._r)
        for src, text in ((label_r, label + ":"), (title_r, title)):
            r = copy.deepcopy(src)
            for x in r.xpath("./w:t"):
                r.remove(x)
            p._p.append(r)
            p.runs[-1].text = text
        n += 1
d.save(sys.argv[1])
print("restyled", n, "captions")
