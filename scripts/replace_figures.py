"""Replace the images of Figs 4.1-4.8 in the report with the current screenshots (same size/placement)."""
import os
import sys

import docx
from docx.shared import Inches

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report_figures")
FILES = {"4.1": "fig_4_1_registration", "4.2": "fig_4_2_login", "4.3": "fig_4_3_dashboard", "4.4": "fig_4_4_image_upload",
         "4.5": "fig_4_5_image_processing", "4.6": "fig_4_6_analysis_result", "4.7": "fig_4_7_result_report",
         "4.8": "fig_4_8_analysis_history"}
d = docx.Document(sys.argv[1])
ps = d.paragraphs
for i, p in enumerate(ps):
    t = p.text.strip()
    for no, fn in FILES.items():
        if t.startswith(f"Fig. {no}:"):
            pic = ps[i - 1]
            for r in pic._p.xpath("./w:r[.//w:drawing]"):
                pic._p.remove(r)
            size = {"height": Inches(7.4)} if no == "4.7" else {"width": Inches(5.6)}
            pic.add_run().add_picture(os.path.join(FIG, fn + ".png"), **size)
            print("replaced", no)
d.save(sys.argv[1])
