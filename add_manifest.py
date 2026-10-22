"""Append mod.json to an .rtz texture pack (zip) if it is missing. usage: add_manifest.py <pack.rtz> <mod.json>"""
import sys
import zipfile

rtz, manifest = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(rtz, "a") as z:
    if "mod.json" not in z.namelist():
        z.write(manifest, "mod.json")
with zipfile.ZipFile(rtz) as z:
    assert "mod.json" in z.namelist() and "rt64.json" in z.namelist()
