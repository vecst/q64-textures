Hand-editing the HD Text & UI pack
==================================

Folders (all PNGs are 6x the original size):
  font/        the dialog font: 3 sheets (digits+symbols, lowercase, uppercase). Every letter sits in a
               60-pixel-wide cell (10 original pixels). In GIMP: Image > Configure Grid, spacing 60 x 78,
               then View > Show Grid.
  text/        other lettering: damage numbers, "PUSH START", labels like Hp / Agi / Def ...
  art/         images the game draws in strips (title logo, menu frames, element wheel). Edit the whole picture;
               it is cut back into strips automatically.
  ui/          single HUD/menu pieces: icons, gems, bars, cursors. The number matches the contact sheets.
  _reference/  the ORIGINAL pixel art at the same size (do not edit; open it as a layer to trace over).

Workflow
  1. Open a PNG in GIMP. Optional: File > Open as Layers... the same name from _reference/ for tracing,
     then hide or delete that layer before exporting.
  2. Keep the transparency (the PNG must keep its alpha channel) and keep the image size. If you want to
     paint at a higher resolution, scale up by a whole number and keep the same proportions.
  3. Export As <name>_edit.png in the same folder (recommended: the original stays for comparison),
     or File > Overwrite <name>.png. An _edit file always wins over the original.
  4. Rebuild + install (game closed):   ~/games/q64-textures/build_split.sh --install
     It prints "applied N hand edits" and writes edit/preview.png so you can check before launching.

Undo an edit: delete the _edit file. (If you overwrote the original instead: delete it and run  ~/games/q64-mods/.venv/bin/python ~/games/q64-textures/edits.py export)
Text colour: the game may tint some text at runtime; the colours you paint are the colours stored in the texture.
