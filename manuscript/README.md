
### 📚 Building the document

```
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

### 📚 VSCode Setup for LaTeX Workshop

To ensure VSCode always builds the correct file (`main.tex`) when working on subfiles like chapters or sections, follow these steps:

#### ✅ 1. Add a magic comment to each subfile

At the top of every `.tex` file that is **not** `main.tex`, add this line:

```latex
% !TeX root = ../main.tex
```

This tells the LaTeX Workshop extension to treat `main.tex` as the root file when compiling, even if you're editing a subfile like `abs1/exp1.tex`.

#### ✅ 2. Enable magic comment support in settings

Open your global or workspace settings JSON file:

* Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS)
* Run: **Preferences: Open Settings (JSON)**

Then add these lines:

```json
{
  "latex-workshop.latex.rootFile.useMagicComment": true,
  "latex-workshop.latex.rootFile.doNotPrompt": true,
  "latex-workshop.latex.rootFile.useSubFile": false
}
```

These settings ensure that:

* Magic comments are respected
* You don’t get prompted to select a root file
* The extension doesn’t try to guess based on subfile structure

---

This setup prevents VSCode from trying to compile subfiles like `abs1.tex` as standalone documents and ensures that all builds happen from `main.tex`.
