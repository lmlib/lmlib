# [lmlib](http://lmlib.ch/)

---
*Open-source and for Python*

A Model-Based Signal Processing Library Working With Windowed Linear State-Space and Polynomial Signal Models.

**Website and Documentation**: **[lmlib.ch](http://lmlib.ch/)**

Direct Links to:
* **[Install](https://lmlib.github.io/lmlib/installation.html)**
* **[API](https://lmlib.github.io/lmlib/api/index.html)**
* **[Coding](https://lmlib.github.io/lmlib/state-space-tutorial.html)**
* **[Examples](https://lmlib.github.io/lmlib/_generated/examples/index.html)**
* **[Signal Catalog](https://lmlib.github.io/lmlib/_generated/catalog/biosignals/index.html)**
* **[About Us](https://lmlib.github.io/lmlib/about_us.html)**
* **[Bibliography](https://lmlib.github.io/lmlib/bibliography.html)**


---
## Install lmlib
* **pip: ```pip install lmlib ```**

* **conda: [see here](https://lmlib.github.io/lmlib/installation.html#conda)**

## Developer Info

Installation via:
* **pip (dev-mode): ```pip install -e lmlib ```**
* **[Git Workflow Guide](https://nvie.com/posts/a-successful-git-branching-model/)**


## How to build documenation
```bash
python3.14 -m venv venv
```
```bash
source venv/bin/activate
```
```bash
pip install -r requirements.txt
```
* create gallery
```bash
python scripts/create_gallery.py
```
* build doc
```bash
mkdocs build -c
```
* livereload
```bash
mkdocs serve --livereload -c
```

## Deployment Documentation

The automated pipeline consists of four sequential jobs:

### 1. release Job (tags only):
* Triggered by: Tags matching refs/tags/*
* Purpose: Creates a GitHub release with tag message as notes
* Outputs: Official release on GitHub

### 2. build_gallery Job
* Triggered by: Any push to develop or tags
* Purpose: Generates gallery documentation from code examples
* Output: docs/_generated folder as artifact

### 3. build_mkdocs Job
* Prerequisite: build_gallery must succeed
* Purpose:
* Downloads generated docs from previous step
* Creates API documentation and changelog
* Replaces version placeholders with actual version
* Builds the complete MkDocs site
* Output: ./site folder as artifact

### 4. deploy Job
* Prerequisite: build_mkdocs must succeed
* Purpose: Deploys the built site to GitHub Pages
* Destination: gh-pages branch
* Result: Documentation becomes live at your GitHub Pages URL

### How to trigger new release:

```bash
git push -a vX.X.X
```

* Then a text editor opens for release notes (make double newlines for a newline in the release notes of github)
```bash
this is a new major release:

* new feature
* removed functions
* faster backend


```

### How to change release notes in news:

* Update release notes directly on github page and save
* Rerun job **3. build_mkdocs**. this will update the news website with the latest release notes.


### How to make changes on documenation without release:

* Push main branche with newest documentation
