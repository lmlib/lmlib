# docs/scripts_docs/create_gallery.py
import os
import sys
import glob
import subprocess
import shutil
import re
import html
from pathlib import Path
import matplotlib
# Force non-interactive backend immediately to prevent GUI issues in headless environments
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# Adjust paths
# This script now lives at: <repo>/docs/scripts_docs/create_gallery.py
#   __file__.parent        -> docs/scripts_docs
#   __file__.parent.parent -> docs
#   .parent.parent.parent  -> repo root
SCRIPT_DIR = Path(__file__).parent              # docs/scripts_docs
DOC_DIR = SCRIPT_DIR.parent                     # docs
BASE_DIR = DOC_DIR.parent                       # repo root (examples/ and coding/ live here)

# All generated content (markdown + PNGs) goes into this single folder, grouped by
# category. It is a build artifact: add `docs/_generated/` to .gitignore.
GENERATED_DIR = DOC_DIR / "_generated"          # docs/_generated

# Ensure lmlib is found
sys.path.insert(0, str(BASE_DIR))

def extract_title_and_description(docstring):
    """
    Extracts the title (first line) and description (text after the title line).
    Preserves line breaks, bullet points (* item), and bold markers (**text**).
    Returns:
        title: str
        description: str (multiline, markdown-formatted)
        description_flat: str (single-line, for use in tables)
    """
    if not docstring:
        return "Example", "No description available.", "No description available."
    
    lines = docstring.strip().split('\n')
    
    # Find the title line (usually the first non-empty line)
    title = ""
    desc_lines = []
    found_separator = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            # Preserve blank lines as paragraph separators
            if found_separator and desc_lines and desc_lines[-1] != "":
                desc_lines.append("")
            continue
            
        if not title:
            # First non-empty line is the title
            title = stripped
            found_separator = False
        elif not found_separator and ('===' in stripped or '---' in stripped):
            # Found the separator line
            found_separator = True
        elif found_separator:
            # Everything after the separator is the description
            desc_lines.append(stripped)
    
    # Remove trailing blank lines
    while desc_lines and desc_lines[-1] == "":
        desc_lines.pop()
    
    if not desc_lines:
        return title, "No description available.", "No description available."
    
    # Multiline description (preserves list formatting for detail pages)
    description = "\n".join(desc_lines).strip()
    
    # Flat description (single line for table cells)
    flat_lines = []
    for line in desc_lines:
        if line == "":
            continue
        # Convert "* item" to "• item" for inline display
        if line.startswith("* ") and not line.startswith("**"):
            flat_lines.append("• " + line[2:])
        else:
            flat_lines.append(line)
    description_flat = " ".join(flat_lines).strip()
    description_flat = re.sub(r'\s+', ' ', description_flat)
    
    return title, description, description_flat


# How many words of the description to show in a gallery cell. Kept generous so a
# blurb can fill roughly the height of its thumbnail (#13).
GALLERY_BLURB_WORDS = 48

# Per-example execution cap (seconds). A single slow/hanging example is skipped
# (with a note on its detail page) instead of blocking the whole gallery build.
EXAMPLE_TIMEOUT = 600


def truncate_words(text, max_words=GALLERY_BLURB_WORDS):
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def description_to_plain(text):
    """Flatten a markdown/RST docstring blurb to clean plain text.

    Gallery cells are emitted as raw HTML <td>, where MkDocs does NOT process
    markdown -- so a literal ``[\\[Cite\\]](../bibliography.md#cite)`` would show
    its raw syntax and the escaped brackets would even be picked up as math. We
    therefore strip link/markup syntax here and keep only the readable text.
    """
    if not text:
        return ""
    s = text
    # [label][ref] cross-references -> label   (label may contain escaped brackets)
    s = re.sub(r'\[((?:\\.|[^\]\\])*)\]\[[^\]]*\]', r'\1', s)
    # [label](url) inline links -> label
    s = re.sub(r'\[((?:\\.|[^\]\\])*)\]\([^)]*\)', r'\1', s)
    # unescape backslash-escaped punctuation (e.g. \[ \] -> [ ])
    s = re.sub(r'\\([\\`*_{}\[\]()#+\-.!])', r'\1', s)
    # inline code ``x`` / `x` -> x
    s = re.sub(r'``([^`]+)``', r'\1', s)
    s = re.sub(r'`([^`]+)`', r'\1', s)
    # bold / italic markers
    s = re.sub(r'\*\*([^*]+)\*\*', r'\1', s)
    s = re.sub(r'\*([^*]+)\*', r'\1', s)
    # drop math delimiters, keep the inner text
    s = s.replace('$', '')
    # collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def read_folder_meta(folder_path):
    """Read a folder's ``README.md`` for its gallery section title and intro.

    The first level-1 heading (``# Title``) is the section/page title; everything
    after it is the intro markdown, inserted verbatim (no RST conversion -- the
    READMEs are authored as proper markdown, so any bibliography links must be
    written relative to the rendered gallery page, e.g. ``../../bibliography.md``).
    Falls back to a title derived from the folder name and an empty intro.
    """
    p = folder_path / "README.md"
    if p.exists():
        text = p.read_text(encoding="utf-8")
        m = re.match(r'\s*#\s+(.+)', text)
        if m:
            return m.group(1).strip(), text[m.end():].strip()
        return (folder_path.name.replace('-', ' ').replace('_', ' ').title(),
                text.strip())
    return folder_path.name.replace('-', ' ').replace('_', ' ').title(), ""


# Prose State-Space Tutorial. Its first paragraph (the one-line intro under the
# title) is reused verbatim as the lead blurb on the Teaching page, so the wording
# lives in a single place (the tutorial) instead of being duplicated here.
STATE_SPACE_TUTORIAL_MD = DOC_DIR / "state-space-tutorial.md"


def read_tutorial_intro(md_path=STATE_SPACE_TUTORIAL_MD):
    """Return the first prose paragraph of a markdown file as a single line.

    Skips a leading level-1 title (``# ...``) and any blank lines, then collects
    the first non-empty paragraph (up to the next blank line), collapsing the soft
    line breaks into spaces. Used so the Teaching page's lead blurb is pulled from
    ``state-space-tutorial.md`` rather than hard-coded here. Returns an empty string
    if the file or an intro paragraph is missing.
    """
    try:
        text = md_path.read_text(encoding="utf-8")
    except OSError:
        return ""
    para = []
    for line in text.splitlines():
        stripped = line.strip()
        if not para:
            # Skip the title and any leading blank lines until the first prose line.
            if not stripped or stripped.startswith("#"):
                continue
            para.append(stripped)
        elif not stripped:
            break  # blank line -> end of the first paragraph
        else:
            para.append(stripped)
    return re.sub(r"\s+", " ", " ".join(para)).strip()


def render_gallery_table(entries, link_prefix="", max_words=GALLERY_BLURB_WORDS):
    """Render gallery rows as a class-tagged HTML table.

    Differs from the default markdown gallery table: the plot column comes first,
    there is no header row, and the table carries the ``gallery`` class so Material
    leaves it unstyled (styling lives in docs/css/custom.css under
    ``.md-typeset table.gallery`` -- larger font, ~70% thumbnails, etc.).

    ``entries`` items are ``(title, description_flat, base_name, plot_filename,
    plot_generated)``; ``link_prefix`` is prepended to links/images (e.g.
    ``"11-detection/"`` when the table is embedded in the combined page one level
    above the detail pages). The description is reduced to clean plain text
    because raw-HTML cells are not markdown-processed by MkDocs.
    """
    rows = []
    for title, description_flat, base_name, plot_filename, plot_generated in entries:
        # Links live inside a raw-HTML <table>, which MkDocs does NOT post-process
        # (it would otherwise rewrite a .md link to .html). So we emit the final
        # .html URL directly. This assumes use_directory_urls: false (as configured).
        href = f"{link_prefix}{base_name}.html"
        if plot_generated:
            plot = (f'<a href="{href}">'
                    f'<img src="{link_prefix}{plot_filename}" alt="Plot"></a>')
        else:
            plot = "<em>No Plot</em>"
        blurb = truncate_words(description_to_plain(description_flat), max_words)
        desc = f'<a href="{href}"><b>{html.escape(title)}</b>'
        if blurb:
            desc += f'<br><small>{html.escape(blurb)}</small>'
        desc += '</a>'
        rows.append(
            "<tr>\n"
            f'  <td class="gallery-plot">{plot}</td>\n'
            f'  <td class="gallery-desc">{desc}</td>\n'
            "</tr>"
        )
    return ('<table class="gallery">\n<tbody>\n'
            + "\n".join(rows)
            + "\n</tbody>\n</table>\n")


def build_combined_gallery_page(page_title, intro_md, sections, output_dir,
                                lead_sections=None):
    """Write a single combined gallery page (Application Examples, Teaching, ...).

    ``page_title``/``intro_md`` come from the category's top-level README.md.
    ``lead_sections`` is an optional list of ``(title, body_md)`` rendered before
    the galleries (used for the State-Space Tutorial link on the Teaching page).
    ``sections`` is an ordered list of ``(section_title, folder_name, entries,
    intro_md)``; one ``## <section_title>`` heading per folder makes the
    right-hand table of contents the in-page navigation.
    """
    parts = [f"# {page_title}\n"]
    if intro_md:
        parts.append(f"\n{intro_md}\n")
    for title, body_md in (lead_sections or []):
        parts.append(f"\n## {title}\n")
        if body_md:
            parts.append(f"\n{body_md}\n")
    for section_title, folder_name, entries, sec_intro in sections:
        parts.append(f"\n## {section_title}\n")
        if sec_intro:
            parts.append(f"\n{sec_intro}\n")
        parts.append(render_gallery_table(entries, link_prefix=f"{folder_name}/"))
    output_dir.mkdir(parents=True, exist_ok=True)
    index_file = output_dir / "index.md"
    with open(index_file, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    total = sum(len(e) for _, _, e, _ in sections)
    print(f"Generated combined page: {index_file} ({total} entries)")


def process_folder(folder_path, folder_parent: str, starting_pattern: str,
                   write_index: bool = True, new_layout: bool = False):
    # --- Helper for Truncation (old markdown-table layout uses a short blurb) ---
    def truncate_words(text, max_words=10):
        if not text:
            return ""
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]) + "..."

    folder_name = folder_path.name
    # Section title and intro come from the folder's README.md (single source of
    # truth; no hard-coded titles or RST parsing here).
    section_title, intro_md = read_folder_meta(folder_path)
    # Everything generated lands under docs/_generated/<category>/<folder_name>/,
    # regardless of where the source .py files are read from.
    target_output_dir = GENERATED_DIR / folder_parent
    output_folder = target_output_dir / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)

    # Copy any local data files (CSVs, JPGs, PNGs sitting next to the examples) into the
    # output folder so they are served by the site and can be linked for download
    # (#16). Library-bundled signals loaded via load_lib_csv are NOT here.
    extensions = ["*.csv", "*.jpg", "*.jpeg", "*.png"]
#    local_assets = {p.name for p in folder_path.glob("*.csv")}
#    for csv in folder_path.glob("*.csv"):
#        shutil.copy2(csv, output_folder / csv.name)
    local_assets = set()
    for pattern in ("*.csv", "*.jpg", "*.jpeg", "*.png"):
       for file in folder_path.glob(pattern):
           local_assets.add(file.name)
           shutil.copy2(file, output_folder / file.name)

    gallery_entries = []
    entries = []  # structured rows for the new gallery layout
    pattern = starting_pattern+"*.py"
    py_files = [Path(f) for f in sorted(glob.glob(str(folder_path / pattern)))]
    
    if not py_files:
        print(f"No example files found in {folder_path}")
        return section_title, entries, intro_md

    md_content = f"# {section_title}\n\n"
    if intro_md:
        md_content += intro_md + "\n\n"

    # --- Standard Markdown Table ---
    # Custom CSS.
    md_content += """
<style>
.md-typeset table th:nth-child(1),
.md-typeset table td:nth-child(1) { width: 30%; }
.md-typeset table th:nth-child(2),
.md-typeset table td:nth-child(2) { width: 70%; }
</style>

| Example | Plot |
| :--- | :--- |
"""
    
    for py_file in py_files:
        file_name = py_file.name
        base_name = file_name.replace('.py', '')
        plot_filename = f"{base_name}.png"
        plot_path = output_folder / plot_filename
        
        env = os.environ.copy()
        env['MPLBACKEND'] = 'Agg'
        
        with open(py_file, 'r', encoding='utf-8') as f:
            code = f.read()
        
        has_show = 'plt.show()' in code
        has_plot_import = 'import matplotlib' in code or 'import pyplot' in code
        modified_code = code

        # Capture every figure the example produces.
        # We inject a helper that, at each plt.show() (and once more at the end),
        # saves all currently-open figures: the first becomes <base>.png (the
        # gallery thumbnail), the rest <base>_2.png, <base>_3.png, ... (shown on
        # the detail page). A simple savefig() per show() would only ever keep the
        # final figure, because every show() wrote to the same file.
        if has_show or has_plot_import:
            stem = str(output_folder / base_name).replace('\\', '/')
            helper = (
                "\n__lm_fig_paths = []\n"
                "def __lm_save_open_figs():\n"
                "    import matplotlib.pyplot as _plt\n"
                "    for _n in _plt.get_fignums():\n"
                "        _i = len(__lm_fig_paths) + 1\n"
                f"        _name = r'{stem}' + ('.png' if _i == 1 else ('_%d.png' % _i))\n"
                "        _plt.figure(_n).savefig(_name, dpi=150, bbox_inches='tight')\n"
                "        __lm_fig_paths.append(_name)\n"
                "    _plt.close('all')\n"
            )
            modified_code = (helper + "\n"
                             + code.replace('plt.show()', '__lm_save_open_figs()')
                             + "\n__lm_save_open_figs()\n")
        
        # Write the temp script INTO the source folder (not the output folder) so that
        # __file__-relative and cwd-relative data loads inside the example resolve
        # correctly (e.g. np.load of a .npy/.csv sitting next to the example). The plot
        # is saved to an absolute path below, so it still lands in the output folder
        # regardless of the working directory.
        temp_file = folder_path / f"_temp_{base_name}.py"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(modified_code)
            
        plot_generated = False
        console_output = ""
        
        try:
            result = subprocess.run([sys.executable, str(temp_file)], check=True,
                                    env=env, cwd=str(folder_path),
                                    capture_output=True, text=True,
                                    timeout=EXAMPLE_TIMEOUT)
            if result.stdout: console_output = result.stdout.strip()
            if result.stderr:
                console_output = (console_output + "\n\n" + result.stderr.strip()) if console_output else result.stderr.strip()
            if plot_path.exists(): plot_generated = True
        except subprocess.TimeoutExpired:
            print(f"Timeout ({EXAMPLE_TIMEOUT}s) executing {file_name}; skipping.")
            console_output = f"(execution exceeded {EXAMPLE_TIMEOUT}s and was skipped)"
        except subprocess.CalledProcessError as e:
            print(f"Error executing {file_name}: {e.stderr.strip() if e.stderr else 'Unknown'}")
            if e.stdout: console_output = e.stdout.strip()
            if e.stderr: console_output = (console_output + "\n\n" + e.stderr.strip()) if console_output else e.stderr.strip()
        except Exception as e:
            print(f"Unexpected error with {file_name}: {e}")
        finally:
            if temp_file.exists(): temp_file.unlink()

        # Collect every figure the example produced: <base>.png plus any
        # <base>_N.png. The first is the gallery thumbnail; all are shown on the
        # detail page.
        def _fig_key(p):
            m = re.search(r'_(\d+)\.png$', p.name)
            return int(m.group(1)) if m else 1
        all_plots = []
        if plot_path.exists():
            all_plots.append(plot_filename)
        extras = sorted(output_folder.glob(f"{base_name}_*.png"), key=_fig_key)
        all_plots += [p.name for p in extras]
        plot_generated = bool(all_plots)

        # Local data files referenced by this example (for download links, #16).
        referenced_assets = [name for name in sorted(local_assets) if name in code]

        # Extract & Truncate
        docstring_match = re.search(r'"""(.*?)"""', code, re.DOTALL)
        docstring = docstring_match.group(1) if docstring_match else ""
        title, description, description_flat = extract_title_and_description(docstring)
        truncated_desc = truncate_words(description_flat, 10)
        safe_description = truncated_desc.replace("|", "&#124;")
        
        # Build Cell Content
        cell_content = f"{title}<br><small>{safe_description}</small>"
        
        if plot_generated:
            # Using HTML img tag inside Markdown cell to control size (e.g., width=300px)
            plot_cell = f'<img src="{plot_filename}" alt="Plot" width="100%">'
        else:
            plot_cell = "*No Plot*"
            
        md_content += f"| [{cell_content}]({base_name}.md) | {plot_cell} |\n"
        entries.append((title, description_flat, base_name, plot_filename, plot_generated))
            
        # --- Create Detailed Page (Plot -> Data -> Console -> Code) ---
        detail_md = f"# {title}\n\n{description}\n\n"
        
        if all_plots:
            detail_md += "## Plot\n\n"
            for pf in all_plots:
                detail_md += f"![Plot]({pf})\n\n"
        else:
            detail_md += "> **Note:** No graphical output.\n\n"

        if referenced_assets:
            detail_md += "## Data\n\nThis example uses the following data file(s):\n\n"
            for name in referenced_assets:
                detail_md += f"- [`{name}`]({name})\n"
            detail_md += "\n"
            
        if console_output:
            detail_md += f"## Console Output\n\n```text\n{console_output}\n```\n\n"
            
        detail_md += f"## Code\n\n```python\n{code}\n```\n"
        
        with open(output_folder / f"{base_name}.md", 'w', encoding='utf-8') as f:
            f.write(detail_md)
            
        gallery_entries.append((title, base_name))

    if write_index:
        index_file = output_folder / "index.md"
        if new_layout:
            # Same layout as the combined Application Examples page (#6/#8):
            # swapped columns, no header row, larger font, ~70% thumbnails.
            page = [f"# {section_title}\n"]
            if intro_md:
                page.append(f"\n{intro_md}\n")
            page.append(render_gallery_table(entries, link_prefix=""))
            content = "\n".join(page)
        else:
            content = md_content
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Generated: {index_file} ({len(gallery_entries)} entries)")
    else:
        print(f"Processed {folder_name} ({len(gallery_entries)} entries) "
              f" ")

    return section_title, entries, intro_md

# Main logic
if __name__ == "__main__":
    # Ensure the top-level generated dir exists up front, so downstream steps (e.g.
    # the Makefile's `.stamp` marker) always have a home even if nothing is generated.
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    # Scan folders. coding/ and examples/ live at the repo root (BASE_DIR) and are
    # auto-discovered (sorted by their numeric prefix); only their generated
    # galleries are written into docs/. catalog/ lives inside docs/.
    def _subfolders(parent):
        return [p for p in sorted(parent.iterdir()) if p.is_dir()] if parent.exists() else []

    coding_folders = _subfolders(BASE_DIR / "coding")
    examples_folders = [p for p in _subfolders(BASE_DIR / "examples") if p.name != "90-beta"] #exclude 90-beta
    catalog_folders = [
        DOC_DIR / "catalog" / "biosignals",
        DOC_DIR / "catalog" / "generators",
    ]

    # --- Teaching: combine the coding galleries into one page (#15), with the
    #     prose State-Space Tutorial linked as the first section. ---
    coding_sections = []
    for folder in coding_folders:
        if folder.exists():
            print(f"Processing Coding: {folder}")
            section_title, entries, intro_md = process_folder(
                folder, "coding", "guide-", write_index=False
            )
            coding_sections.append((section_title, folder.name, entries, intro_md))
        else:
            print(f"Folder not found: {folder}")

    if coding_sections:
        page_title, page_intro = read_folder_meta(BASE_DIR / "coding")
        # Reuse the tutorial's own intro sentence (first paragraph of
        # state-space-tutorial.md) instead of duplicating it here.
        tutorial_intro = read_tutorial_intro()
        tutorial = (
            f"{tutorial_intro}\n\n"
            "[Open the State-Space Tutorial](../../state-space-tutorial.md)"
        )
        build_combined_gallery_page(
            page_title, page_intro, coding_sections, GENERATED_DIR / "coding",
            lead_sections=[("State-Space Tutorial", tutorial)],
        )

    # --- Catalog. Biosignals uses the new gallery layout (#8); the generators
    #     catalog keeps the classic markdown table (its images are linked from the
    #     lmlib.utils.generator docstrings). ---
    for folder in catalog_folders:
        if folder.exists():
            print(f"Processing Catalog: {folder}")
            process_folder(folder, "catalog", "example-",
                           new_layout=(folder.name == "biosignals"))
        else:
            print(f"Folder not found: {folder}")

    # --- Application Examples: one combined page with a right-hand TOC (#6). ---
    example_sections = []
    for folder in examples_folders:
        if folder.exists():
            print(f"Processing Examples: {folder}")
            section_title, entries, intro_md = process_folder(
                folder, "examples", "example-", write_index=False
            )
            example_sections.append((section_title, folder.name, entries, intro_md))
        else:
            print(f"Folder not found: {folder}")

    if example_sections:
        page_title, page_intro = read_folder_meta(BASE_DIR / "examples")
        build_combined_gallery_page(
            page_title, page_intro, example_sections, GENERATED_DIR / "examples")

    print("Gallery build completed.")
