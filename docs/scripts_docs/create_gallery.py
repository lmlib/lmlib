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


def rst_readme_to_md(readme_path, bib_depth="../../"):
    """Extract the body text of an RST README and convert it to markdown.

    Drops the RST target (``.. _label:``), the section title and its underline,
    keeping only the descriptive sentence(s). RST citations like ``[Cite2022]_``
    become markdown links into the bibliography (``bib_depth`` is the relative
    path prefix from the page that will host the text to ``bibliography.md``).
    Returns an empty string if the README has no body text.
    """
    if not readme_path.exists():
        return ""
    lines = readme_path.read_text(encoding="utf-8").splitlines()
    cleaned, j, n = [], 0, len(lines)
    underline = set("-=~^\"'`#*+.:")
    while j < n:
        line = lines[j]
        st = line.strip()
        if st.startswith(".. "):                      # RST directive / target
            j += 1
            continue
        # a title line immediately followed by an underline row -> skip both
        if (st and j + 1 < n and lines[j + 1].strip()
                and set(lines[j + 1].strip()) <= underline
                and len(lines[j + 1].strip()) >= 3):
            j += 2
            continue
        cleaned.append(line)
        j += 1
    text = re.sub(r'\n{2,}', '\n\n', "\n".join(cleaned)).strip()
    # RST citation [Key2022]_ -> markdown bibliography link
    text = re.sub(
        r'\[([A-Za-z]+\d{4}[a-z]?)\]_',
        lambda m: f'[\\[{m.group(1)}\\]]({bib_depth}bibliography.md#{m.group(1).lower()})',
        text,
    )
    return text

# Nice section titles for the combined Application Examples page (folder -> title).
EXAMPLE_SECTION_TITLES = {
    "11-detection": "Detection",
    "12-filtering": "Filtering",
    "21-polynomials-calculus": "Polynomials Calculus",
    "40-app-changepoint-detection": "Two-Sided Line Models",
    "50-convolution": "Convolution",
    "70-localized-polynomials": "Localized Polynomials",
    "80-nDimensional": "N Dimensional Processing",
}

# Heading for the combined Application Examples page.
EXAMPLES_PAGE_TITLE = "Application and Productive Examples"


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


def build_examples_combined_page(sections, output_dir):
    """Write the single combined Application Examples page.

    One ``## <category>`` heading per folder (so the right-hand TOC becomes the
    page navigation), each optionally followed by an intro sentence (converted
    from the folder's README.rst) and then its gallery table. ``sections`` is an
    ordered list of ``(section_title, folder_name, entries, intro_md)``.
    """
    parts = [f"# {EXAMPLES_PAGE_TITLE}\n"]
    for section_title, folder_name, entries, intro_md in sections:
        parts.append(f"\n## {section_title}\n")
        if intro_md:
            parts.append(f"\n{intro_md}\n")
        parts.append(render_gallery_table(entries, link_prefix=f"{folder_name}/"))
    output_dir.mkdir(parents=True, exist_ok=True)
    index_file = output_dir / "index.md"
    with open(index_file, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    total = sum(len(e) for _, _, e, _ in sections)
    print(f"Generated combined examples page: {index_file} ({total} entries)")


def process_folder(folder_path, folder_parent: str, starting_pattern: str,
                   write_index: bool = True, new_layout: bool = False,
                   readme_bib_depth: str = "../../"):
    # --- Helper for Truncation (old markdown-table layout uses a short blurb) ---
    def truncate_words(text, max_words=10):
        if not text:
            return ""
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]) + "..."

    folder_name = folder_path.name
    section_title = EXAMPLE_SECTION_TITLES.get(
        folder_name, folder_name.replace('-', ' ').replace('_', ' ').title()
    )
    # Everything generated lands under docs/_generated/<category>/<folder_name>/,
    # regardless of where the source .py files are read from.
    target_output_dir = GENERATED_DIR / folder_parent
    output_folder = target_output_dir / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)
    
    gallery_entries = []
    entries = []  # structured rows for the new gallery layout
    # Intro sentence(s) for this section, converted from an RST README if present.
    intro_md = rst_readme_to_md(folder_path / "README.rst", bib_depth=readme_bib_depth)
    pattern = starting_pattern+"*.py"
    py_files = [Path(f) for f in sorted(glob.glob(str(folder_path / pattern)))]
    
    if not py_files:
        print(f"No example files found in {folder_path}")
        return section_title, entries, intro_md

    md_content = f"# {folder_name.replace('-', ' ').replace('_', ' ').title()}\n\n"
    
    # Insert README if exists
    readme_path = folder_path / "README.md"
    if readme_path.exists():
        with open(readme_path, 'r', encoding='utf-8') as f:
            md_content += f.read() + "\n\n---\n\n"

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
        
        if has_show:
            modified_code = code.replace('plt.show()', f'plt.savefig(r"{plot_path}", dpi=150, bbox_inches="tight"); plt.close()')
        elif has_plot_import:
            modified_code = code + f'\nplt.savefig(r"{plot_path}", dpi=150, bbox_inches="tight"); plt.close()'
        
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
            result = subprocess.run([sys.executable, str(temp_file)], check=True, env=env, cwd=str(folder_path), capture_output=True, text=True)
            if result.stdout: console_output = result.stdout.strip()
            if result.stderr:
                console_output = (console_output + "\n\n" + result.stderr.strip()) if console_output else result.stderr.strip()
            if plot_path.exists(): plot_generated = True
        except subprocess.CalledProcessError as e:
            print(f"Error executing {file_name}: {e.stderr.strip() if e.stderr else 'Unknown'}")
            if e.stdout: console_output = e.stdout.strip()
            if e.stderr: console_output = (console_output + "\n\n" + e.stderr.strip()) if console_output else e.stderr.strip()
        except Exception as e:
            print(f"Unexpected error with {file_name}: {e}")
        finally:
            if temp_file.exists(): temp_file.unlink()

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
            
        # --- Create Detailed Page (Plot -> Console -> Code) ---
        detail_md = f"# {title}\n\n{description}\n\n"
        
        if plot_generated:
            detail_md += f"## Plot\n\n![Plot]({plot_filename})\n\n"
        else:
            detail_md += "> **Note:** No graphical output.\n\n"
            
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
              f"-> combined page")

    return section_title, entries, intro_md

# Main logic
if __name__ == "__main__":
    # Ensure the top-level generated dir exists up front, so downstream steps (e.g.
    # the Makefile's `.stamp` marker) always have a home even if nothing is generated.
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    # Scan folders
    # Define paths explicitly.
    #
    # NOTE: `coding/` and `examples/` now live at the repo root (BASE_DIR), they are
    # no longer part of `docs/`. Only their *generated* galleries get written into
    # docs/ (see process_folder). `catalog/` still lives inside docs/.
    coding_folders = [
        BASE_DIR / "coding" / "10-windowed-state-space-filters-basic",
        BASE_DIR / "coding" / "13-backend",
        BASE_DIR / "coding" / "20-polynomials-basics",
    ]
    
    catalog_folders = [
        DOC_DIR / "catalog" / "biosignals",
        DOC_DIR / "catalog" / "generators",
    ]

    examples_folders = [
        BASE_DIR / "examples" / "11-detection",
        BASE_DIR / "examples" / "12-filtering",
        BASE_DIR / "examples" / "21-polynomials-calculus",
        BASE_DIR / "examples" / "40-app-changepoint-detection",
        BASE_DIR / "examples" / "50-convolution",
        BASE_DIR / "examples" / "70-localized-polynomials",
        BASE_DIR / "examples" / "80-nDimensional",
    ]
    
    # Process Coding folders
    for folder in coding_folders:
        if folder.exists():
            print(f"Processing Coding: {folder}")
            process_folder(folder, "coding", "guide-")
        else:
            print(f"Folder not found: {folder}")
            
    # Process Catalog folders. Biosignals uses the new gallery layout (#8); the
    # generators catalog keeps the classic markdown table (its images are linked
    # from the lmlib.utils.generator docstrings).
    for folder in catalog_folders:
        if folder.exists():
            print(f"Processing Catalog: {folder}")
            use_new = folder.name == "biosignals"
            process_folder(folder, "catalog", "example-",
                           new_layout=use_new, readme_bib_depth="../../../")
        else:
            print(f"Folder not found: {folder}")

    # Examples are combined into a single long page (with a right-hand TOC),
    # instead of one sub-page per category.
    example_sections = []
    for folder in examples_folders:
        if folder.exists():
            print(f"Processing Examples: {folder}")
            section_title, entries, intro_md = process_folder(
                folder, "examples", "example-", write_index=False,
                readme_bib_depth="../../",
            )
            example_sections.append((section_title, folder.name, entries, intro_md))
        else:
            print(f"Folder not found: {folder}")

    if example_sections:
        build_examples_combined_page(example_sections, GENERATED_DIR / "examples")
            
    print("Gallery build completed.")
