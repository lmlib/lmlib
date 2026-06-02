# scripts/build_gallery.py
import os
import sys
import glob
import subprocess
import shutil
import re
from pathlib import Path
import matplotlib
# Force non-interactive backend immediately to prevent GUI issues in headless environments
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# Adjust paths
BASE_DIR = Path(__file__).parent.parent
DOC_DIR = BASE_DIR / "docs" 
OUTPUT_DIR = DOC_DIR / "generated_galleries"

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

def process_folder(folder_path, folder_parent: str, starting_pattern: str):
    # --- Helper for Truncation ---
    def truncate_words(text, max_words=10):
        if not text:
            return ""
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]) + "..."

    folder_name = folder_path.name
    target_output_dir = DOC_DIR / folder_parent / "_generated_galleries"
    output_folder = target_output_dir / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)
    
    gallery_entries = []
    pattern = starting_pattern+"*.py"
    py_files = [Path(f) for f in sorted(glob.glob(str(folder_path / pattern)))]
    
    if not py_files:
        print(f"No example files found in {folder_path}")
        return

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
        
        temp_file = output_folder / f"_temp_{base_name}.py"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(modified_code)
            
        plot_generated = False
        console_output = ""
        
        try:
            result = subprocess.run([sys.executable, str(temp_file)], check=True, env=env, cwd=str(output_folder), capture_output=True, text=True)
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

    index_file = output_folder / "index.md"
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
        
    print(f"Generated: {index_file} ({len(gallery_entries)} entries)")

# Main logic
if __name__ == "__main__":
    # Scan folders
    # Define paths explicitly
    coding_folders = [
        DOC_DIR / "coding" / "10-windowed-state-space-filters-basic",
        DOC_DIR / "coding" / "13-backend",
        DOC_DIR / "coding" / "20-polynomials-basics",
    ]
    
    catalog_folders = [
        DOC_DIR / "catalog" / "biosignals",
        DOC_DIR / "catalog" / "generators",
    ]

    examples_folders = [
        DOC_DIR / "examples" / "11-detection",
        DOC_DIR / "examples" / "12-filtering",
        DOC_DIR / "examples" / "21-polynomials-calculus",
        DOC_DIR / "examples" / "40-app-changepoint-detection",
        DOC_DIR / "examples" / "50-convolution",
        DOC_DIR / "examples" / "70-localized-polynomials",
        DOC_DIR / "examples" / "80-nDimensional",
    ]
    
    # Process Coding folders
    for folder in coding_folders:
        if folder.exists():
            print(f"Processing Coding: {folder}")
            process_folder(folder, "coding", "guide-")
        else:
            print(f"Folder not found: {folder}")
            
    # Process Catalog folders
    for folder in catalog_folders:
        if folder.exists():
            print(f"Processing Catalog: {folder}")
            process_folder(folder, "catalog", "example-")
        else:
            print(f"Folder not found: {folder}")

    for folder in examples_folders:
        if folder.exists():
            print(f"Processing Examples: {folder}")
            process_folder(folder, "examples", "example-")
        else:
            print(f"Folder not found: {folder}")
            
    print("Gallery build completed.")