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
DOC_DIR = BASE_DIR / "lmlib_doc" / "docs" 
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

def process_folder(folder_path, folder_parent: str):
    folder_name = folder_path.name
    # Determine output directory based on parent type
    target_output_dir = DOC_DIR / folder_parent / "_generated_galleries"
        
    output_folder = target_output_dir / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # List for gallery entries
    gallery_entries = []
    
    # Determine file pattern based on type
    # Coding: example-ex101.0.py (hyphen)
    # Catalog: example_eecg_baseline.py (underscore)
    pattern = "example-*.py"
    
    # FIX: glob.glob() returns strings, convert to Path
    py_files = [Path(f) for f in sorted(glob.glob(str(folder_path / pattern)))]
    
    if not py_files:
        print(f"No example files found in {folder_path}")
        return

    md_content = f"# {folder_name.replace('-', ' ').replace('_', ' ').title()}\n\n"
    
    # Insert README.md if it exists
    readme_path = folder_path / "README.md"
    if readme_path.exists():
        with open(readme_path, 'r', encoding='utf-8') as f:
            md_content += f.read()
            md_content += "\n\n---\n\n"
    
    # Table headers
    md_content += "| Example | Plot |\n"
    md_content += "| :--- | :--- |\n"
    
    for py_file in py_files:
        file_name = py_file.name
        # Normalize base_name: remove extension and replace hyphens/underscores if needed for consistency
        base_name = file_name.replace('.py', '')
        
        # 1. Plot generation setup
        plot_filename = f"{base_name}.png"
        plot_path = output_folder / plot_filename
        
        # Set backend to Agg to enforce headless rendering
        env = os.environ.copy()
        env['MPLBACKEND'] = 'Agg'
        
        with open(py_file, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Check if plt.show() exists
        has_show = 'plt.show()' in code
        has_plot_import = 'import matplotlib' in code or 'import pyplot' in code
        
        modified_code = code
        
        if has_show:
            # Replace plt.show() with plt.savefig and plt.close()
            modified_code = code.replace('plt.show()', f'plt.savefig(r"{plot_path}", dpi=150, bbox_inches="tight"); plt.close()')
        elif has_plot_import:
            # If no show() but import exists, try to save anyway
            modified_code = code + f'\nplt.savefig(r"{plot_path}", dpi=150, bbox_inches="tight"); plt.close()'
        else:
            # No plot import found, skip image generation completely
            pass

        # Write temporary file
        temp_file = output_folder / f"_temp_{base_name}.py"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(modified_code)
            
        plot_generated = False
        console_output = ""
        
        try:
            # Execute with capture_output=True to get print statements
            result = subprocess.run(
                [sys.executable, str(temp_file)], 
                check=True, 
                env=env, 
                cwd=str(output_folder), 
                capture_output=True, 
                text=True
            )
            
            # Capture standard output (print statements)
            if result.stdout:
                console_output = result.stdout.strip()
            
            # Capture standard error (warnings/errors) if any
            if result.stderr:
                if console_output:
                    console_output += "\n\n" + result.stderr.strip()
                else:
                    console_output = result.stderr.strip()

            if plot_path.exists():
                plot_generated = True
            else:
                pass
                
        except subprocess.CalledProcessError as e:
            stderr_msg = e.stderr if e.stderr else "Unknown error"
            print(f"Error executing {file_name}: {stderr_msg.strip()}")
            # Capture output even on error
            if e.stdout:
                console_output = e.stdout.strip()
            if e.stderr:
                if console_output:
                    console_output += "\n\n" + e.stderr.strip()
                else:
                    console_output = e.stderr.strip()
        except Exception as e:
            print(f"Unexpected error with {file_name}: {e}")
        finally:
            # Cleanup temporary file regardless of success/failure
            if temp_file.exists():
                temp_file.unlink()

        # Extract Title and Description
        docstring_match = re.search(r'"""(.*?)"""', code, re.DOTALL)
        docstring = docstring_match.group(1) if docstring_match else ""
        title, description, description_flat = extract_title_and_description(docstring)
        
        # --- Create Gallery Entry for Index ---
        # Use flat description for the table cell
        safe_description = description_flat.replace("|", "&#124;")
        table_entry = f"{title}<br><small>{safe_description}</small>"
        
        if plot_generated:
            md_content += f"| [{table_entry}]({base_name}.md) | ![Plot]({plot_filename}) |\n"
        else:
            md_content += f"| [{table_entry}]({base_name}.md) | *No Plot* |\n"
            
        # --- Create Detailed Page ---
        # Order: Title → Description → Code → Console Output → Plot
        
        detail_md = f"# {title}\n\n"
        
        # 1. Description (full formatted version with lists and bold)
        detail_md += f"{description}\n\n"
        
        # 2. Code
        detail_md += "## Code\n\n"
        detail_md += "```python\n"
        detail_md += code
        detail_md += "\n```\n\n"
        
        # 3. Console Output (if any print statements or errors occurred)
        if console_output:
            detail_md += "## Console Output\n\n"
            detail_md += "```text\n"
            detail_md += console_output
            detail_md += "\n```\n\n"
        
        # 4. Plot (if available)
        if plot_generated:
            detail_md += "## Plot\n\n"
            detail_md += f"![Plot]({plot_filename})\n\n"
        else:
            detail_md += "> **Note:** This example does not generate a graphical output.\n\n"
        
        detail_file = output_folder / f"{base_name}.md"
        with open(detail_file, 'w', encoding='utf-8') as f:
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
        #DOC_DIR / "coding" / "13-backend",
        #DOC_DIR / "coding" / "20-polynomials-basics",
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
            process_folder(folder, "coding")
        else:
            print(f"Folder not found: {folder}")
            
    # Process Catalog folders
    for folder in catalog_folders:
        if folder.exists():
            print(f"Processing Catalog: {folder}")
            process_folder(folder, "catalog")
        else:
            print(f"Folder not found: {folder}")

    for folder in examples_folders:
        if folder.exists():
            print(f"Processing Examples: {folder}")
            process_folder(folder, "examples")
        else:
            print(f"Folder not found: {folder}")
            
    print("Gallery build completed.")