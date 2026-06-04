"""Auto-generate the API reference pages plus a literate-nav SUMMARY.

Layout (Sphinx-like, shallow left navigation + per-page right-hand TOC):

* A module that defines public **classes** becomes a collapsible nav *section*
  containing:
    - an "Overview" page  (the module docstring + any module-level functions
      and attributes, with a linked summary of the module's classes), and
    - one page per public class.
  Each class page carries its own table of contents (methods, attributes) in
  the right-hand sidebar, so the left navigation only ever nests
  ``module -> class`` (never down to individual methods).

* A module with **no classes** is rendered as a single page, exactly as before.

Public-member discovery uses griffe (the same static-analysis library
mkdocstrings relies on), so the generated structure always matches the
rendered public API (it respects each module's ``__all__``).
"""
from __future__ import annotations

import logging

import griffe
import mkdocs_gen_files

# This module statically loads the package with griffe purely for member
# discovery; mkdocstrings emits any docstring-parsing warnings itself at render
# time, so keep griffe quiet here to avoid duplicate log lines.
logging.getLogger("griffe").setLevel(logging.ERROR)

# (nav section title, [dotted module paths]) -- order defines nav order.
API_STRUCTURE: list[tuple[str, list[str]]] = [
    ("Polynomial", [
        "lmlib.polynomial.poly",
    ]),
    ("State Space", [
        "lmlib.statespace.model",
        "lmlib.statespace.cost",
        "lmlib.statespace.rls",
        "lmlib.statespace.window",
        "lmlib.statespace.segment",
        "lmlib.statespace.trajectory",
        "lmlib.statespace.backend",
        "lmlib.statespace.applications",
    ]),
    ("Utils", [
        "lmlib.utils.generator",
        "lmlib.utils.check",
        "lmlib.utils.colors",
    ]),
]

# Optional thematic grouping of a module's *functions* on its Overview page.
#
# Maps a module's dotted path to an ordered list of ``(section_title, [prefixes])``
# rules. A function is placed in the first section whose *longest* matching name
# prefix it starts with (so e.g. ``mpoly_dilate_ind_*`` is matched by
# ``"mpoly_dilate_ind"`` rather than ``"mpoly_dilate"``). Section order below is
# the order the sections appear on the page; within a section functions keep
# their source order.
#
# Adding a new function requires NO change here as long as its name starts with
# an existing prefix -- it lands in the right section automatically. A function
# matching no prefix is collected into a trailing "Other" section, so it always
# appears (and is a visible nudge to add a prefix rule if desired).
FUNCTION_GROUPS: dict[str, list[tuple[str, list[str]]]] = {
    "lmlib.polynomial.poly": [
        ("Sum of Polynomials", ["poly_sum"]),
        ("Product of Polynomials", ["poly_prod"]),
        ("Square of Polynomials", ["poly_square"]),
        ("Shift of Polynomials", ["poly_shift"]),
        ("Dilation of Polynomials", ["poly_dilation"]),
        ("Integration of Polynomials", ["poly_int"]),
        ("Differentiation of Polynomials", ["poly_diff"]),
        ("Addition of Multivariate Polynomials", ["mpoly_add"]),
        ("Multiplication of Multivariate Polynomials", ["mpoly_multiply"]),
        ("Product of Multivariate Polynomials", ["mpoly_prod"]),
        ("Square of Multivariate Polynomials", ["mpoly_square"]),
        ("Shift of Multivariate Polynomials", ["mpoly_shift"]),
        ("Integration of Multivariate Polynomials", ["mpoly_int"]),
        ("Differentiation of Multivariate Polynomials", ["mpoly_diff"]),
        ("Definite Integration of Multivariate Polynomials", ["mpoly_def_int"]),
        ("Substitution of Multivariate Polynomials", ["mpoly_substitute"]),
        ("Independent Dilation of Multivariate Polynomials", ["mpoly_dilate_ind"]),
        ("Dilation of Multivariate Polynomials", ["mpoly_dilate"]),
        ("Sequences, Matrices and Basis Utilities", [
            "kron_sequence", "extend_basis", "permutation_matrix",
            "commutation_matrix", "remove_redundancy", "mpoly_remove_redundancy",
            "mpoly_transformation", "mpoly_extend",
        ]),
    ],
}

# Title used for functions that match no configured prefix.
UNGROUPED_TITLE = "Other"


def _group_functions(mod: str, func_names: list[str]) -> list[tuple[str, list[str]]]:
    """Bucket ``func_names`` into the sections configured for ``mod``.

    Returns an ordered list of ``(section_title, [names])`` for non-empty
    sections (configured sections first, in config order, then "Other"). If no
    grouping is configured for ``mod`` the result is a single unnamed section
    holding every function (caller treats an empty title as "no heading").
    """
    rules = FUNCTION_GROUPS.get(mod)
    if not rules:
        return [("", list(func_names))]

    # (prefix, title) pairs; longest prefix wins on a tie of multiple matches.
    prefix_title = [(p, title) for title, prefixes in rules for p in prefixes]

    buckets: dict[str, list[str]] = {title: [] for title, _ in rules}
    ungrouped: list[str] = []
    for name in func_names:
        matches = [(len(p), title) for p, title in prefix_title if name.startswith(p)]
        if matches:
            buckets[max(matches)[1]].append(name)
        else:
            ungrouped.append(name)

    out = [(title, buckets[title]) for title, _ in rules if buckets[title]]
    if ungrouped:
        out.append((UNGROUPED_TITLE, ungrouped))
    return out

# Load the package once for static member discovery.
_ROOT = griffe.load("lmlib", submodules=True)


def _module(dotted: str):
    obj = _ROOT
    for part in dotted.split(".")[1:]:
        obj = obj.members[part]
    return obj


def _summary_line(member) -> str:
    """First non-empty line of a member's docstring (its short summary)."""
    doc = getattr(member, "docstring", None)
    if not doc or not doc.value:
        return ""
    for line in doc.value.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _public_split(mod_obj) -> tuple[list[str], list[str]]:
    """Return (class_names, other_member_names) for the module's public API."""
    classes: list[str] = []
    others: list[str] = []
    for name, member in mod_obj.members.items():
        try:
            if not member.is_public:
                continue
        except Exception:
            continue
        kind = member.kind.value
        if kind == "class":
            classes.append(name)
        elif kind in ("function", "attribute"):
            others.append(name)
    return classes, others


nav = mkdocs_gen_files.Nav()

# Keep the existing hand-written Overview page as the first API entry.
nav["Overview"] = "index.md"

for section, modules in API_STRUCTURE:
    for mod in modules:
        mod_obj = _module(mod)
        classes, others = _public_split(mod_obj)

        # ---- modules without classes: one page for the whole module ----------
        if not classes:
            doc_path = f"api/{mod}.md"
            with mkdocs_gen_files.open(doc_path, "w") as fd:
                fd.write(f"# {mod}\n\n")
                fd.write(f"::: {mod}\n")
                fd.write("    options:\n")
                fd.write("      show_root_heading: false\n")
            nav[(section, mod)] = f"{mod}.md"
            continue

        # ---- modules with classes: Overview page + one page per class --------
        overview_path = f"api/{mod}.md"
        grouped = mod in FUNCTION_GROUPS
        if grouped:
            # Split module-level members into functions (grouped into thematic
            # sections below) and attributes (rendered with the module docstring).
            func_names = [n for n in others
                          if mod_obj.members[n].kind.value == "function"]
            attr_names = [n for n in others
                          if mod_obj.members[n].kind.value == "attribute"]

        with mkdocs_gen_files.open(overview_path, "w") as fd:
            fd.write(f"# {mod}\n\n")
            fd.write(f"::: {mod}\n")
            fd.write("    options:\n")
            fd.write("      show_root_heading: false\n")
            # Render only module-level (non-class) members in detail; classes
            # live on their own pages and are surfaced here as a linked summary.
            # For grouped modules the functions are rendered per-section further
            # below (as individual ``::: module.func`` blocks), so the module
            # block here carries only the module docstring + any attributes.
            detail_members = attr_names if grouped else others
            if detail_members:
                fd.write("      members:\n")
                for name in detail_members:
                    fd.write(f"        - {name}\n")
            else:
                fd.write("      members: []\n")
            # Linked summary of the module's classes (each links to its page).
            fd.write("\n## Classes\n\n")
            for cls in classes:
                summary = _summary_line(mod_obj.members[cls])
                if summary:
                    fd.write(f"- [`{cls}`][{mod}.{cls}] \u2014 {summary}\n")
                else:
                    fd.write(f"- [`{cls}`][{mod}.{cls}]\n")

            # Thematic function sections (only for modules with a grouping).
            # Each function is rendered as its own root object so the module
            # docstring is not repeated per section, and so functions nest under
            # their section in the right-hand table of contents.
            if grouped:
                for title, names in _group_functions(mod, func_names):
                    if title:
                        fd.write(f"\n## {title}\n\n")
                    for name in names:
                        fd.write(f"::: {mod}.{name}\n")
                        fd.write("    options:\n")
                        fd.write("      show_root_heading: true\n")
                        fd.write("      show_root_full_path: false\n")
                        fd.write("      heading_level: 3\n")
        nav[(section, mod, "Overview")] = f"{mod}.md"

        for cls in classes:
            cls_path = f"api/{mod}.{cls}.md"
            with mkdocs_gen_files.open(cls_path, "w") as fd:
                fd.write(f"::: {mod}.{cls}\n")
                fd.write("    options:\n")
                fd.write("      show_root_heading: true\n")
                fd.write("      show_root_full_path: false\n")
                fd.write("      heading_level: 1\n")
            nav[(section, mod, cls)] = f"{mod}.{cls}.md"

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as fd:
    fd.writelines(nav.build_literate_nav())
