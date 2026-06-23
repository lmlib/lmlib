#!/usr/bin/env python3
"""Fetch GitHub Releases and generate docs/news.md"""
import os
import re
import requests
import datetime
from zoneinfo import ZoneInfo

# Fallback to your local repository name if running locally outside of GitHub Actions
REPO = os.environ.get("GITHUB_REPOSITORY", "") or "lmlib/lmlib"  # e.g. "lmlib/lmlib"
OUTPUT = os.path.join(os.path.dirname(__file__), "..", "news.md")

def format_date(iso_date):
    """Convert 2026-06-08T00:00:00Z -> June 08, 2026"""
    dt = datetime.datetime.strptime(iso_date[:10], "%Y-%m-%d")
    return dt.strftime("%B %d, %Y")

def main():
    if not REPO:
        print("ERROR: GITHUB_REPOSITORY not set")
        exit(1)

    url = f"https://api.github.com/repos/{REPO}/releases?per_page=50"
    headers = {"Accept": "application/vnd.github+json"}
    
    # Use GITHUB_TOKEN if available (avoids rate limits in CI)
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    releases = resp.json()

    if not releases:
        print("No releases found.")
        return
    now = datetime.datetime.now(ZoneInfo("Europe/Zurich"))

    lines = [
        "# News\n",
        f"*Updated at: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}*\n",
    ]
    
    for release in releases:
        tag = release["tag_name"]
        date = format_date(release["published_at"])
        body = (release.get("body") or "").strip()
        body = re.sub(r'([^\n])\n([-*+] )', r'\1\n\n\2', body)

        lines.append(f"## {tag} released\n")
        lines.append(f"{date} — {tag} has been released!\n")
        if body:
            lines.append(f"{body}\n")
        lines.append("---\n")

    with open(OUTPUT, "w") as f:
        f.write("\n".join(lines))

    print(f"Generated {OUTPUT} with {len(releases)} releases.")

main()
