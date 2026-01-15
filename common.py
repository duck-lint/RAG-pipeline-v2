from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(\|([^\]]+))?\]\]")  # [[Target]] or [[Target|Alias]]
H2_RE = re.compile(r"^\s{0,3}##\s+(.*)$")  # split at H2 for V1

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8", errors="replace"))

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")

def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")

def strip_yaml_frontmatter(md: str) -> Tuple[str, Optional[str]]:
    """
    If file begins with YAML frontmatter delimited by --- ... --- (Obsidian style),
    return (body_without_yaml, yaml_block_text_or_None).
    """
    lines = md.splitlines()
    if not lines or lines[0].strip() != "---":
        return md, None

    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            yaml_block = "\n".join(lines[1:i]).strip()
            body = "\n".join(lines[i + 1 :]).lstrip("\n")
            return body, yaml_block

    return md, None

def parse_yaml_frontmatter(yaml_text: str) -> Dict[str, Any]:
    """
    Real YAML parsing (safe). If YAML is malformed, raise loudly.
    """
    if not yaml_text or not yaml_text.strip():
        return {}

    import yaml  # PyYAML is in your requirements

    data = yaml.safe_load(yaml_text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Frontmatter YAML parsed but did not produce a dict/object.")
    return data

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-") or "section"

def parse_source_date(meta: Dict[str, Any], filename: str) -> Optional[str]:
    """
    Try:
    1) meta['note_creation_date'] if present (YYYY-MM-DD... or ISO)
    2) YYYY-MM-DD in filename
    Return YYYY-MM-DD or None.
    """
    raw = str(meta.get("note_creation_date", "")).strip()
    if raw:
        try:
            if re.match(r"^\d{4}-\d{2}-\d{2}", raw):
                return raw[:10]
            dt = datetime.fromisoformat(raw)
            return dt.date().isoformat()
        except Exception:
            pass

    m = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    if m:
        return m.group(1)

    return None

def parse_date_field(meta: Dict[str, Any], key: str) -> Optional[str]:
    """
    Parse a date field from YAML metadata.
    Accepts:
      - 'YYYY-MM-DD'
      - ISO datetime like '2025-12-31T22:10:00'
    Returns 'YYYY-MM-DD' or None.
    """
    raw = str(meta.get(key, "")).strip()
    if not raw:
        return None

    # Accept YYYY-MM-DD prefix
    if re.match(r"^\d{4}-\d{2}-\d{2}", raw):
        return raw[:10]

    # Try ISO parse
    try:
        dt = datetime.fromisoformat(raw)
        return dt.date().isoformat()
    except Exception:
        return None

def normalize_markdown_light(md: str) -> str:
    """
    V1 normalization:
    - Keep headings (we want them visible and chunkable): '## Title' stays '## Title'
    - Remove YAML (handled elsewhere)
    - Remove code-fence markers but keep code content
    - Replace blockquote markers but keep content
    - Collapse excessive blank lines
    """
    lines = md.splitlines()
    out: List[str] = []
    in_code = False

    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            continue

        if not in_code and line.lstrip().startswith(">"):
            out.append(line.lstrip()[1:].lstrip())
            continue

        out.append(line.rstrip("\n"))

    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    return text

def replace_wikilinks_and_collect(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Replace [[Target|Alias]] -> Alias (or Target).
    Collect out_links: [{"target":..., "alias":...}, ...]
    """
    out_links: List[Dict[str, str]] = []

    def _repl(m: re.Match) -> str:
        target = (m.group(1) or "").strip()
        alias = (m.group(3) or "").strip() if m.group(3) else ""
        rec: Dict[str, str] = {"target": target}
        if alias:
            rec["alias"] = alias
        out_links.append(rec)
        return alias if alias else target

    new_text = WIKILINK_RE.sub(_repl, text)
    return new_text, out_links

HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")

def split_into_sections(body_md: str) -> List[Tuple[str, str, str]]:
    """
    Split into sections by heading.
    Preference rule:
      - If there are any H2 headings, split on H2.
      - Else split on the smallest heading level present (H1 if present, else H3, etc.).
    Returns (anchor, title, section_text) where section_text EXCLUDES the heading line.
    Includes a preamble section for text before first split heading.
    """
    lines = body_md.splitlines()

    levels_present = []
    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            levels_present.append(len(m.group(1)))

    if not levels_present:
        txt = "\n".join(lines).strip() + "\n"
        return [("preamble", "preamble", txt)] if txt.strip() else []

    target_level = 2 if 2 in levels_present else min(levels_present)

    sections: List[Tuple[str, str, List[str]]] = []
    current_title = "preamble"
    current_lines: List[str] = []

    def flush() -> None:
        nonlocal current_title, current_lines
        anchor = slugify(current_title)
        sections.append((anchor, current_title, current_lines))
        current_lines = []

    for line in lines:
        m = HEADING_RE.match(line)
        if m and len(m.group(1)) == target_level:
            # Start new section; DO NOT include the heading line in section_text
            flush()
            current_title = m.group(2).strip()
        else:
            current_lines.append(line)

    flush()

    out: List[Tuple[str, str, str]] = []
    for anchor, title, lns in sections:
        txt = "\n".join(lns).strip() + "\n"
        if anchor == "preamble" and not txt.strip():
            continue
        out.append((anchor, title, txt))
    return out

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
