"""
MarkdownParser — converts Notion markdown exports into QAPair objects.

Parsing rules (state machine over lines):
  - Line matches ^## or ^### heading  → update current section
  - Line matches ^- (no indent)       → open a new QAPair (question)
  - Line has leading whitespace        → accumulate into current answer
  - Image syntax ![](<path>)           → set has_diagram=True, record img path
  - Blank lines / "---" / "# " titles → skip

No project-level imports — this module is a leaf dependency.
"""

import os
import re
from pathlib import Path

from ml.models import QAPair


_IMAGE_RE = re.compile(r"!\[.*?\]\((.+?)\)")
_SECTION_HEADING_RE = re.compile(r"^#{2,3}\s+(.+)")
_TOP_BULLET_RE = re.compile(r"^- (.+)")
_INDENTED_RE = re.compile(r"^[ \t]+(.+)")


def _normalise_section(heading: str) -> str:
    """Map a heading string to a canonical section tag."""
    lower = heading.lower()
    if "main idea" in lower:
        return "main_ideas"
    if "exercise" in lower:
        return "exercises"
    return ""


def _page_title_from_path(path: Path) -> str:
    """
    Derive a human-readable page title from the .md filename stem.

    Notion exports use long slugs like:
      "Decision Trees c8081b39bf374f5593433767d008a689 d819...md"
    We strip the trailing hex ID portions (32-char hex blocks) and
    clean up the remainder.
    """
    stem = path.stem
    # Remove trailing hex IDs (Notion appends them separated by spaces)
    cleaned = re.sub(r"\s+[0-9a-f]{32}$", "", stem).strip()
    cleaned = re.sub(r"\s+[0-9a-f]{24,}$", "", cleaned).strip()
    return cleaned


class MarkdownParser:
    """Parse all .md files under a directory into a flat list of QAPair objects."""

    def parse_file(self, filepath: Path) -> list[QAPair]:
        """
        State-machine parser for a single Notion markdown export file.

        Returns a list of QAPair objects found in the file.
        Skips pairs where either question or answer is empty after stripping.
        Logs a warning (to stdout) for malformed indented blocks encountered
        before any question bullet has been seen.
        """
        page_title = _page_title_from_path(filepath)
        page_slug = page_title.lower().replace(" ", "_")

        pairs: list[QAPair] = []
        current_section: str = ""
        open_pair: QAPair | None = None

        with open(filepath, encoding="utf-8") as fh:
            lines = fh.readlines()

        for line_num, raw_line in enumerate(lines, start=1):
            line = raw_line.rstrip("\n")

            # Section heading
            heading_match = _SECTION_HEADING_RE.match(line)
            if heading_match:
                if open_pair and open_pair.answer.strip():
                    pairs.append(open_pair)
                    open_pair = None
                elif open_pair:
                    open_pair = None  # discard empty-answer pair
                current_section = _normalise_section(heading_match.group(1))
                continue

            # Top-level bullet → new question
            bullet_match = _TOP_BULLET_RE.match(line)
            if bullet_match:
                if open_pair and open_pair.answer.strip():
                    pairs.append(open_pair)
                elif open_pair:
                    pass  # discard empty-answer pair silently

                question_text = bullet_match.group(1).strip()
                open_pair = QAPair(
                    id=f"{page_slug}_{line_num}",
                    question=question_text,
                    answer="",
                    page_title=page_title,
                    section=current_section,
                    has_diagram=False,
                    img_paths=[],
                )
                continue

            # Indented content → accumulate into answer
            indented_match = _INDENTED_RE.match(line)
            if indented_match and open_pair is not None:
                content = indented_match.group(1).strip()
                img_match = _IMAGE_RE.search(line)
                if img_match:
                    open_pair.has_diagram = True
                    img_path = img_match.group(1)
                    open_pair.img_paths.append(img_path)
                    fname = os.path.basename(img_path)
                    if open_pair.answer:
                        open_pair.answer += f"\n[diagram: {fname}]"
                    else:
                        open_pair.answer = f"[diagram: {fname}]"
                else:
                    if open_pair.answer:
                        open_pair.answer += "\n" + content
                    else:
                        open_pair.answer = content
                continue

            # Blank lines, "---", and "# " title lines are skipped

        # Flush the last open pair
        if open_pair and open_pair.answer.strip():
            pairs.append(open_pair)

        return pairs

    def parse(self, directory: str = "data/assets/notion") -> list[QAPair]:
        """
        Walk all .md files under `directory` recursively and return every
        QAPair found.  The top-level index file (if present) is included;
        it typically contains no Q&A bullets and will produce zero pairs.
        """
        all_pairs: list[QAPair] = []
        root = Path(directory)

        for md_file in sorted(root.rglob("*.md")):
            try:
                file_pairs = self.parse_file(md_file)
                all_pairs.extend(file_pairs)
            except Exception as exc:
                print(f"[MarkdownParser] Warning: skipping {md_file} — {exc}")

        return all_pairs
