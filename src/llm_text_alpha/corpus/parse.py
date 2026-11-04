"""Text extraction from EDGAR HTML, inline XBRL and plain-text documents (FR-02, DQ-13).

Scripts, styles, the hidden inline-XBRL header and anything styled ``display:none`` are
dropped; inline-XBRL tags are unwrapped so their numbers stay in the text. Each table row
becomes one pipe-delimited line. Block elements start new lines, runs of spaces collapse,
and blank lines are kept to one.
"""

from __future__ import annotations

import re
import unicodedata

from selectolax.lexbor import LexborHTMLParser, LexborNode

_DROP = frozenset({"script", "style", "head", "title", "noscript", "ix:header", "xml"})
_BLOCK = frozenset(
    {
        "p",
        "div",
        "br",
        "li",
        "ul",
        "ol",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "tr",
        "table",
        "section",
        "center",
        "blockquote",
        "pre",
        "hr",
        "dl",
        "dt",
        "dd",
        "body",
        "html",
        "article",
        "header",
        "footer",
    }
)
_HIDDEN = re.compile(r"display\s*:\s*none", re.IGNORECASE)
_SPACES = re.compile("[ \t\f\v\u00a0\u2000-\u200b\u202f\u205f\u3000]+")
_PAGE = re.compile(
    r"^\s*<(/?PAGE|/?TEXT|/?DOCUMENT|TYPE|SEQUENCE|FILENAME|DESCRIPTION)>.*$",
    re.IGNORECASE | re.MULTILINE,
)
_HTML_HINT = re.compile(r"<\s*(html|body|div|p|table|font)\b", re.IGNORECASE)


def decode(raw: bytes) -> str:
    """UTF-8 when it decodes cleanly, else Windows-1252, the usual EDGAR fallback."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def _hidden(node: LexborNode) -> bool:
    style = node.attributes.get("style") if node.attributes else None
    return bool(style and _HIDDEN.search(style))


def _cell_text(node: LexborNode) -> str:
    parts: list[str] = []
    _walk(node, parts, in_cell=True)
    return _SPACES.sub(" ", "".join(parts)).strip()


def _table_lines(table: LexborNode) -> list[str]:
    lines: list[str] = []
    for row in table.css("tr"):
        cells = [_cell_text(cell) for cell in row.iter() if cell.tag in ("td", "th")]
        cells = [cell for cell in cells if cell]
        if cells:
            lines.append(" | ".join(cells))
    return lines


def _walk(root: LexborNode, out: list[str], in_cell: bool = False) -> None:
    stack: list[LexborNode | str] = [root]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            out.append(item)
            continue
        tag = item.tag or ""
        if tag == "-text":
            out.append(item.text_content or "")
            continue
        if tag in _DROP or tag.startswith("-") or _hidden(item):
            continue
        if tag == "table" and not in_cell:
            out.append("\n" + "\n".join(_table_lines(item)) + "\n")
            continue
        separator = " " if in_cell else "\n"
        if tag == "br":
            out.append(separator)
            continue
        block = tag in _BLOCK
        if block:
            out.append(separator)
            stack.append(separator)
        children = list(item.iter(include_text=True))
        stack.extend(reversed(children))


def normalize(text: str) -> str:
    """NFC, one space between words, trimmed lines, at most one blank line in a row."""
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    kept: list[str] = []
    for line in lines:
        if line or (kept and kept[-1]):
            kept.append(line)
    return "\n".join(kept).strip()


def html_to_text(markup: str) -> str:
    """Readable text from an HTML or inline-XBRL document.

    SGML wrapper lines (``<TYPE>``, ``<SEQUENCE>``, ...) that EDGAR leaves in some files are
    removed first. selectolax's lexbor parser is primary; lxml takes over if lexbor raises
    or finds no text.
    """
    markup = _PAGE.sub("", markup)
    try:
        tree = LexborHTMLParser(markup)
        root = tree.body or tree.root
        parts: list[str] = []
        if root is not None:
            _walk(root, parts)
        text = normalize("".join(parts))
    except Exception:  # lexbor is a C parser; any failure falls back to lxml
        text = ""
    if not text and _HTML_HINT.search(markup):
        return _lxml_text(markup)
    return text


def _lxml_text(markup: str) -> str:
    import lxml.etree
    import lxml.html

    try:
        document = lxml.html.document_fromstring(markup)
    except (ValueError, lxml.etree.ParserError):
        return ""
    for bad in document.xpath("//script|//style|//*[local-name()='header']"):
        bad.drop_tree()
    return normalize(document.text_content())


def plain_to_text(text: str) -> str:
    """Text from a plain-text EDGAR document, without SGML wrapper lines."""
    return normalize(_PAGE.sub("", text))


def extract_text(raw: bytes, filename: str) -> str:
    """Text of one document file, by extension; ``.txt`` files holding HTML are parsed."""
    text = decode(raw)
    if filename.lower().endswith(".txt") and not _HTML_HINT.search(text[:5000]):
        return plain_to_text(text)
    return html_to_text(text)
