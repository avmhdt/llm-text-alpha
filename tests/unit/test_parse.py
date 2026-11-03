from __future__ import annotations

from llm_text_alpha.corpus import parse

IXBRL = """<html><head><title>t</title><style>p{}</style></head><body>
<div style="display: none"><ix:header><ix:hidden>HIDDEN FACT</ix:hidden></ix:header></div>
<p>Revenue was <ix:nonFraction name="us-gaap:Revenues">1,234</ix:nonFraction>&nbsp;million.</p>
<script>var x = 1;</script>
<table><tr><td>Item</td><td>2026</td><td></td></tr>
<tr><td><p>Net&nbsp;income</p></td><td>$ 10</td></tr><tr><td> </td></tr></table>
<div>Line one<br>Line   two</div>


<p>End</p></body></html>"""


def test_html_with_inline_xbrl_and_tables() -> None:
    text = parse.html_to_text(IXBRL)
    assert "HIDDEN" not in text and "var x" not in text and "p{}" not in text
    assert "Revenue was 1,234 million." in text
    assert "Item | 2026" in text
    assert "Net income | $ 10" in text
    assert "Line one\nLine two" in text
    assert "\n\n\n" not in text
    assert text.endswith("End")


def test_sgml_wrapper_lines_are_removed() -> None:
    markup = "<DOCUMENT>\n<TYPE>EX-99.1\n<SEQUENCE>2\n<FILENAME>x.htm\n<TEXT>\n<p>Press release</p>"
    assert parse.html_to_text(markup) == "Press release"


def test_plain_text_documents() -> None:
    raw = b"<PAGE>  1\n   EXHIBIT 99.1\r\n\r\n\r\nResults   improved.\n<PAGE> 2\n"
    assert parse.extract_text(raw, "ex99.txt") == "EXHIBIT 99.1\n\nResults improved."
    html_in_txt = b"<html><body><p>Hello</p></body></html>"
    assert parse.extract_text(html_in_txt, "d.txt") == "Hello"


def test_decoding_falls_back_to_cp1252() -> None:
    assert parse.decode("café".encode()) == "café"
    assert parse.decode(b"\x93quoted\x94") == "“quoted”"


def test_empty_and_fallback() -> None:
    assert parse.html_to_text("") == ""
    assert parse.html_to_text("<html><body></body></html>") == ""
    assert parse._lxml_text("<html><body><p>Kept</p><script>x</script></body></html>") == "Kept"
    assert parse._lxml_text("") == ""


def test_normalize() -> None:
    assert parse.normalize(" a \u00a0 b \r\n\r\n\r\n c ") == "a b\n\nc"
