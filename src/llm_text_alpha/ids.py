"""Identifier helpers shared by the corpus and the market-data panels."""

from __future__ import annotations

import polars as pl


def permno_of(security_id: pl.Expr) -> pl.Expr:
    """The integer after the prefix: Sharadar's permaticker for ``SHR:<n>`` ids (section 7b).

    eqpit keys securities as ``SHR:<permaticker>``, so this is the provider id section 7b
    allows in place of a CRSP PERMNO.
    """
    return security_id.str.split(":").list.last().cast(pl.Int64, strict=False)
