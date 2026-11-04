"""Spend tracker with a hard stop at ``LLM_BUDGET_USD`` (FR-12, NFR-06).

Before any paid request, a provider reserves its worst-case cost. A reservation that would
take committed spend (settled spend plus open reservations) above the budget raises
:class:`BudgetExceededError` and writes nothing. After the results arrive the reservation is
settled with the actual cost, which closes it. The ledger lives in DuckDB, so the cap is
cumulative across every run of the project.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import yaml

from llm_text_alpha.scoring.llm.base import TokenUsage

MAX_PRICE_TABLE_AGE_DAYS = 45


class BudgetExceededError(RuntimeError):
    """A reservation would take committed spend above the budget; nothing was sent."""


class StalePriceTableError(RuntimeError):
    """``configs/llm_prices.yaml`` was retrieved too long ago to trust (section 7f)."""


@dataclass(frozen=True)
class Rates:
    """USD per million tokens."""

    input: float
    cache_write: float
    cache_read: float
    output: float


@dataclass(frozen=True)
class ModelPrices:
    """Standard rates, and long-prompt rates above ``long_threshold`` input tokens."""

    standard: Rates
    long: Rates | None = None
    long_threshold: int | None = None

    def rates(self, prompt_tokens: int) -> Rates:
        """The rate card a request with ``prompt_tokens`` input tokens pays."""
        if self.long and self.long_threshold is not None and prompt_tokens > self.long_threshold:
            return self.long
        return self.standard


def _rates(raw: dict[str, Any]) -> Rates:
    return Rates(
        input=float(raw["input"]),
        cache_write=float(raw["cache_write_5m"]),
        cache_read=float(raw["cache_read"]),
        output=float(raw["output"]),
    )


@dataclass(frozen=True)
class PriceTable:
    """Prices from ``configs/llm_prices.yaml``, with the date they were read off the source."""

    models: dict[str, ModelPrices]
    batch_discount: float
    retrieved: date
    source: str

    @classmethod
    def load(
        cls, path: Path, today: date | None = None, max_age_days: int = MAX_PRICE_TABLE_AGE_DAYS
    ) -> PriceTable:
        """Read the table; refuse it when it is older than ``max_age_days``."""
        raw = yaml.safe_load(path.read_text("utf-8"))
        retrieved = date.fromisoformat(str(raw["retrieved"]))
        age = ((today or datetime.now(UTC).date()) - retrieved).days
        if age > max_age_days:
            raise StalePriceTableError(
                f"{path} was retrieved {age} days ago (limit {max_age_days}); re-verify the prices"
            )
        models = {
            model_id: ModelPrices(
                standard=_rates(entry),
                long=_rates(entry["long"]) if "long" in entry else None,
                long_threshold=entry.get("long_context_threshold"),
            )
            for model_id, entry in raw["models"].items()
        }
        return cls(models, float(raw["batch_discount"]), retrieved, str(raw["source"]))

    def prices(self, model_id: str) -> ModelPrices:
        """The model's prices; a model without a row cannot be priced, so cannot run."""
        if model_id not in self.models:
            raise KeyError(f"no price row for {model_id} in the price table")
        return self.models[model_id]

    def cost(self, model_id: str, usage: TokenUsage, batch: bool) -> float:
        """USD for one response's ``usage``."""
        prompt = usage.input_tokens + usage.cache_write_tokens + usage.cache_read_tokens
        rates = self.prices(model_id).rates(prompt)
        usd = (
            usage.input_tokens * rates.input
            + usage.cache_write_tokens * rates.cache_write
            + usage.cache_read_tokens * rates.cache_read
            + usage.output_tokens * rates.output
        ) / 1_000_000
        return usd * (1 - self.batch_discount) if batch else usd

    def worst_case(
        self, model_id: str, input_tokens: int, max_output_tokens: int, batch: bool
    ) -> float:
        """Upper bound for one request: every input token at the cache-write rate (the higher
        of input and cache write) and the full ``max_output_tokens``."""
        rates = self.prices(model_id).rates(input_tokens)
        usd = (
            input_tokens * max(rates.input, rates.cache_write) + max_output_tokens * rates.output
        ) / 1_000_000
        return usd * (1 - self.batch_discount) if batch else usd


@dataclass(frozen=True)
class Reservation:
    """An open hold on the budget."""

    reservation_id: str
    run_id: str
    model_id: str
    usd: float


_SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger (
    entry_id VARCHAR PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    run_id VARCHAR NOT NULL,
    model_id VARCHAR NOT NULL,
    kind VARCHAR NOT NULL,
    usd DOUBLE NOT NULL,
    committed_usd DOUBLE NOT NULL,
    budget_usd DOUBLE NOT NULL,
    reservation_id VARCHAR NOT NULL
)
"""


class SpendTracker:
    """The ledger table ``ledger`` in ``spend.duckdb`` (section 7b).

    ``kind`` is ``reserve`` (worst-case hold), ``settle`` (actual spend; closes the hold) or
    ``release`` (closes the hold with no spend). ``committed_usd`` on each row is settled
    spend plus open holds after that row.
    """

    def __init__(self, ledger_path: Path, budget_usd: float, prices: PriceTable) -> None:
        self.ledger_path = ledger_path
        self.budget_usd = budget_usd
        self.prices = prices
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(_SCHEMA)

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.ledger_path))

    @staticmethod
    def _committed(connection: duckdb.DuckDBPyConnection) -> float:
        row = connection.execute(
            """
            SELECT
                COALESCE(SUM(usd) FILTER (WHERE kind = 'settle'), 0)
                + COALESCE(SUM(usd) FILTER (
                    WHERE kind = 'reserve' AND reservation_id NOT IN (
                        SELECT reservation_id FROM ledger WHERE kind IN ('settle', 'release')
                    )
                ), 0)
            FROM ledger
            """
        ).fetchone()
        return float(row[0]) if row else 0.0

    def committed_usd(self) -> float:
        """Settled spend plus open reservations."""
        with self._connect() as connection:
            return self._committed(connection)

    def _append(
        self,
        connection: duckdb.DuckDBPyConnection,
        kind: str,
        run_id: str,
        model_id: str,
        usd: float,
        reservation_id: str,
        committed: float,
    ) -> None:
        connection.execute(
            "INSERT INTO ledger VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                str(uuid.uuid4()),
                datetime.now(UTC),
                run_id,
                model_id,
                kind,
                usd,
                committed,
                self.budget_usd,
                reservation_id,
            ],
        )

    def reserve(self, run_id: str, model_id: str, usd: float) -> Reservation:
        """Hold ``usd`` against the budget, or raise :class:`BudgetExceededError`."""
        if usd < 0:
            raise ValueError("a reservation cannot be negative")
        with self._connect() as connection:
            connection.execute("BEGIN TRANSACTION")
            committed = self._committed(connection)
            if committed + usd > self.budget_usd:
                connection.execute("ROLLBACK")
                raise BudgetExceededError(
                    f"reserving ${usd:.4f} would commit ${committed + usd:.4f} of the "
                    f"${self.budget_usd:.2f} budget"
                )
            reservation = Reservation(str(uuid.uuid4()), run_id, model_id, usd)
            self._append(
                connection,
                "reserve",
                run_id,
                model_id,
                usd,
                reservation.reservation_id,
                committed + usd,
            )
            connection.execute("COMMIT")
        return reservation

    def _close(self, reservation: Reservation, kind: str, usd: float) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN TRANSACTION")
            closed = connection.execute(
                "SELECT COUNT(*) FROM ledger "
                "WHERE reservation_id = ? AND kind IN ('settle', 'release')",
                [reservation.reservation_id],
            ).fetchone()
            if closed and closed[0]:
                connection.execute("ROLLBACK")
                raise ValueError(f"reservation {reservation.reservation_id} is already closed")
            committed = self._committed(connection) - reservation.usd + usd
            self._append(
                connection,
                kind,
                reservation.run_id,
                reservation.model_id,
                usd,
                reservation.reservation_id,
                committed,
            )
            connection.execute("COMMIT")

    def settle(
        self, reservation: Reservation, actual: Sequence[TokenUsage], batch: bool = False
    ) -> float:
        """Close the hold with the actual cost of ``actual``; return that cost."""
        usd = sum(self.prices.cost(reservation.model_id, usage, batch) for usage in actual)
        self._close(reservation, "settle", usd)
        return usd

    def settle_usd(self, reservation: Reservation, usd: float) -> float:
        """Close the hold with an actual cost already computed in USD."""
        self._close(reservation, "settle", usd)
        return usd

    def release(self, reservation: Reservation) -> None:
        """Close the hold with no spend."""
        self._close(reservation, "release", 0.0)

    def open_reservations(self) -> list[Reservation]:
        """Holds not yet settled or released, e.g. batches still in flight after a crash."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT reservation_id, run_id, model_id, usd FROM ledger
                WHERE kind = 'reserve' AND reservation_id NOT IN (
                    SELECT reservation_id FROM ledger WHERE kind IN ('settle', 'release')
                ) ORDER BY ts
                """
            ).fetchall()
        return [Reservation(str(r[0]), str(r[1]), str(r[2]), float(r[3])) for r in rows]

    def totals(self, by: str = "model") -> list[tuple[str, float]]:
        """Settled spend grouped by ``model`` or ``run``."""
        column = {"model": "model_id", "run": "run_id"}[by]
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT {column}, SUM(usd) FROM ledger WHERE kind = 'settle' "
                f"GROUP BY {column} ORDER BY {column}"
            ).fetchall()
        return [(str(r[0]), float(r[1])) for r in rows]

    def n_entries(self) -> int:
        """Rows in the ledger."""
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM ledger").fetchone()
        return int(row[0]) if row else 0
