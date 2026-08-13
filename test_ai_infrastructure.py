from __future__ import annotations

from typing import Any

import ai_infrastructure
from config import Config


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _Session:
    def get(self, _url: str, *, params: dict[str, Any], **_kwargs: Any) -> _Response:
        results = []
        prices = {"DELL": 112.0, "HPE": 110.0, "SMCI": 109.0}
        for symbol in str(params["symbols"]).split(","):
            current = prices.get(symbol, 101.0)
            results.append({
                "symbol": symbol,
                "response": [{
                    "meta": {
                        "regularMarketPrice": current,
                        "chartPreviousClose": 100.0,
                        "regularMarketTime": 1_786_560_000,
                        "shortName": symbol,
                    },
                    "timestamp": list(range(21)),
                    "indicators": {"quote": [{"close": [100.0] * 20 + [current]}]},
                }],
            })
        return _Response({"spark": {"result": results}})


def test_ai_infrastructure_scanner_finds_server_rotation_and_preserves_execution_boundary() -> None:
    report = ai_infrastructure.build_ai_infrastructure_context(force_refresh=True, session=_Session())

    assert report["active"] is True
    assert report["summary"]["covered_count"] == report["summary"]["universe_count"]
    assert report["sectors"][0]["key"] == "servers_racks"
    server_members = {row["coin"]: row for row in report["sectors"][0]["members"]}
    assert server_members["DELL"]["coverage_mode"] == "EXECUTABLE"
    assert server_members["HPE"]["coverage_mode"] == "REFERENCE_ONLY"
    assert "DELL" in report["focus_symbols"]
    assert "HPE" in report["focus_symbols"]


def test_reference_only_ai_names_inform_analysis_but_cannot_enter_execution_universe() -> None:
    cfg = Config()

    assert "HPE" in cfg.trading.analysis_coins
    assert "HPE" in cfg.trading.analysis_only_coins
    assert "HPE" not in cfg.trading.coins
    assert "DELL" in cfg.trading.analysis_coins
    assert "DELL" not in cfg.trading.analysis_only_coins
    assert "DELL" in cfg.trading.coins

