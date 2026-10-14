"""Read-only data contract for the F09 overview dashboard."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fda_risk import food_country_analysis, sample_overview

DEFAULT_INPUT_PATH = sample_overview.DEFAULT_INPUT_PATH


class DashboardDataError(ValueError):
    """A source or parameter prevents a complete dashboard result."""


def build_dashboard(
    fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]],
    *, min_rate_samples: int = 50,
) -> dict[str, Any]:
    """Reuse the formal F03/F04 definitions on the same snapshot of rows."""
    try:
        overview = sample_overview.build_overview(fieldnames, rows)
        analysis = food_country_analysis.build_analysis(
            fieldnames, rows, top_n=10, min_rate_samples=min_rate_samples,
        )
    except (sample_overview.SampleOverviewError,
            food_country_analysis.FoodCountryAnalysisError) as exc:
        raise DashboardDataError(str(exc)) from exc
    return {
        'overview': overview,
        'analysis': analysis,
        'labclass3_count': sum(item['labclass3'] for item in overview['yearly']),
    }


def load_dashboard(
    path: Path = DEFAULT_INPUT_PATH, *, min_rate_samples: int = 50,
) -> dict[str, Any]:
    """Read once on every invocation; do not cache or mutate the source."""
    try:
        fieldnames, rows = sample_overview.load_samples(path)
    except (sample_overview.SampleOverviewError, OSError) as exc:
        raise DashboardDataError(str(exc)) from exc
    result = build_dashboard(fieldnames, rows, min_rate_samples=min_rate_samples)
    result['source_path'] = str(path.resolve())
    return result
