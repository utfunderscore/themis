"""Plain-text tables for training and evaluation metrics."""

from collections.abc import Mapping

# Throughput and bookkeeping, not model quality.
_SKIPPED = ("runtime", "samples_per_second", "steps_per_second", "epoch")


def format_run_metrics(metrics: Mapping[str, object]) -> str:
    """Tabulate a run's metrics.json: one column per evaluated split, then the training summary."""
    splits = {name: _strip(value, name) for name, value in metrics.items()
              if name != "train" and isinstance(value, Mapping)}
    rows = [[name, *(_number(values.get(name)) for values in splits.values())]
            for name in _metric_names(splits.values())]
    lines = [_table(["metric", *splits], rows)]
    train = metrics.get("train")
    if isinstance(train, Mapping) and "train_loss" in train:
        lines.append(f"train loss {train['train_loss']:.4f} over {train.get('epoch', 0):g} epochs")
    if metrics.get("best_checkpoint"):
        lines.append(f"best checkpoint: {metrics['best_checkpoint']}")
    return "\n".join(lines)


def format_comparison(
    before: Mapping[str, Mapping[str, float]],
    after: Mapping[str, Mapping[str, float]],
    *,
    labels: tuple[str, str] = ("distilled", "post-trained"),
) -> str:
    """One table per split that both sides evaluated, with the change from before to after."""
    tables = []
    for split in (name for name in before if name in after):
        old, new = _strip(before[split], split), _strip(after[split], split)
        rows = []
        for name in _metric_names((old, new)):
            change = (f"{new[name] - old[name]:+.4f}"
                      if isinstance(old.get(name), (int, float)) and isinstance(new.get(name), (int, float))
                      else "-")
            rows.append([name, _number(old.get(name)), _number(new.get(name)), change])
        tables.append(f"{split}\n{_table(['metric', *labels, 'change'], rows)}")
    return "\n\n".join(tables)


def _strip(values: Mapping[str, object], split: str) -> dict[str, object]:
    prefix = f"{split}_"
    return {key.removeprefix(prefix): value for key, value in values.items()
            if not key.removeprefix(prefix).endswith(_SKIPPED)}


def _metric_names(columns) -> list[str]:
    return list(dict.fromkeys(name for values in columns for name in values))


def _number(value: object) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "-"


def _table(header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(row[index]) for row in (header, *rows)) for index in range(len(header))]

    def line(cells: list[str]) -> str:
        return "  ".join(cell.ljust(width) if index == 0 else cell.rjust(width)
                         for index, (cell, width) in enumerate(zip(cells, widths, strict=True)))

    return "\n".join([line(header), "  ".join("-" * width for width in widths), *map(line, rows)])
