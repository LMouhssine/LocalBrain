# Activity log schema

The model trains on a single local CSV file.

## File

- `data/activity_log.csv`

## Columns

- `timestamp` (required): ISO datetime string. Examples:
  - `2025-01-15 09:30:00`
  - `2025-01-15T09:30:00`
- `task_name` (required): free text, e.g. `Write project proposal`
- `task_category` (required): stable label, e.g. `writing`, `coding`, `email`, `planning`
- `duration_min` (required): minutes spent (float allowed)
- `energy_level` (optional): integer 1-5 (blank allowed)
- `outcome` (required): `completed` or `interrupted`

## Notes

- Keep `task_category` relatively small (5-20 categories). The model predicts this label.
- `task_name` is used for human-readable recommendations and simple heuristics.
