"""Allow: python -m membranequant  (must be run from parent of this package)."""

from membranequant._bootstrap import ensure_import_path

ensure_import_path()

from membranequant.main import main

raise SystemExit(main())
