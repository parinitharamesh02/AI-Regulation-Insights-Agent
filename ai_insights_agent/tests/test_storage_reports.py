import sys
from pathlib import Path

# Make project root importable so we can "import src"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datetime import datetime
from src.models import Report
from src.data import storage
from src.data.storage import save_report, load_all_reports


def test_save_and_load_report(tmp_path, monkeypatch):
    # Override REPORTS_DIR directly on the storage module,
    # because storage imported REPORTS_DIR from config at import time.
    monkeypatch.setattr(storage, "REPORTS_DIR", tmp_path, raising=False)

    report = Report(
        id="r1",
        created_at=datetime.utcnow(),
        topic="Test",
        summary="Summary",
        takeaways=["A", "B"],
        entities={"organisations": ["Org1"], "people": [], "locations": [], "terms": []},
    )

    save_report(report)
    loaded = load_all_reports()

    assert len(loaded) == 1
    assert loaded[0].id == "r1"
    assert loaded[0].summary == "Summary"
