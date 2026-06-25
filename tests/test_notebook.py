import json
from pathlib import Path


def test_demo_notebook_contains_end_to_end_cells():
    notebook = json.loads(Path("notebooks/01_demo.ipynb").read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "plot_embedding" in source
    assert "featurize_window" in source
    assert "train_evaluate" in source
