import pytest
import tempfile
import os
from main import main

def test_main(monkeypatch):
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f1:
        f1.write("wind energy power.")
        fname1 = f1.name
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f2:
        f2.write("solar panels renewable.")
        fname2 = f2.name

    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as fout:
        output_file = fout.name

    monkeypatch.setattr("builtins.input", lambda _: ",".join([fname1, fname2]))
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)
    monkeypatch.setattr("main.OUTPUT_FILE", output_file)  # If using a constant

    main()

    assert os.path.exists(output_file)
    with open(output_file) as f:
        content = f.read()
        assert "keywords" in content.lower()

    os.remove(fname1)
    os.remove(fname2)
    os.remove(output_file)
