from pathlib import Path
import subprocess, sys

def test_imports():
    import carecfgen
    assert carecfgen.__version__ == "0.1.0"
