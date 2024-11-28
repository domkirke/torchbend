from typing import Literal
import torch
import os, sys
from enum import Enum

libpath = os.path.abspath((os.path.join(os.path.dirname(__file__), "..", "..")))
if libpath not in sys.path:
    sys.path.append(libpath)

def log_to_file(f, label, value):
    f.write(f"{label} : \n{value}\n\n{'-' * 16}")

outdir = __file__
def get_log_file(outdir=outdir):
    outdir = os.path.join(os.path.dirname(outdir), "outs")
    test_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    return os.path.join(outdir, test_name+"_out.txt")

    