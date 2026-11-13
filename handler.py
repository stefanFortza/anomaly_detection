import io
import traceback
from contextlib import redirect_stdout

import runpod

from anomaly_detection.lab3.ex3 import run


def handler(job):
    buffer = io.StringIO()
    inputs = job.get("input", {}) or {}
    n_runs = inputs.get("n_runs", 10)

    try:
        with redirect_stdout(buffer):
            run(n_runs=n_runs)
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

    return {"output": buffer.getvalue()}


runpod.serverless.start({"handler": handler})
