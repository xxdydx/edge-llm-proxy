# Sandbox image for grading model completions with the OFFICIAL EvalPlus
# differential-testing semantics. Model code only ever runs inside a container
# built from this image, launched with --network none --read-only --cap-drop
# ALL etc. (see grading.py).
FROM python:3.12-slim

RUN pip install --no-cache-dir "evalplus==0.3.1" "numpy>=1.26,<3"

# Do NOT set EVALPLUS_TIMEOUT_PER_TASK: evalplus.eval.untrusted_check does
# `min(os.getenv("EVALPLUS_TIMEOUT_PER_TASK", 60), sum(time_limits))` and
# crashes on `min(str, float)` if the env var is present. Leave it unset so
# the int default is used.
#
# No groundtruth download at run time: the host precomputes expected outputs
# from the trusted canonical solution and passes them in.
WORKDIR /w
