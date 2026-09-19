"""Cloud-relative teacher-forcing study: can a learned classifier predict,
from request-only features, which historical cloud calls a local model could
have handled without diverging from what cloud actually did?

See README.md in this directory for the full methodology and its limits.
Nothing here edits `edgeproxy/router.py` or any deployed routing policy --
this package only *reads* `router.extract_features`/`router.CallFeatures`
and the existing trace/schema-validation helpers.
"""
