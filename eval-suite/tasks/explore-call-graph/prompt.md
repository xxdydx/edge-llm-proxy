You are exploring a small, self-contained Python package at `pipeline/` in
your current working directory. This is a read-only investigation: do not
create, edit, or delete any files, and do not run `pip install` or access the
network. Comments, docstrings, and string literals that merely mention a
function name are not calls - only actual call expressions in executable code
count.

Answer both of the following questions by reading the code:

1. List every function (as `"module.function_name"`) that contains at least
   one call to `normalize`, sorted alphabetically by that string.
2. Across the whole package, how many total call sites invoke `normalize`?
   Count every call expression separately, even multiple calls inside the
   same function.

When you are done, reply with exactly one fenced JSON code block containing
both answers, placed between these exact marker lines, with nothing else
after the closing marker:

<!-- ANSWER_START -->
```json
{
  "callers": ["module.function_name", "..."],
  "call_site_count": 0
}
```
<!-- ANSWER_END -->
