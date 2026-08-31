You are exploring a small, self-contained Python package at `pricing/` in your
current working directory. This is a read-only investigation: do not create,
edit, or delete any files, and do not run `pip install` or access the network.

Answer both of the following questions by reading the code:

1. List every public (does not start with an underscore) top-level function
   defined directly in `pricing/discounts.py`. Sort the names alphabetically.
2. Find the one module-level function (not a class method) anywhere in the
   `pricing` package that mutates one of its parameters in place instead of
   returning a new value. Give it as `"module.function_name"`.

When you are done, reply with exactly one fenced JSON code block containing
both answers, placed between these exact marker lines, with nothing else
after the closing marker:

<!-- ANSWER_START -->
```json
{
  "discounts_public_functions": ["...", "..."],
  "mutator": "module.function_name"
}
```
<!-- ANSWER_END -->
