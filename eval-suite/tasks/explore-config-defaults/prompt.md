You are exploring a small, self-contained Python package at `appconfig/` in
your current working directory. This is a read-only investigation: do not
create, edit, or delete any files, and do not run `pip install` or access the
network.

Answer both of the following questions by reading the code (not by guessing
from names or comments):

1. Across every file in the `appconfig` package, how many distinct exception
   *types* are actually caught by an `except` clause anywhere? Count each
   type once even if it is caught in more than one place; a tuple such as
   `except (TypeError, ValueError):` catches two types. A type merely
   mentioned in a comment or string does not count.
2. Which function raises `ValueError` when `load_settings` is given an
   override key it does not recognize? Give it as `"module.function_name"`.

When you are done, reply with exactly one fenced JSON code block containing
both answers, placed between these exact marker lines, with nothing else
after the closing marker:

<!-- ANSWER_START -->
```json
{
  "distinct_exception_types_caught": 0,
  "unknown_key_raiser": "module.function_name"
}
```
<!-- ANSWER_END -->
