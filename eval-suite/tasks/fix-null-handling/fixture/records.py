"""Find the first usable email address in a list of contact records."""


def first_valid_email(records):
    """Return the first non-empty, lowercased/stripped email, or None."""
    for record in records:
        if record["email"]:
            return record["email"].strip().lower()
    return None
