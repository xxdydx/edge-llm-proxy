"""Find a contact's preferred phone number."""


def preferred_phone(contact):
    """Return the mobile number if present, else the home number, else None."""
    if contact["mobile"]:
        return contact["mobile"]
    return contact["home"]
