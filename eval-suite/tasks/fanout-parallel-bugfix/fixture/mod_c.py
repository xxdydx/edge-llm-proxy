"""Sort tasks by priority, breaking ties by earliest due date."""


def sort_tasks(tasks):
    return sorted(tasks, key=lambda t: (-t["priority"], t["title"]))
