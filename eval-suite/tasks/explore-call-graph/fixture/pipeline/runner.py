"""Top-level pipeline entry point."""

from .stages import stage_one, stage_two, stage_three


def run_pipeline(batch):
    """Run every stage in order and return the final batch.

    Does not call normalize() itself - each stage decides whether it needs to.
    """
    step1 = [stage_one(record) for record in batch]
    step2 = stage_two(step1)
    return stage_three(step2)
