"""Infrastructure shared by every domain: paths, config and downloads.

This is the bottom of the dependency graph. Nothing here knows about samples,
drugs, response or omics; the domain packages import from ``core`` and never the
other way round, which is what keeps the layering acyclic.

That rule is the admission criterion for this package, and it is enforced by
``tests/core/test_layering.py`` rather than left to good intentions. It is also
what makes this ``core`` rather than a ``utils`` bag: "is this shared by several
domains and ignorant of all of them?" has an answer, where "is this a utility?"
does not, so a module either belongs here or it does not.
"""
