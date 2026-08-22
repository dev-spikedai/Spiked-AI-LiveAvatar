"""Provider-agnostic agent core.

Nothing in this package may import from `src.providers`. The dependency runs
one way: adapters know about the core, the core never knows which adapter is
live. That rule is what makes a provider swappable at all -- see
docs/PROVIDER_REFACTOR_PLAN.md.
"""
