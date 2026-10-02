"""Domain-generalization training strategies used by the cross-validation runners.

Only a regular package here; without ``__init__.py`` this directory worked solely
via PEP 420 implicit namespace packaging inside a regular package, which is
fragile under zipimport and some build backends.

Importing the submodules requires ``eegproc[deep-learning]``.
"""

__all__ = ["MetaLearningSubjectSequence", "AlternatingSubjectSetSequence"]


def __getattr__(name: str):
    if name == "MetaLearningSubjectSequence":
        from .meta_learning import MetaLearningSubjectSequence

        return MetaLearningSubjectSequence
    if name == "AlternatingSubjectSetSequence":
        from .alternating_group_learning import AlternatingSubjectSetSequence

        return AlternatingSubjectSetSequence
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
