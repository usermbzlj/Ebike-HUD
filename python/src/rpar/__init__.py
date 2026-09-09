"""Road Perception AR desktop runtime (schema-compatible with the Android app)."""

from rpar.enums import (
    Direction,
    GeometryType,
    LifecycleState,
    ObjectState,
    PerceptionStatus,
    SemanticType,
    Severity,
    VisibilityClass,
)

__all__ = [
    "Direction",
    "GeometryType",
    "LifecycleState",
    "ObjectState",
    "PerceptionStatus",
    "SemanticType",
    "Severity",
    "VisibilityClass",
    "__version__",
]

__version__ = "0.2.0"
SCHEMA_VERSION = "1.1"
