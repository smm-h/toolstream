"""Generate JSON Schema from Python function type hints.

Foundation for the tool registration system. Converts function signatures
(parameters, type annotations, docstrings) into JSON Schema objects
suitable for LLM tool definitions.
"""

from __future__ import annotations

import dataclasses
import enum
import inspect
import re
import typing


def _type_to_schema(annotation: type) -> dict:
    """Convert a Python type hint to a JSON Schema fragment."""

    # Handle None / NoneType directly
    if annotation is type(None):
        return {"type": "null"}

    # Handle missing annotation
    if annotation is inspect.Parameter.empty:
        return {"type": "object"}

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    # Optional[T] is Union[T, None]; T | None has origin types.UnionType
    if origin is typing.Union or _is_union_type(annotation):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and len(args) == 2:
            # Optional[T] -- produce nullable schema
            inner = _type_to_schema(non_none[0])
            if "type" in inner:
                t = inner["type"]
                if isinstance(t, list):
                    if "null" not in t:
                        inner["type"] = t + ["null"]
                else:
                    inner["type"] = [t, "null"]
            elif "enum" in inner:
                inner["enum"] = inner["enum"] + [None]
            else:
                inner["type"] = ["object", "null"]
            return inner
        # General Union -- not handled beyond Optional, fall through
        return {"type": "object"}

    # Literal["a", "b"]
    if origin is typing.Literal:
        return {"type": "string", "enum": list(args)}

    # Enum subclass
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return {"enum": [member.value for member in annotation]}

    # list[T]
    if origin is list:
        if args:
            return {"type": "array", "items": _type_to_schema(args[0])}
        return {"type": "array"}

    # dict[str, T]
    if origin is dict:
        if args and len(args) == 2:
            return {
                "type": "object",
                "additionalProperties": _type_to_schema(args[1]),
            }
        return {"type": "object"}

    # dataclass
    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        properties = {}
        required = []
        # Resolve stringified annotations for dataclass fields
        try:
            resolved = typing.get_type_hints(annotation)
        except Exception:
            resolved = {}
        for field in dataclasses.fields(annotation):
            field_type = resolved.get(field.name, field.type)
            prop = _type_to_schema(field_type)
            properties[field.name] = prop
            # Fields with default or default_factory are optional
            has_default = (
                field.default is not dataclasses.MISSING
                or field.default_factory is not dataclasses.MISSING
            )
            if not has_default:
                required.append(field.name)
        schema: dict = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema

    # TypedDict
    if _is_typed_dict(annotation):
        properties = {}
        hints = typing.get_type_hints(annotation)
        req_keys = getattr(annotation, "__required_keys__", frozenset())
        for name, hint in hints.items():
            properties[name] = _type_to_schema(hint)
        schema = {"type": "object", "properties": properties}
        req = [k for k in hints if k in req_keys]
        if req:
            schema["required"] = req
        return schema

    # Primitive types
    primitives = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        dict: {"type": "object"},
        list: {"type": "array"},
    }
    if annotation in primitives:
        return dict(primitives[annotation])

    return {"type": "object"}


def _is_union_type(annotation: type) -> bool:
    """Check if annotation is a PEP 604 union (X | Y)."""
    import types

    return isinstance(annotation, types.UnionType)


def _is_typed_dict(annotation: type) -> bool:
    """Check if annotation is a TypedDict subclass."""
    return (
        isinstance(annotation, type)
        and issubclass(annotation, dict)
        and hasattr(annotation, "__annotations__")
        and hasattr(annotation, "__required_keys__")
    )


# Regex for Google-style docstring param lines:
#   Args:
#       name: description text
#       name (type): description text
_GOOGLE_ARGS_RE = re.compile(
    r"^\s{2,}(\w+)(?:\s*\([^)]*\))?\s*:\s*(.+)", re.MULTILINE
)

# Regex for reST-style docstring param lines:
#   :param name: description text
_REST_PARAM_RE = re.compile(
    r"^\s*:param\s+(\w+)\s*:\s*(.+)", re.MULTILINE
)


def _parse_param_descriptions(fn: typing.Callable) -> dict[str, str]:
    """Extract parameter descriptions from a function's docstring.

    Handles Google-style (Args: section) and reST-style (:param:) formats.
    Follows __wrapped__ for functools.wraps-decorated functions.
    """
    # Follow __wrapped__ chain to find the original docstring
    target = fn
    while hasattr(target, "__wrapped__"):
        target = target.__wrapped__

    doc = inspect.getdoc(target)
    if not doc:
        return {}

    descriptions: dict[str, str] = {}

    # Try Google-style: find the Args: section
    args_match = re.search(r"^\s*Args?\s*:\s*$", doc, re.MULTILINE)
    if args_match:
        # Extract the block after "Args:" until the next section or end
        after_args = doc[args_match.end() :]
        # Stop at next section header (word followed by colon at start of line,
        # or end of string)
        section_end = re.search(r"^\S", after_args, re.MULTILINE)
        args_block = after_args[: section_end.start()] if section_end else after_args
        for m in _GOOGLE_ARGS_RE.finditer(args_block):
            descriptions[m.group(1)] = m.group(2).strip()

    # Try reST-style
    for m in _REST_PARAM_RE.finditer(doc):
        # Don't overwrite Google-style if both present
        if m.group(1) not in descriptions:
            descriptions[m.group(1)] = m.group(2).strip()

    return descriptions


def _generate_schema(
    fn: typing.Callable,
    inject: set[str] | None = None,
) -> dict:
    """Generate a JSON Schema object from a function's signature.

    Args:
        fn: The function to generate a schema for.
        inject: Parameter names to exclude (they will be injected at call time).

    Returns:
        A JSON Schema dict with "type", "properties", and "required" keys.
    """
    inject = inject or set()
    sig = inspect.signature(fn)
    descriptions = _parse_param_descriptions(fn)

    # Resolve stringified annotations (from `from __future__ import annotations`)
    try:
        resolved_hints = typing.get_type_hints(fn)
    except Exception:
        resolved_hints = {}

    properties: dict[str, dict] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        # Skip self/cls
        if name in ("self", "cls"):
            continue
        # Skip injected parameters
        if name in inject:
            continue

        annotation = resolved_hints.get(name, param.annotation)
        prop = _type_to_schema(annotation)

        # Add description if available
        if name in descriptions:
            prop["description"] = descriptions[name]

        properties[name] = prop

        # Parameters with no default are required
        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema: dict = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema
