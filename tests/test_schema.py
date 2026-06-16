"""Tests for openstream._schema -- JSON Schema generation from type hints."""

from __future__ import annotations

import dataclasses
import enum
import functools
from typing import Literal, Optional, TypedDict

from openstream._schema import (
    _generate_schema,
    _parse_param_descriptions,
    _type_to_schema,
)


# -- Fixtures for complex types --


class Color(enum.Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


@dataclasses.dataclass
class Point:
    x: float
    y: float
    label: str = ""


@dataclasses.dataclass
class NestedDC:
    name: str
    point: Point


class AddressTypedDict(TypedDict, total=False):
    street: str
    city: str


class PersonTypedDict(TypedDict):
    name: str
    age: int


class MixedTypedDict(TypedDict, total=False):
    optional_field: str
    name: str  # required_keys won't include these when total=False


class MixedRequiredTypedDict(PersonTypedDict, total=False):
    nickname: str  # optional because total=False on this class


# ============================================================
# _type_to_schema tests
# ============================================================


class TestPrimitiveTypes:
    def test_str(self):
        assert _type_to_schema(str) == {"type": "string"}

    def test_int(self):
        assert _type_to_schema(int) == {"type": "integer"}

    def test_float(self):
        assert _type_to_schema(float) == {"type": "number"}

    def test_bool(self):
        assert _type_to_schema(bool) == {"type": "boolean"}

    def test_bare_dict(self):
        assert _type_to_schema(dict) == {"type": "object"}

    def test_bare_list(self):
        assert _type_to_schema(list) == {"type": "array"}


class TestOptionalTypes:
    def test_optional_str(self):
        schema = _type_to_schema(Optional[str])
        assert schema == {"type": ["string", "null"]}

    def test_optional_int(self):
        schema = _type_to_schema(Optional[int])
        assert schema == {"type": ["integer", "null"]}

    def test_optional_float(self):
        schema = _type_to_schema(Optional[float])
        assert schema == {"type": ["number", "null"]}

    def test_optional_bool(self):
        schema = _type_to_schema(Optional[bool])
        assert schema == {"type": ["boolean", "null"]}

    def test_pep604_union_none(self):
        schema = _type_to_schema(str | None)
        assert schema == {"type": ["string", "null"]}

    def test_pep604_int_none(self):
        schema = _type_to_schema(int | None)
        assert schema == {"type": ["integer", "null"]}

    def test_optional_enum(self):
        schema = _type_to_schema(Optional[Color])
        assert schema == {"enum": ["red", "green", "blue", None]}


class TestGenericTypes:
    def test_list_of_str(self):
        schema = _type_to_schema(list[str])
        assert schema == {"type": "array", "items": {"type": "string"}}

    def test_list_of_int(self):
        schema = _type_to_schema(list[int])
        assert schema == {"type": "array", "items": {"type": "integer"}}

    def test_dict_str_int(self):
        schema = _type_to_schema(dict[str, int])
        assert schema == {
            "type": "object",
            "additionalProperties": {"type": "integer"},
        }

    def test_dict_str_list_str(self):
        schema = _type_to_schema(dict[str, list[str]])
        assert schema == {
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {"type": "string"},
            },
        }

    def test_list_of_optional(self):
        schema = _type_to_schema(list[Optional[str]])
        assert schema == {
            "type": "array",
            "items": {"type": ["string", "null"]},
        }


class TestLiteralType:
    def test_string_literal(self):
        schema = _type_to_schema(Literal["a", "b", "c"])
        assert schema == {"type": "string", "enum": ["a", "b", "c"]}

    def test_single_literal(self):
        schema = _type_to_schema(Literal["only"])
        assert schema == {"type": "string", "enum": ["only"]}

    def test_int_literal(self):
        schema = _type_to_schema(Literal[1, 2])
        assert schema == {"type": "integer", "enum": [1, 2]}

    def test_bool_literal(self):
        schema = _type_to_schema(Literal[True, False])
        assert schema == {"type": "boolean", "enum": [True, False]}

    def test_mixed_literal(self):
        schema = _type_to_schema(Literal["a", 1])
        assert schema == {"type": "string", "enum": ["a", 1]}


class TestEnumType:
    def test_enum_values(self):
        schema = _type_to_schema(Color)
        assert schema == {"enum": ["red", "green", "blue"]}

    def test_int_enum(self):
        class Priority(enum.Enum):
            LOW = 1
            HIGH = 2

        schema = _type_to_schema(Priority)
        assert schema == {"enum": [1, 2]}


class TestDataclass:
    def test_simple_dataclass(self):
        schema = _type_to_schema(Point)
        assert schema["type"] == "object"
        assert schema["properties"]["x"] == {"type": "number"}
        assert schema["properties"]["y"] == {"type": "number"}
        assert schema["properties"]["label"] == {"type": "string"}
        # x and y are required (no default), label has default
        assert "x" in schema["required"]
        assert "y" in schema["required"]
        assert "label" not in schema.get("required", [])

    def test_nested_dataclass(self):
        schema = _type_to_schema(NestedDC)
        assert schema["type"] == "object"
        assert schema["properties"]["name"] == {"type": "string"}
        point_schema = schema["properties"]["point"]
        assert point_schema["type"] == "object"
        assert "x" in point_schema["properties"]


class TestTypedDict:
    def test_all_required(self):
        schema = _type_to_schema(PersonTypedDict)
        assert schema["type"] == "object"
        assert schema["properties"]["name"] == {"type": "string"}
        assert schema["properties"]["age"] == {"type": "integer"}
        assert set(schema["required"]) == {"name", "age"}

    def test_all_optional(self):
        schema = _type_to_schema(AddressTypedDict)
        assert schema["type"] == "object"
        assert "street" in schema["properties"]
        assert "city" in schema["properties"]
        assert "required" not in schema

    def test_mixed_required(self):
        schema = _type_to_schema(MixedRequiredTypedDict)
        assert schema["type"] == "object"
        # name and age come from PersonTypedDict (total=True)
        assert "name" in schema["required"]
        assert "age" in schema["required"]
        # nickname is from the total=False subclass
        assert "nickname" not in schema.get("required", [])


class TestUnknownType:
    def test_custom_class(self):
        class Foo:
            pass

        assert _type_to_schema(Foo) == {"type": "object"}

    def test_none_type(self):
        assert _type_to_schema(type(None)) == {"type": "null"}


# ============================================================
# _parse_param_descriptions tests
# ============================================================


class TestGoogleStyleDocstring:
    def test_basic(self):
        def fn(name, age):
            """Do something.

            Args:
                name: The person's name.
                age: The person's age.
            """

        desc = _parse_param_descriptions(fn)
        assert desc == {
            "name": "The person's name.",
            "age": "The person's age.",
        }

    def test_with_type_annotation_in_docstring(self):
        def fn(x):
            """Process.

            Args:
                x (int): The value.
            """

        desc = _parse_param_descriptions(fn)
        assert desc == {"x": "The value."}

    def test_multiline_stops_at_section(self):
        def fn(a, b):
            """Summary.

            Args:
                a: First param.
                b: Second param.

            Returns:
                Something.
            """

        desc = _parse_param_descriptions(fn)
        assert desc == {"a": "First param.", "b": "Second param."}


class TestReSTStyleDocstring:
    def test_basic(self):
        def fn(x, y):
            """Do math.

            :param x: The x coordinate.
            :param y: The y coordinate.
            """

        desc = _parse_param_descriptions(fn)
        assert desc == {
            "x": "The x coordinate.",
            "y": "The y coordinate.",
        }

    def test_mixed_with_google(self):
        # If both styles present, Google-style wins for overlapping params
        def fn(a, b):
            """Summary.

            Args:
                a: From Google.

            :param a: From reST.
            :param b: Only reST.
            """

        desc = _parse_param_descriptions(fn)
        assert desc["a"] == "From Google."  # Google takes precedence
        assert desc["b"] == "Only reST."


class TestNoDocstring:
    def test_returns_empty(self):
        def fn(x, y):
            pass

        assert _parse_param_descriptions(fn) == {}


class TestWrappedFunction:
    def test_follows_wrapped(self):
        def original(name, count):
            """Original docs.

            Args:
                name: The name value.
                count: How many times.
            """

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            return original(*args, **kwargs)

        desc = _parse_param_descriptions(wrapper)
        assert desc == {
            "name": "The name value.",
            "count": "How many times.",
        }


# ============================================================
# _generate_schema tests
# ============================================================


class TestGenerateSchemaBasic:
    def test_simple_function(self):
        def greet(name: str, excited: bool = False) -> str:
            """Greet someone.

            Args:
                name: Who to greet.
                excited: Add exclamation mark.
            """

        schema = _generate_schema(greet)
        assert schema["type"] == "object"
        assert schema["properties"]["name"] == {
            "type": "string",
            "description": "Who to greet.",
        }
        assert schema["properties"]["excited"] == {
            "type": "boolean",
            "description": "Add exclamation mark.",
        }
        assert schema["required"] == ["name"]

    def test_all_required(self):
        def fn(a: int, b: str):
            pass

        schema = _generate_schema(fn)
        assert set(schema["required"]) == {"a", "b"}

    def test_no_required(self):
        def fn(a: int = 0, b: str = ""):
            pass

        schema = _generate_schema(fn)
        assert "required" not in schema


class TestGenerateSchemaInject:
    def test_excludes_injected(self):
        def fn(conn, name: str, age: int):
            pass

        schema = _generate_schema(fn, inject={"conn"})
        assert "conn" not in schema["properties"]
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]

    def test_inject_multiple(self):
        def fn(ctx, db, value: str):
            pass

        schema = _generate_schema(fn, inject={"ctx", "db"})
        assert set(schema["properties"].keys()) == {"value"}


class TestGenerateSchemaSkipSelfCls:
    def test_skip_self(self):
        # Simulate a method signature
        def method(self, name: str):
            pass

        schema = _generate_schema(method)
        assert "self" not in schema["properties"]
        assert "name" in schema["properties"]

    def test_skip_cls(self):
        def classmethod_like(cls, name: str):
            pass

        schema = _generate_schema(classmethod_like)
        assert "cls" not in schema["properties"]


class TestGenerateSchemaNoAnnotations:
    def test_unannotated_params(self):
        def fn(x, y, z):
            pass

        schema = _generate_schema(fn)
        assert schema["properties"]["x"] == {"type": "object"}
        assert schema["properties"]["y"] == {"type": "object"}
        assert schema["properties"]["z"] == {"type": "object"}
        assert set(schema["required"]) == {"x", "y", "z"}


class TestGenerateSchemaWrapped:
    def test_wrapped_function_schema(self):
        def original(name: str, count: int = 1):
            """Do work.

            Args:
                name: The target name.
                count: Repeat count.
            """

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            return original(*args, **kwargs)

        schema = _generate_schema(wrapper)
        assert schema["properties"]["name"] == {
            "type": "string",
            "description": "The target name.",
        }
        assert schema["properties"]["count"] == {
            "type": "integer",
            "description": "Repeat count.",
        }
        # count has default, so only name is required
        assert schema["required"] == ["name"]


class TestGenerateSchemaComplex:
    def test_optional_and_literal(self):
        def fn(
            mode: Literal["fast", "slow"],
            tags: list[str] = [],
            meta: Optional[dict[str, int]] = None,
        ):
            """Configure.

            Args:
                mode: Processing mode.
                tags: Tags to apply.
                meta: Optional metadata.
            """

        schema = _generate_schema(fn)
        assert schema["properties"]["mode"] == {
            "type": "string",
            "enum": ["fast", "slow"],
            "description": "Processing mode.",
        }
        assert schema["properties"]["tags"] == {
            "type": "array",
            "items": {"type": "string"},
            "description": "Tags to apply.",
        }
        meta_prop = schema["properties"]["meta"]
        assert meta_prop["description"] == "Optional metadata."
        assert meta_prop["type"] == ["object", "null"]
        assert meta_prop["additionalProperties"] == {"type": "integer"}
        assert schema["required"] == ["mode"]

    def test_enum_param(self):
        def fn(color: Color):
            """Pick color.

            Args:
                color: The color to use.
            """

        schema = _generate_schema(fn)
        assert schema["properties"]["color"] == {
            "enum": ["red", "green", "blue"],
            "description": "The color to use.",
        }

    def test_dataclass_param(self):
        def fn(point: Point):
            pass

        schema = _generate_schema(fn)
        point_prop = schema["properties"]["point"]
        assert point_prop["type"] == "object"
        assert "x" in point_prop["properties"]
