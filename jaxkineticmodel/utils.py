import logging
import logging.config
import sys
import os

import libsbml
import sympy
from sympy import Basic

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def get_logger(name):
    package_root = os.path.dirname(os.path.dirname(__file__))  # Adjust as needed to reach package root
    toml_path = os.path.join(package_root, "pyproject.toml")
    if not getattr(get_logger, "configured", False):
        with open(toml_path, "rb") as f:
            config = tomllib.load(f).get("tool", {}).get("logging", {})
        if not config:
            raise KeyError("No logging configuration found")

        # NOTE: due to a bug in the logging library (?), handlers and formatters can't reliably be set through
        # dictConfig(). We set them manually now.

        console_handler = logging.StreamHandler(sys.stdout)
        format_dict = config.pop("formatters", {}).get("formatter", {})
        if format_dict:
            formatter = logging.Formatter(format_dict.get("format"))
            if "default_time_format" in format_dict:
                formatter.default_time_format = format_dict["default_time_format"]
            console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)

        logging.config.dictConfig(config)
        setattr(get_logger, "configured", True)

    return logging.getLogger(name)


SYMPY_AST_TABLE = [
    [sympy.Add, libsbml.AST_PLUS, None],
    [sympy.Mul, libsbml.AST_TIMES, None],
    [sympy.Pow, libsbml.AST_POWER, 2],
    [sympy.sin, libsbml.AST_FUNCTION_SIN, 1],
    [sympy.Symbol, libsbml.AST_NAME, 0],
    [sympy.Integer, libsbml.AST_INTEGER, 0],
    [sympy.Float, libsbml.AST_REAL, 0],
]


def sympy_to_libsbml(expression: Basic) -> libsbml.ASTNode:
    result = libsbml.ASTNode()

    if isinstance(expression, sympy.Symbol):
        result.setName(expression.name)
    elif isinstance(expression, sympy.Integer):
        result.setValue(int(expression))

    for sympy_op, sbml_op, arg_count in SYMPY_AST_TABLE:
        if isinstance(expression, sympy_op):
            if arg_count is not None and len(expression.args) != arg_count:
                raise ValueError(f'Unexpected number of arguments for '
                                 f'{sympy_op}: expected {arg_count}, got '
                                 f'{len(expression.args)}')
            result.setType(sbml_op)
            for child in expression.args:
                result.addChild(sympy_to_libsbml(child))
            if not result.isWellFormedASTNode():
                raise RuntimeError('Failed to build a well-formed '
                                   'LibSBML AST node')
            return result

    raise NotImplementedError(f"can't deal yet with expression type {type(expression)}")
