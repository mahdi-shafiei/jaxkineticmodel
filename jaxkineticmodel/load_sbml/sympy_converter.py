import types
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto, unique
from logging import Logger
from typing import Callable, Optional

import libsbml
import sympy


@unique
class ASTNodeType(Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return libsbml.__dict__['AST_' + name]  # noqa

    def matches(self, node: libsbml.ASTNode) -> bool:
        return self.value == node.getType()

    CONSTANT_E = auto();          CONSTANT_FALSE = auto()       # noqa: E702
    CONSTANT_PI = auto();         CONSTANT_TRUE = auto()        # noqa: E702
    DIVIDE = auto();              FUNCTION = auto()             # noqa: E702

    FUNCTION_ABS = auto();        FUNCTION_DELAY = auto()       # noqa: E702
    FUNCTION_ARCCOS = auto();     FUNCTION_EXP = auto()         # noqa: E702
    FUNCTION_ARCCOSH = auto();    FUNCTION_FACTORIAL = auto()   # noqa: E702
    FUNCTION_ARCCOT = auto();     FUNCTION_FLOOR = auto()       # noqa: E702
    FUNCTION_ARCCOTH = auto();    FUNCTION_LN = auto()          # noqa: E702
    FUNCTION_ARCCSC = auto();     FUNCTION_LOG = auto()         # noqa: E702
    FUNCTION_ARCCSCH = auto();    FUNCTION_MAX = auto()         # noqa: E702
    FUNCTION_ARCSEC = auto();     FUNCTION_MIN = auto()         # noqa: E702
    FUNCTION_ARCSECH = auto();    FUNCTION_PIECEWISE = auto()   # noqa: E702
    FUNCTION_ARCSIN = auto();     FUNCTION_POWER = auto()       # noqa: E702
    FUNCTION_ARCSINH = auto();    FUNCTION_QUOTIENT = auto()    # noqa: E702
    FUNCTION_ARCTAN = auto();     FUNCTION_RATE_OF = auto()     # noqa: E702
    FUNCTION_ARCTANH = auto();    FUNCTION_REM = auto()         # noqa: E702
    FUNCTION_CEILING = auto();    FUNCTION_ROOT = auto()        # noqa: E702
    FUNCTION_COS = auto();        FUNCTION_SEC = auto()         # noqa: E702
    FUNCTION_COSH = auto();       FUNCTION_SECH = auto()        # noqa: E702
    FUNCTION_COT = auto();        FUNCTION_SIN = auto()         # noqa: E702
    FUNCTION_COTH = auto();       FUNCTION_SINH = auto()        # noqa: E702
    FUNCTION_CSC = auto();        FUNCTION_TAN = auto()         # noqa: E702
    FUNCTION_CSCH = auto();       FUNCTION_TANH = auto()        # noqa: E702

    INTEGER = auto();             POWER = auto()                # noqa: E702
    LAMBDA = auto();              RATIONAL = auto()             # noqa: E702
    LOGICAL_AND = auto();         REAL = auto()                 # noqa: E702
    LOGICAL_IMPLIES = auto();     REAL_E = auto()               # noqa: E702
    LOGICAL_NOT = auto();         RELATIONAL_EQ = auto()        # noqa: E702
    LOGICAL_OR = auto();          RELATIONAL_GEQ = auto()       # noqa: E702
    LOGICAL_XOR = auto();         RELATIONAL_GT = auto()        # noqa: E702
    MINUS = auto();               RELATIONAL_LEQ = auto()       # noqa: E702
    NAME = auto();                RELATIONAL_LT = auto()        # noqa: E702
    NAME_AVOGADRO = auto();       RELATIONAL_NEQ = auto()       # noqa: E702
    NAME_TIME = auto();           TIMES = auto()                # noqa: E702
    PLUS = auto();                UNKNOWN = auto()              # noqa: E702

    # ORIGINATES_IN_PACKAGE = auto()

    def __str__(self):
        return self.name


LIBSBML_TIME_NAME = "time"


@dataclass
class Mapping:
    # If a mapping has exactly one of sympy_op and libsbml_op set,
    # then SympyConverter should have a custom method for that op.
    # Otherwise, it must have both sympy_op and libsbml_op set and
    # should _not_ have such a method.
    sympy_op: Optional[type[sympy.Basic]]
    libsbml_op: Optional[ASTNodeType]
    arg_count: Optional[int]


# TODO: Rewrite the below as documentation for this class/module
# Here, we would really like to use sympy's MathML utilities.  However, we run into several issues that
#  make them unsuitable for now (i.e. requiring a lot of work):
#  - sympy takes presentation issues into account when producing content MathML, e.g. parsing A_Km as a variable
#  A with subscript Km and producing <mml:msub> tags for it, which libsbml can't handle.
#  - sympy also seems to always produce (xml entities for) e.g. Greek letters in the MathML.
#  - libsbml sometimes sets <cn type="integer"> while sympy can only create bare <cn>.
#  A final small issue is that MathML string must be preceded by an <?xml?> preamble and surrounded by a <math>
#  tag.
#  The sympy implementation seems to store the produced XML DOM in MathMLPrinterBase.dom, which would allow for
#  traversing it and fixing some of these issues.  But this seems like a lot more trouble than it's worth.

#here I can add extra mappings ! #PAUL
MAPPINGS = [
    Mapping(sympy.Add, ASTNodeType.PLUS, None),
    Mapping(sympy.Mul, ASTNodeType.TIMES, None),
    Mapping(None, ASTNodeType.DIVIDE, 2), #new
    Mapping(None, ASTNodeType.FUNCTION, None),#new
    Mapping(None, ASTNodeType.MINUS, None), #new
    Mapping(None, ASTNodeType.REAL_E, 0), #new
    Mapping(None, ASTNodeType.FUNCTION_PIECEWISE, None),
    Mapping(None, ASTNodeType.LAMBDA, None), #new
    Mapping(sympy.Piecewise, None, None),
    Mapping(sympy.Pow, ASTNodeType.POWER, 2),
    Mapping(sympy.Pow, ASTNodeType.FUNCTION_POWER, 2), #new
    Mapping(sympy.Lt, ASTNodeType.RELATIONAL_LT, 2),
    Mapping(sympy.Le, ASTNodeType.RELATIONAL_LEQ, 2),
    Mapping(sympy.Gt, ASTNodeType.RELATIONAL_GT, 2),
    Mapping(sympy.Ge, ASTNodeType.RELATIONAL_GEQ, 2),
    Mapping(sympy.Eq, ASTNodeType.RELATIONAL_EQ, 2),
    Mapping(sympy.Ne, ASTNodeType.RELATIONAL_NEQ, 2),
    Mapping(sympy.sin, ASTNodeType.FUNCTION_SIN, 1),
    Mapping(sympy.cos, ASTNodeType.FUNCTION_COS, 1), #new
    Mapping(sympy.ln,ASTNodeType.FUNCTION_LN,1), #new

    # Mapping(sympy.LessThan,LibSBMLASTNode.RELATIONAL_LT,2), #new
    # Mapping(sympy.GreaterThan,LibSBMLASTNode.RELATIONAL_GT,2), #new
    Mapping(sympy.S.true, ASTNodeType.CONSTANT_TRUE, 0),
    Mapping(sympy.S.false, ASTNodeType.CONSTANT_FALSE, 0),
    Mapping(sympy.Symbol, None, 0),
    Mapping(sympy.Integer, None, 0),
    Mapping(sympy.Float, None, 0),
    Mapping(sympy.S.NaN, None, 0),
    Mapping(None, ASTNodeType.NAME, 0),
    Mapping(None, ASTNodeType.NAME_TIME, 0),
    Mapping(None, ASTNodeType.INTEGER, 0),
    Mapping(None, ASTNodeType.REAL, 0),
]


class Converter:
    time_variable_name: str

    def __init__(self, time_variable_name='t'):
        self.time_variable_name = time_variable_name


class SympyConverter(Converter):
    SYMPY2LIBSBML: dict[type[sympy.Basic], Mapping] = {
        mp.sympy_op: mp for mp in MAPPINGS
    }

    def sympy2libsbml(self, expression: sympy.Basic) -> libsbml.ASTNode:
        result = libsbml.ASTNode()
        for sympy_op in expression.__class__.__mro__:
            custom_method = getattr(self,
                                    f'convert_sympy_{sympy_op.__name__}',
                                    None)
            mp = self.SYMPY2LIBSBML.get(sympy_op, None)
            if mp is not None or custom_method is not None:
                break
        else:
            raise NotImplementedError(f"can't deal yet with expression type {type(expression)}")

        assert mp is not None
        if mp.arg_count is not None and len(expression.args) != mp.arg_count:
            raise ValueError(f'Unexpected number of arguments for '
                             f'{mp.sympy_op}: expected {mp.arg_count}, got '
                             f'{len(expression.args)}')

        if custom_method is not None:
            assert mp.libsbml_op is None
            result = custom_method(expression, result)
        else:
            result.setType(mp.libsbml_op.value)
            for child in expression.args:
                # Recursively convert child nodes
                result.addChild(self.sympy2libsbml(child))

        if not result.isWellFormedASTNode():
            raise RuntimeError('Failed to build a well-formed '
                               'LibSBML AST node')
        return result

    def convert_sympy_Integer(self, number, result) -> libsbml.ASTNode:
        assert isinstance(number, sympy.Integer) and len(number.args) == 0
        result.setType(ASTNodeType.INTEGER.value)
        result.setValue(int(number))
        return result

    def convert_sympy_Float(self, number, result) -> libsbml.ASTNode:
        assert isinstance(number, sympy.Float) and len(number.args) == 0
        result.setType(ASTNodeType.REAL.value)
        result.setValue(float(number))
        return result

    def convert_sympy_NaN(self, nan, result) -> libsbml.ASTNode:
        assert isinstance(nan, sympy.core.numbers.NaN) and len(nan.args) == 0
        result.setType(ASTNodeType.REAL.value)
        result.setValue(float('nan'))
        return result

    def convert_sympy_Symbol(self, symbol, result) -> libsbml.ASTNode:
        assert isinstance(symbol, sympy.Symbol) and len(symbol.args) == 0
        if symbol.name == self.time_variable_name:
            result.setType(ASTNodeType.NAME_TIME.value)
            result.setName(LIBSBML_TIME_NAME)
        else:
            result.setType(ASTNodeType.NAME.value)
            result.setName(symbol.name)
        return result

    def convert_sympy_Piecewise(self, expr, result) -> libsbml.ASTNode:
        assert isinstance(expr, sympy.Piecewise)
        result.setType(ASTNodeType.FUNCTION_PIECEWISE.value)
        # For sympy piecewise functions, the conditions don't have to be
        # mutually exclusive; they are evaluated left-to-right and the
        # first one that matches is applied.
        # However, for libsbml, no order is assumed, and the entire
        # expression is considered to be undefined if multiple conditions
        # evaluate to true.
        # Fortunately, sympy offers functionality to rewrite a piecewise
        # expression to make the conditions mutually exclusive.
        piecewise = sympy.functions.piecewise_exclusive(expr)
        for (value, condition) in piecewise.args:
            result.addChild(self.sympy2libsbml(value))
            result.addChild(self.sympy2libsbml(condition))
        return result


class LibSBMLConverter(Converter):
    LIBSBML2SYMPY: dict[ASTNodeType, Mapping] = {
        mp.libsbml_op: mp for mp in MAPPINGS
    }

    def libsbml2sympy(self, node: libsbml.ASTNode) -> sympy.Basic:
        node = libsbml.ASTNode(node)    # Work around a bug in libsbml
        if not node.isWellFormedASTNode():
            raise ValueError('Got invalid libSBML AST node')

        children = []
        for idx in range(node.getNumChildren()):
            child = node.getChild(idx)
            children.append(self.libsbml2sympy(child))

        libsbml_op = ASTNodeType(node.getType())

        m = self.LIBSBML2SYMPY.get(libsbml_op, None)
        if m is None:
            raise NotImplementedError(f"can't deal yet with libsbml ASTNode "
                                      f"type {libsbml_op}")
        if m.arg_count is not None and len(children) != m.arg_count:
            raise ValueError(f'Unexpected number of children for '
                             f'{libsbml_op}: expected {m.arg_count}, '
                             f'got {len(children)}')

        custom_method = getattr(self,
                                f'convert_libsbml_{libsbml_op}',
                                None)

        if custom_method is not None:
            assert m.sympy_op is None
            result = custom_method(node, children)
        else:
            result = m.sympy_op(*children)


        return result

    def convert_libsbml_NAME(self, node, children) -> sympy.Basic:
        assert ASTNodeType.NAME.matches(node)
        assert len(children) == 0
        return sympy.Symbol(node.getName())

    def convert_libsbml_NAME_TIME(self, node, children) -> sympy.Basic:
        assert ASTNodeType.NAME_TIME.matches(node)
        assert len(children) == 0
        return sympy.Symbol(self.time_variable_name)

    def convert_libsbml_INTEGER(self, node, children) -> sympy.Basic:
        assert ASTNodeType.INTEGER.matches(node)
        assert len(children) == 0
        return sympy.Integer(node.getValue())

    def convert_libsbml_REAL(self, node, children) -> sympy.Basic:
        assert ASTNodeType.REAL.matches(node)
        assert len(children) == 0
        return sympy.Float(node.getValue())

    def convert_libsbml_DIVIDE(self, node, children) -> sympy.Basic:
        "Division has two children a and b (a/b)"
        assert ASTNodeType.DIVIDE.matches(node)
        assert len(children) == 2
        numerator, denominator = children
        return sympy.Mul(numerator,sympy.Pow(denominator,-1))

    def convert_libsbml_FUNCTION_PIECEWISE(self, node, children) -> sympy.Basic:
        assert ASTNodeType.FUNCTION_PIECEWISE.matches(node)
        if len(children) == 0:
            # According to MathML documentation: "The degenerate case of no
            # piece elements and no otherwise element is treated as
            # undefined for all values of the domain."
            return sympy.S.NaN

        if len(children) % 2 == 1:
            # Handle <otherwise> case.  This can be dealt with in sympy by
            # having the last condition-value-pair always match.
            children.append(sympy.S.true)
        pieces = []
        for idx in range(0, len(children), 2):
            value = children[idx]
            condition = children[idx + 1]
            pieces.append((value, condition))
        return sympy.Piecewise(*pieces)

    def convert_libsbml_MINUS(self, node, children) -> sympy.Basic:
        "MINUS can have one child (symbol is negative) or two children (a-b)"
        assert ASTNodeType.MINUS.matches(node)
        if len(children) == 1:
            a=children[0]
            return -a
        if len(children) == 2:
            a, b = children
            return sympy.Add(a, -b)
        else:
            raise Logger.error(f"ERROR: Unexpected number of children for MINUS: {len(children)}")

    def convert_libsbml_REAL_E(self, node, children) -> sympy.Basic:
        assert ASTNodeType.REAL_E.matches(node)
        assert len(children) == 0

        base = sympy.Float(node.getMantissa())  # Extracts the base (mantissa)
        exponent = sympy.Integer(node.getExponent())  # Extracts the exponent

        return sympy.Mul(base, sympy.Pow(10, exponent))  # Represents base * 10^exponent


    def convert_libsbml_FUNCTION(self, node, children) -> sympy.Basic:
        assert ASTNodeType.FUNCTION.matches(node)
        assert len(children) >= 1
        function_name = node.getName()  # Get function name (e.g., 'f', 'g', etc.)
        if not function_name:
            raise ValueError("FUNCTION node has no associated name")

        sympy_function = sympy.Function(function_name)  # Define function
        return sympy_function(*children)  # Apply function to arguments

    def convert_libsbml_LAMBDA(self,node,children)-> sympy.Basic:
        """Mapping to sp lambda.
        The argument order matters in lambda functions and needs to be retrieved properly """
        assert ASTNodeType.LAMBDA.matches(node)
        assert len(children)>=1
        lambda_function=sympy.Lambda(tuple(children[:-1]),children[-1]) #last child is the expression
        return lambda_function



