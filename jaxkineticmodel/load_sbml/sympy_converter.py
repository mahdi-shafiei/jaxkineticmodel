import libsbml
import sympy

SYMPY_AST_TABLE = {
    sympy.Add: (libsbml.AST_PLUS, None),
    sympy.Mul: (libsbml.AST_TIMES, None),
    sympy.Pow: (libsbml.AST_POWER, 2),
    sympy.sin: (libsbml.AST_FUNCTION_SIN, 1),
    sympy.Symbol: (libsbml.AST_NAME, 0),
    sympy.Integer: (libsbml.AST_INTEGER, 0),
    sympy.Float: (libsbml.AST_REAL, 0),
}

LIBSBML_TIME_NAME = "time"


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


class SympyConverter:
    time_variable_name: str

    def __init__(self, time_variable_name='t'):
        self.time_variable_name = time_variable_name

    def convert(self, expression: sympy.Basic) -> libsbml.ASTNode:
        result = libsbml.ASTNode()

        if isinstance(expression, sympy.Symbol):
            if expression.name == self.time_variable_name:
                result.setName(LIBSBML_TIME_NAME)
                result.setType(libsbml.AST_NAME_TIME)
            else:
                result.setName(expression.name)
        elif isinstance(expression, sympy.Integer):
            result.setValue(int(expression))
        elif isinstance(expression, sympy.Float):
            result.setValue(float(expression))

        for sympy_op in expression.__class__.__mro__:
            sbml_op, arg_count = SYMPY_AST_TABLE.get(sympy_op, (None, None))
            if sbml_op is not None:
                break
        else:
            raise NotImplementedError(f"can't deal yet with expression type {type(expression)}")

        if arg_count is not None and len(expression.args) != arg_count:
            raise ValueError(f'Unexpected number of arguments for '
                             f'{sympy_op}: expected {arg_count}, got '
                             f'{len(expression.args)}')
        if result.getType() == libsbml.AST_UNKNOWN:
            result.setType(sbml_op)
        for child in expression.args:
            # recursively translate children
            result.addChild(self.convert(child))
        if not result.isWellFormedASTNode():
            raise RuntimeError('Failed to build a well-formed '
                               'LibSBML AST node')
        return result
