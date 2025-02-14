# type_of_bug = "NONE"
# type_of_bug = "ONLY_MEMORY_LEAK"
type_of_bug = "BOTH"
workaround = False

if type_of_bug == "ONLY_MEMORY_LEAK":
    import libcombine
import libsbml
if type_of_bug == "BOTH":
    import libcombine

plus_minus = {libsbml.AST_PLUS, libsbml.AST_MINUS}

m = libsbml.parseL3Formula('42')
if workaround:
    m = libsbml.ASTNode(m)
t = m.getType()  # This yields the memory leak message
try:
    if t in plus_minus:  # This yields the TypeError
        pass
except TypeError as e:
    print(e)
else:
    print("don't panic!")
