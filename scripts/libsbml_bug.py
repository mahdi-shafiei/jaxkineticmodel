# type_of_bug = "NONE"
# type_of_bug = "ONLY_MEMORY_LEAK"
type_of_bug = "BOTH"

if type_of_bug == "ONLY_MEMORY_LEAK":
    import libcombine
import libsbml
if type_of_bug == "BOTH":
    import libcombine

plus_minus = {libsbml.AST_PLUS, libsbml.AST_MINUS}

m = libsbml.parseL3Formula('42')
t = m.getType()  # This yields a memory leak message
try:
    if t in plus_minus:  # This yields a TypeError
        pass
except TypeError as e:
    print(e)
else:
    print("don't panic!")
