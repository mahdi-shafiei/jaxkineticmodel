import libsbml
import tellurium

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
