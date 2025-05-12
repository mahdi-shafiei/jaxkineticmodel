import importlib
import pkgutil
import runpy
import os
# Tests whether for package the imports work


def check_imports(package_name):
    package = importlib.import_module(package_name)
    for _, modname, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        try:
            importlib.import_module(modname)
            print(f"Successfully imported {modname}")
        except ImportError as e:
            print(f"Failed to import {modname}: {e}")
            return False
    return True


def test_imports():
    assert check_imports("jaxkineticmodel")

def test_building_models():
    runpy.run_path("docs/code/building_models.py")

def test_custom_reactions():
    runpy.run_path("docs/code/custom_reactions.py")

def test_minimal_example():
    runpy.run_path("docs/code/minimal_example.py")

def test_sbml():
    runpy.run_path("docs/code/sbml.py")

def training_models():
    runpy.run_path("docs/code/training_models.py")

def training_models():
    runpy.run_path("docs/code/glycolysis.py")