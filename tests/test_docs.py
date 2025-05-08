import importlib
import pkgutil

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

def test_example_script_runs_without_error():
    import runpy
    runpy.run_path("../docs/code/building_models.py")