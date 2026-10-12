# Project guidance

- Python modules in this project are called internally by other Python code.
- Do not make each module a standalone CLI script or add `argparse` / `if __name__ == "__main__"` entry points unless the user explicitly requests a command-line interface.
- Prefer importable functions and classes with clear parameters and return values.
