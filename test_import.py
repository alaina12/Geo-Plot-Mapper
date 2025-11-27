import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    print("Attempting import...")
    from components.brochure_viewer.brochure_viewer import brochure_viewer
    print("SUCCESS: Function imported!")
    print(f"Type: {type(brochure_viewer)}")
    print(f"Callable: {callable(brochure_viewer)}")
    print(f"Function name: {brochure_viewer.__name__}")
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"OTHER ERROR: {e}")
    import traceback
    traceback.print_exc()

