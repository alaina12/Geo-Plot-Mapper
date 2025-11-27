import sys
import os

# Same path setup as app.py
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

print(f"Parent dir: {parent_dir}")
print(f"Python path: {sys.path[:3]}")

try:
    print("Attempting import...")
    from components.brochure_viewer.brochure_viewer import brochure_viewer
    print("SUCCESS: Function imported!")
    print(f"Type: {type(brochure_viewer)}")
    print(f"Callable: {callable(brochure_viewer)}")
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"OTHER ERROR: {e}")
    import traceback
    traceback.print_exc()

