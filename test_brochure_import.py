import sys
import os

# Same setup as app.py
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import brochure_viewer function - robust import with multiple fallbacks
import inspect
import importlib
import importlib.util

brochure_viewer = None

# Method 1: Try package import (preferred)
try:
    from components.brochure_viewer import brochure_viewer
    if inspect.isfunction(brochure_viewer):
        print('✓ Method 1: SUCCESS - Package import')
    elif inspect.ismodule(brochure_viewer):
        print('Method 1: Got module, extracting function...')
        brochure_viewer = getattr(brochure_viewer, 'brochure_viewer', None)
except (ImportError, AttributeError) as e:
    print(f'Method 1: Failed - {e}')

# Method 2: If Method 1 failed, try direct module import using importlib
if not inspect.isfunction(brochure_viewer):
    try:
        mod = importlib.import_module('components.brochure_viewer.brochure_viewer')
        brochure_viewer = getattr(mod, 'brochure_viewer', None)
        if inspect.isfunction(brochure_viewer):
            print('✓ Method 2: SUCCESS - Direct module import')
    except (ImportError, AttributeError) as e:
        print(f'Method 2: Failed - {e}')

# Final verification
if inspect.isfunction(brochure_viewer):
    print(f'\n✓✓✓ SUCCESS! Function imported correctly.')
    print(f'   Type: {type(brochure_viewer).__name__}')
    print(f'   Name: {brochure_viewer.__name__}')
    print(f'   Callable: {callable(brochure_viewer)}')
else:
    print(f'\n✗✗✗ FAILED! Could not import function.')
    print(f'   Got: {type(brochure_viewer)}')

