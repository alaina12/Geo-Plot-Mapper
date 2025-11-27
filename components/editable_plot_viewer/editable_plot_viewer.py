"""
Editable Plot Viewer Component - Interactive Fabric.js canvas for editing plot points
"""
import streamlit.components.v1 as components
import os
import json

# Get the directory of this file
_component_dir = os.path.dirname(os.path.abspath(__file__))
_frontend_dir = os.path.join(_component_dir, "frontend")


def editable_plot_viewer(
    background_image_url: str,
    plots: list,
    mode: str = "points"  # "points" or "lines"
):
    """
    Renders an interactive editable plot viewer with Fabric.js canvas.
    
    Parameters:
    -----------
    background_image_url : str
        Base64 data URL of the background image
    plots : list
        List of plot dictionaries with keys:
        - id: str - Plot identifier
        - plot_number: int - Plot number
        - points: list - List of {'x': int, 'y': int} corner points
    mode : str
        Display mode: "points" (red dots only) or "lines" (red lines with dots)
    
    Returns:
    --------
    dict or None
        Component return value with updated plot coordinates
    """
    # Read the HTML file
    html_file = os.path.join(_frontend_dir, "index.html")
    
    if not os.path.exists(html_file):
        raise FileNotFoundError(f"Frontend HTML file not found: {html_file}")
    
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Convert plots to JSON string
    plots_json = json.dumps(plots)
    
    # Escape for JavaScript string context
    def escape_js_string(s):
        if s is None:
            return 'null'
        return (s.replace('\\', '\\\\')
                 .replace("'", "\\'")
                 .replace('"', '\\"')
                 .replace('\n', '\\n')
                 .replace('\r', '\\r'))
    
    plots_json_escaped = escape_js_string(plots_json)
    background_image_url_escaped = escape_js_string(background_image_url)
    
    # Replace placeholders with actual data
    html_content = html_content.replace(
        '{{BACKGROUND_IMAGE_URL}}', background_image_url_escaped
    ).replace(
        '{{PLOTS_DATA}}', plots_json_escaped
    ).replace(
        '{{MODE}}', mode
    )
    
    # Render the component and get return value
    # Note: components.html() doesn't support 'key' parameter
    result = components.html(
        html_content,
        height=800,
        scrolling=False
    )
    
    return result

