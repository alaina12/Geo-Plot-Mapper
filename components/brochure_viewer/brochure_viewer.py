"""
Brochure Viewer Component - Interactive Fabric.js canvas for plot visualization
"""
import streamlit.components.v1 as components
import os
import json

# Get the directory of this file
_component_dir = os.path.dirname(os.path.abspath(__file__))
_frontend_dir = os.path.join(_component_dir, "frontend")


def brochure_viewer(
    background_image_url: str,
    plots: list,
    plot_overlay_url: str = None,
    key: str = None
):
    """
    Renders an interactive brochure viewer with Fabric.js canvas.
    
    Parameters:
    -----------
    background_image_url : str
        Base64 data URL of the background image
    plots : list
        List of plot dictionaries with keys:
        - id: str - Plot identifier
        - points: list - List of {'x': int, 'y': int} corner points
        - lat: float or None - Latitude
        - lon: float or None - Longitude
        - status: str - Plot status ('available', 'booked', 'sold')
    plot_overlay_url : str, optional
        Base64 data URL of the plot overlay image (optional - colored plots are enough)
    key : str, optional
        Streamlit component key for state management (not used, kept for compatibility)
    
    Returns:
    --------
    dict or None
        Component return value (currently returns None, but can be extended)
    """
    # Read the HTML file
    html_file = os.path.join(_frontend_dir, "index.html")
    
    if not os.path.exists(html_file):
        raise FileNotFoundError(f"Frontend HTML file not found: {html_file}")
    
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Convert plots to JSON string
    plots_json = json.dumps(plots)
    
    # Escape for JavaScript string context (escape quotes and backslashes)
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
    plot_overlay_url_escaped = escape_js_string(plot_overlay_url) if plot_overlay_url else 'null'
    
    # Replace placeholders with actual data
    html_content = html_content.replace(
        '{{BACKGROUND_IMAGE_URL}}', background_image_url_escaped
    ).replace(
        '{{PLOT_OVERLAY_URL}}', plot_overlay_url_escaped
    ).replace(
        '{{PLOTS_DATA}}', plots_json_escaped
    )
    
    # Render the component
    # Note: components.html() doesn't support 'key' parameter, so we omit it
    return components.html(
        html_content,
        height=800,
        scrolling=False
    )

