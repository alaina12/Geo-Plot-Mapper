# Plot Detection

A Streamlit application for plot detection and mapping.

## Setup

1. Create and activate a virtual environment (if not already created):
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

2. Run the Streamlit application:
   ```bash
   streamlit run src/app.py
   ```

## Requirements

The project requires the following Python packages (see `requirements.txt`):
- streamlit
- opencv-python
- numpy
- folium
- streamlit-folium
- Pillow
- google-generativeai
- pytesseract
