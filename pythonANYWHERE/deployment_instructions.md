# Completing the PythonAnywhere Deployment Setup

## Complete the create_app_for_deployment method

The create_app_for_deployment method has been added to the sdn_experiment_visualizer_PYTHONANYWHERE.py file, but it needs to be completed with all the layout and callback code from the show() method.

Here's what you need to do:

1. Open sdn_experiment_visualizer_PYTHONANYWHERE.py
2. Find the create_app_for_deployment method
3. Copy everything from the show() method after these lines:
   ```python
   # Create Dash app
   app = dash.Dash(__name__)
   server = app.server
   ```
   
4. Paste all this content into the create_app_for_deployment method right after the existing route definitions
5. Remove the code at the end of the show method that starts the server:
   ```python
   Timer(1, open_browser).start()
   server_address = f"http://127.0.0.1:{port}/"
   print("\n" + "=" * 70)
   print(f"Dash server is running at: {server_address}")
   print("If the browser doesn't open automatically, please copy and paste the URL above.")
   print("=" * 70)
   # Run the app
   app.run_server(debug=True, port=port)
   ```
   
6. Instead, just make sure the method ends with:
   ```python
   return app
   ```

7. Save the file

## Create a WSGI Configuration File

When setting up your PythonAnywhere web app, you'll need to configure the WSGI file. Replace the content with:

```python
import sys
import os

path = '/home/yourusername/SDN-visualizer'  # Replace with your actual path
if path not in sys.path:
    sys.path.append(path)

from pythonANYWHERE.flask_app import application
```

## Required Files for PythonAnywhere

Ensure you have all these files in your PythonAnywhere directory:

1. sdn_experiment_visualizer_PYTHONANYWHERE.py
2. flask_app.py
3. sdn_manager_load_sims.py
4. analysis.py
5. EchoDensity.py
6. plot_room.py (you may need to create this file or adjust code if not available)
7. sdn_experiment_manager.py (you may need to create this file or adjust code if not available)
8. trumpet_cr_Track34.wav
9. 002_bongo_original_cr.wav
10. requirements.txt

## Project Directory Structure

```
/home/yourusername/SDN-visualizer/
│
├── flask_app.py
├── sdn_experiment_visualizer_PYTHONANYWHERE.py
├── sdn_manager_load_sims.py
├── analysis.py
├── EchoDensity.py
├── plot_room.py
├── sdn_experiment_manager.py
├── trumpet_cr_Track34.wav
├── 002_bongo_original_cr.wav
├── requirements.txt
│
└── results/
    └── room_singulars/
        └── trials_aes_4/
            └── [your data files]
```

## Final Deployment Steps

1. Create a virtual environment in PythonAnywhere:
   ```
   mkvirtualenv --python=python3.9 sdn-env
   ```

2. Install the requirements:
   ```
   pip install -r requirements.txt
   ```

3. Configure your web app in the PythonAnywhere Web tab

4. Set your virtual environment path
   ```
   /home/yourusername/.virtualenvs/sdn-env
   ```

5. Set your WSGI file as described above

6. Click "Reload" to start your app 