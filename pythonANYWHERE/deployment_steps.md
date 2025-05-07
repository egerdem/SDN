# Fixing PythonAnywhere Deployment

## Issues Fixed

1. **SciPy OpenBLAS Error**: This was due to compatibility issues with OpenBLAS on PythonAnywhere
2. **NoneType has no attribute server Error**: This was caused by a commented-out line in the `create_app_for_deployment` method
3. **WSGI Configuration Issue**: We need to properly configure the WSGI file to import the application correctly

## Step-by-Step Deployment Instructions

### 1. Fix the Code

The following files have been fixed and should be uploaded to PythonAnywhere:
- `sdn_experiment_visualizer_PYTHONANYWHERE.py` - Uncommented the `server = app.server` line
- `flask_app.py` - Updated to use `application = app.server` for WSGI compatibility

### 2. Update Your WSGI Configuration File

On PythonAnywhere, go to the Web tab and find your WSGI configuration file. Replace its contents with the code from `wsgi_configuration.txt` or use the following:

```python
import sys
import os

# Add your project directory to path
path = '/home/egeerdem/sdn_app'  # Adjust this to your actual directory path
if path not in sys.path:
    sys.path.append(path)

# Import the application object
from flask_app import application  # This imports the 'application' variable from flask_app.py
```

Make sure to adjust the `path` to point to your actual project directory.

### 3. SciPy Installation

If you're still having SciPy-related errors, try installing an older version:

```
pip install scipy==1.7.3
```

Or you can try installing with minimal dependencies:

```
pip install --only-binary=scipy scipy==1.8.1
```

### 4. Audio Files

Make sure the audio files are uploaded to the correct location:
- `trumpet_cr_Track34.wav`
- `002_bongo_original_cr.wav`

### 5. Restart the Application

After making these changes:
1. Go to the Web tab on PythonAnywhere
2. Click the "Reload" button for your web app

### Troubleshooting

If you continue to have issues:
1. Check the error logs in the Web tab
2. Try installing a different version of SciPy
3. Verify all project files are in the correct locations
4. Make sure the paths in your WSGI configuration match your actual directory structure 