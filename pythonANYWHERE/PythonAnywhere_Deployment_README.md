# SDN Visualizer - PythonAnywhere Deployment Guide

This guide provides step-by-step instructions for deploying the SDN Visualizer on PythonAnywhere.

## Step 1: Set up a PythonAnywhere account

1. Sign up for a PythonAnywhere account at https://www.pythonanywhere.com
2. Log in to your account

## Step 2: Upload your files

1. Go to the "Files" tab in PythonAnywhere
2. Create a new directory for your project (e.g., `/home/yourusername/SDN-visualizer`)
3. Upload the following files:
   - `flask_app.py`
   - `sdn_experiment_visualizer_PYTHONANYWHERE.py`
   - `sdn_manager_load_sims.py`
   - `analysis.py`
   - `EchoDensity.py`
   - `plot_room.py` (if available, or create it)
   - `sdn_experiment_manager.py` (if available, or create it)
   - `trumpet_cr_Track34.wav`
   - `002_bongo_original_cr.wav`
   - `requirements.txt`

4. Create the results directory structure:
   - Click "New directory" to create `/home/yourusername/SDN-visualizer/results`
   - Create subdirectories: `results/room_singulars/trials_aes_4`
   - Upload your data files to the appropriate directories

## Step 3: Set up virtual environment

1. Go to the "Consoles" tab in PythonAnywhere
2. Start a new Bash console
3. Navigate to your project directory:
   ```
   cd ~/SDN-visualizer
   ```
4. Create a virtual environment:
   ```
   mkvirtualenv --python=python3.9 sdn-env
   ```
5. Activate the virtual environment:
   ```
   workon sdn-env
   ```
6. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Step 4: Configure web app

1. Go to the "Web" tab
2. Click "Add a new web app"
3. Choose "Manual configuration"
4. Select Python 3.9
5. Enter the path to your Flask app:
   - Code: `/home/yourusername/SDN-visualizer`
   - WSGI config file: Will be auto-created (you'll modify it next)
6. Set your virtual environment path:
   - `/home/yourusername/.virtualenvs/sdn-env`

## Step 5: Configure WSGI file

1. In the Web tab, click on the link to your WSGI configuration file
2. Replace its contents with:
   ```python
   import sys
   import os
   
   path = '/home/yourusername/SDN-visualizer'  # Replace with your actual path
   if path not in sys.path:
       sys.path.append(path)
   
   from pythonANYWHERE.flask_app import application
   ```
3. Save the file

## Step 6: Start your app

1. Go back to the Web tab
2. Click the "Reload" button for your web app
3. Once it reloads, click the link to your web app (it should look like `yourusername.pythonanywhere.com`)

## Troubleshooting

If your app doesn't work correctly:

1. Check the error logs in the Web tab
2. Verify all files are uploaded and the directory structure is correct
3. Make sure the virtual environment is properly set up with all requirements
4. Check that audio files are in the correct location
5. Verify that your data files are in the expected format and location

## Notes

- Free PythonAnywhere accounts have CPU time and memory limitations
- The audio convolution features may be resource-intensive
- If possible, pre-compute your audio convolutions to reduce server load

## Updating your app

When you need to make changes:

1. Upload the updated files via the Files tab
2. Reload your web app from the Web tab 