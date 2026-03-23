@echo off
REM Register the current venv as a Jupyter kernel
REM Run this AFTER activating the venv and installing requirements.txt

echo Registering venv as Jupyter kernel...
python -m ipykernel install --user --name efl-fantasy --display-name "EFL Fantasy (venv)"
echo Done! Kernel registered as "EFL Fantasy (venv)"
echo.
echo When you open Jupyter Lab, select this kernel from the kernel selector (top right).
