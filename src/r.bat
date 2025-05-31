@echo off
REM Check if any arguments were provided
if "%~1"=="" (
    echo No experiments specified. Running default: python run.py
    python run.py
) else (
    REM Loop over each argument and run it
    for %%E in (%*) do (
        echo Running experiment %%E
        python run.py %%E
    )
)
