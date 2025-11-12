# KLinterSel <img align="right" src="images/thinkazul.svg" alt="Think in azul" width="100">

KLinterSel is a Python project for calculating intersections between selective sites detected by different methods. The main script performs operations on genomic data from selective scans and supports statistical tests and plotting.

## Requirements

- Python 3.7+ (recommended >= 3.10)
- Libraries: numpy, pandas, matplotlib, seaborn, scipy, psutil

## Quick start
### Option A - You already have Python 3.7+

From the project folder:


    python -m pip install -U pip setuptools wheel packaging # optional but recommended
    # If your Python is 3.8, install numpy first to avoid build issues:
    python -m pip install "numpy<1.25"  # safe on 3.8
    # Then the rest:
    python -m pip install -r requirements.txt


### Option B - You **don't** have Python >= 3.7 (install & use a virtual environment)

1. **Install Python (recommended >= 3.10)**

    - **Ubuntu/Debian:** Prefer your distro's Python 3.x packages. If you need a newer Python, you can use a backport repo (e.g., deadsnakes) per your IT policy.
    
        sudo apt update
    
        sudo apt install -y python3.10 python3.10-venv
    

    - **macOS (Homebrew):**
    
        brew install python@3.11

    - **Windows:**
   Download and install from [python.org](https://www.python.org/downloads/)  
   *(check the "Add python.exe to PATH" box during setup).*

2. **Create and activate a virtual environment (recommended)**
  
    If you use Anaconda, deactivate it before creating a venv:
     
        conda deactivate

        # Use the exact interpreter you installed (examples below):
        
        # Linux/macOS
        python3 -m venv venv310  # or: /usr/bin/python3.11 -m venv venv310
        source venv310/bin/activate

        # Windows (PowerShell)
        py -3.11 -m venv venv310
        .\venv310\Scripts\Activate.ps1
  

3. **Install dependencies inside the venv**

    python -m pip install -U pip setuptools wheel packaging
    
    \# if your Python is 3.8, install numpy first:
    python -m pip install "numpy<1.25"
    python -m pip install -r requirements.txt
   
🧠 Note: You can name your environment folder .venv, venv3.10, or any name you prefer.

## Installation (from source)

    git clone https://github.com/noosdev0/KLinterSel.git
    cd KLinterSel
    # (Pick Option A or B above to install dependencies)


## Usage

Run the main script:

    python3 KLinterSel.py <file1> <file2> <file3> [file4 ...]

Only intersections at a specified distance:


    python3 KLinterSel.py <file1> <file2> <file3> [file4 ...] --dist <distance> --notest


Plot distributions:


    python3 KLinterSel.py <file1> <file2> <file3> [file4 ...] --paint

  
## Features

- Filter clusters in genomic data from selective scans
- Calculate intersections between different methods
- Generate random data for control
- Perform statistical tests
- Plot distributions

## Troubleshooting
  - **pandas complains about missing numpy during install**
  
    Install numpy first, then the rest:

        python -m pip install "numpy<1.25"   # useful if Python 3.8 is installed
        python -m pip install -r requirements.txt

  - **pip attempts to compile and fails**
  
    Make sure build tools and pip are up to date:

        python -m pip install -U pip setuptools wheel packaging
    On Linux, if you really need to compile:
        sudo apt install -y build-essential python3-dev

  
  - **The script is running with the wrong Python**
  
    Enable venv and use Python (without the 3). Verify with:

        python -V
        which python   # (Linux/macOS)
        where python   # (Windows)

  