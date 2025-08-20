# KLinterSel <img align="right" src="images/thinkazul.svg" alt="Think in azul" width="100">

KLinterSel is a Python project for calculating intersections between selective sites detected by different methods. The main script performs operations on genomic data from selective scans and supports statistical tests and plotting.

## Requirements

- Python 3.7+
- Libraries: numpy, pandas, matplotlib, seaborn, scipy, psutil

## Installation

You can get KLinterSel by downloading the prebuilt binaries from the [Releases section](https://github.com/noosdev0/KLinterSel/releases) or by installing the program from source:

1. Clone this repository or download the source files by clicking on the green "Code" button and selecting "Download ZIP."
   ```bash
   git clone https://github.com/noosdev0/KLinterSel.git


2. Create and activate a virtual environment (recommended).

```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```


3. Install the required libraries using `pip install -r requirements.txt`

## Usage

To run the script, use the following command:

```bash
   python3 KLinterSel.py <file1> <file2> <file3> [file4 ...]
```
For just intersections at a specified distance:

```bash
python3 KLinterSel.py <file1> <file2> <file3> [file4 ...] --dist <distance> --notest
```
For plotting the distribution:

```bash
python3 KLinterSel.py <file1> <file2> <file3> [file4 ...] --paint
```
## Features

- Filter clusters in genomic data from selective scans
- Calculate intersections between different methods
- Generate random data for control
- Perform statistical tests
- Plot distributions
