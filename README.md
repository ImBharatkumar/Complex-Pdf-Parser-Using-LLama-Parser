# Project 1: Health Summary Generator 

## Objective

The project aims to reduce the time veterinarians spend reviewing a patient’s medical records before appointments.
Problem Statement
Veterinary patients often have extensive medical histories, including diagnostic details, lab reports, vaccination records, and more. These records are typically stored as unstructured PDF files, sometimes spanning hundreds of pages. Currently, veterinarians must spend 30-40 minutes manually reviewing these documents to find relevant information, which reduces overall clinic productivity.

### Proposed Solution

We propose an AI-based information extraction system that efficiently analyzes large volumes of medical records and generates a structured summary document.

### Usage

clone the repo , `git clone https://github.com/ImBharatkumar/Complex-Pdf-Parser-Using-LLama-Parser.git`

cd Cognito Labs

### Create virtual environment

`conda create -n 'env name'`

### Install dependencies

`pip install -r requirements.txt`

### Export api keys in CLI

export LLAMA_API_KEY="your_api_key" && export LLAMA_API_SECRET="your_api_secret"

### Run application

`python3 app.py`