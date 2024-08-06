# 777
Taxi Data Analysis Report
# 777 - Taxi Data Analysis Report

This repository contains scripts for analyzing New York City taxi trip data. The project includes data cleaning, feature selection, and model training scripts.

## Table of Contents

- [Data Cleaning](#data-cleaning)
- [Feature Selection](#feature-selection)
- [Model Training](#model-training)
- [Getting Started](#getting-started)

## Data Cleaning

The `777finalproject_datacleaning.py` script is responsible for cleaning and preprocessing the taxi trip data. It performs the following tasks:
- Reads the data from a CSV file.
- Parses datetime columns.
- Calculates trip duration.
- Extracts the pickup hour.
- Handles missing values and duplicates.
- Creates a target variable indicating whether a tip was given.
- Saves the cleaned data to a new CSV file.

### Usage

```bash
python 777finalproject_datacleaning.py <input_file> <output_file_path>
