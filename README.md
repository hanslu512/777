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

## Feature Selection
The `777finalproject_fs.py` script is used for selecting important features that influence whether a tip is given. It uses Lasso Regression for feature selection.
- Reading Preprocessed Data.
- Preparing Features.
- Assembling Features.
- Standardizing Features.
- Training Lasso Regression Model.
- Extracting Important Features.

 ## Model Training
The `777finalproject_md.py` script trains a logistic regression model to predict whether a passenger will tip based on the selected features. It includes steps for hyperparameter tuning and model evaluation.
- Reading Preprocessed Data.
- Selecting Important Feature.
- Assembling Features.
- Indexing the Label Column.
- Defining the Model.
- Creating a Pipeline.
- Hyperparameter Tuning.
- Splitting Data.
- Training the Model.
- Evaluating the Model.



