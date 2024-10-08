This code is associated with the paper from [Shah et. al., "Seasonal antigenic prediction of influenza A H3N2 using machine learning"](https://doi.org/10.1038/s41467-024-47862-9).

# Seasonal antigenic prediction of influenza A H3N2 using machine learning

**Syed Awais W. Shah<sup>1</sup>, Daniel P. Palomar<sup>1,2</sup>, Ian Barr<sup>3,4</sup>, Leo L. M. Poon<sup>5,6</sup>, Ahmed Abdul Quadeer<sup>1</sup>, and Matthew R. McKay<sup>4,7</sup>**

<sup>1</sup> Department of Electronic and Computer Engineering, The Hong Kong University of Science and Technology, Clear Water Bay, Hong Kong SAR, China

<sup>2</sup> Department of Industrial Engineering & Decision Analytics, The Hong Kong University of Science and Technology, Clear Water Bay, Hong Kong SAR, China

<sup>3</sup> WHO Collaborating Centre for Reference and Research on Influenza, Melbourne, Victoria, Australia

<sup>4</sup> Department of Microbiology and Immunology, University of Melbourne, at The Peter Doherty Institute for Infection and Immunity, Melbourne, Victoria, Australia

<sup>5</sup> School of Public Health, LKS Faculty of Medicine, The University of Hong Kong, Hong Kong SAR, China

<sup>6</sup> Centre for Immunology & Infection, Hong Kong Science Park, Hong Kong SAR, China

<sup>7</sup> Department of Electrical and Electronic Engineering, University of Melbourne, Melbourne, Victoria, Australia


## Abstract
Antigenic characterization of circulating influenza A virus (IAV) isolates is routinely assessed by using the hemagglutination inhibition (HI) assays for surveillance purposes. It is also used to determine the need for annual influenza vaccine updates as well as for pandemic preparedness. Performing antigenic characterization of IAV on a global scale is confronted with high costs, animal availability, and other practical challenges. Here we present a machine learning model that accurately predicts (normalized) outputs of HI assays involving circulating human IAV H3N2 viruses, using their hemagglutinin subunit 1 (HA1) sequences and associated metadata. Each season, the model learns an updated nonlinear mapping of genetic to antigenic changes using data from past seasons only. The model accurately distinguishes antigenic variants from non-variants and adaptively characterizes seasonal dynamics of HA1 sites having the strongest influence on antigenic change. Antigenic predictions produced by the model can aid influenza surveillance, public health management, and vaccine strain selection activities.

## Source codes
The source codes (compiled in Jupyter notebooks) regenerate the figures in the above-mentioned paper.

## Installation
1. Install [miniconda](https://conda.io/miniconda.html) or [anaconda](https://www.anaconda.com/)
2. Clone the SAP_H3N2_ML repository using "git clone https://github.com/saws-lab/SAP_H3N2_ML.git"
3. Create a virtual environment (e.g., named as SAP_H3N2_ML) with Python version 3.8.12 using "conda create --name SAP_H3N2_ML python=3.8.12"
4. Within the environment install the dependencies provided in file "[requirements.txt](https://github.com/saws-lab/SAP_H3N2_ML/blob/main/requirements.txt)"
5. Install Jupyter notebook in your base environment using "conda install jupyter"
6. Install nb_conda_kernel in your base environment using "conda install nb_conda_kernel"
7. Install Jupyter notebook kernels within the environment using "python -m ipykernel install --user --name SAP_H3N2_ML --display-name "Python (SAP_H3N2_ML)"

## Quickstart
1. Download data published with the paper as Supplementary Data 1.
2. Prepare input data as described in Materials and Methods section of the paper in the form of CSV file containing columns mentioned in file "[Supp_Fig2a_seasonal_data_distribution.ipynb](https://github.com/saws-lab/SAP_H3N2_ML/blob/main/src/SuppFig2a_seasonal_data_distribution.ipynb)".
3. To generate a figure, run the corresponding notebook. Each notebook contains step by step instructions to run the simulation and to regenerate the corresponding figure.

## Web application
The web server running the web application for seasonal antigenic prediction of IAV H3N2 using our proposed AdaBoost model can be accessed from Hugging Face Spaces (https://huggingface.co/spaces/sawshah/SAP_H3N2)

## Troubleshooting
For any questions or comments, please email at [awais.shah@unimelb.edu.au](mailto:awais.shah@unimelb.edu.au) or [sawshah@connect.ust.hk](mailto:sawshah@connect.ust.hk)
