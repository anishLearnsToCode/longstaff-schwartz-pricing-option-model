% Mathematical Modelling and Simulation Project
% Using Longstaff & Schwartz Method for Pricing American Basket Options 
% Using Multivariate Geomrtric Brownian Motion (GBM) Process 
% Anish Sachdeva 
% DTU/2K16/MC/013
% Delhi Technological University
% Mathemtical Modelling and Simulation (MC-409)

clc;
clear;
close all;

% Import the Supporting Historical Dataset
% Load a daily historical dataset of 3-month Euribor, the trading dates 
% spanning the interval 07-Feb-2001 to 24-Apr-2006, and the closing index 
% levels of the following representative large-cap equity indices:
% 1. TSX Composite (Canada)
% 2. CAC 40 (France)
% 3. DAX (Germany)
% 4. Nikkei 225 (Japan)
% 5. FTSE 100 (UK)
% 6. S&P 500 (US)

load Data_GlobalIdx2;
dates = datetime(dates,'ConvertFrom', 'datenum');
