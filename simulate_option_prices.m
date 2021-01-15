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
dates = datetime(dates, 'ConvertFrom', 'datenum');

% sub

nIndices  = size(Data,2)-1;     % # of indices

prices = Data(:,1:end-1);

yields = Data(:,end);             % daily effective yields
yields = 360 * log(1 + yields);   % continuously-compounded, annualized yield 

returns = tick2ret(prices,'Method','continuous');        % convert prices to returns
returns = returns - mean(returns);  % center the returns
index   = 2;                                       % France stored in column 2

% sub 

tailFraction = 0.1;               % decimal fraction allocated to each tail
tails = cell(nIndices,1);  % cell array of Pareto tail objects

for i = 1:nIndices
    tails{i} = paretotails(returns(:,i), tailFraction, 1 - tailFraction, 'kernel');
end

% sub 

minProbability = cdf(tails{index}, (min(returns(:,index))));
maxProbability = cdf(tails{index}, (max(returns(:,index))));

pLowerTail = linspace(minProbability  , tailFraction    , 200); % lower tail
pUpperTail = linspace(1 - tailFraction, maxProbability  , 200); % upper tail
pInterior  = linspace(tailFraction    , 1 - tailFraction, 200); % interior


limits = axis;
x = linspace(limits(1), limits(2));


U = zeros(size(returns));

for i = 1:nIndices
    U(:,i) = cdf(tails{i}, returns(:,i));    % transform each margin to uniform
end

options     = statset('Display', 'off', 'TolX', 1e-4);
[rhoT, DoF] = copulafit('t', U, 'Method', 'ApproximateML', 'Options', options);
rhoG        = copulafit('Gaussian', U);

 nPoints = 10000;                          % # of simulated observations

s = RandStream.getGlobalStream();
reset(s)

R = zeros(nPoints, nIndices);             % pre-allocate simulated returns array
U = copularnd('t', rhoT, DoF, nPoints);   % simulate U(0,1) from t copula

for j = 1:nIndices
    R(:,j) = icdf(tails{j}, U(:,j));
end

% Gaussian Copula
reset(s)
R = zeros(nPoints, nIndices);             % pre-allocate simulated returns array
U = copularnd('Gaussian', rhoG, nPoints); % simulate U(0,1) from Gaussian copula

for j = 1:nIndices
    R(:,j) = icdf(tails{j}, U(:,j));
end


reset(s)

% sub 

dt       = 1 / 252;                  % time increment = 1 day = 1/252 years
yields   = Data(:,end);              % daily effective yields
yields   = 360 * log(1 + yields);    % continuously-compounded, annualized yields
r        = mean(yields);             % historical 3M Euribor average
X        = repmat(100, nIndices, 1); % initial state vector
strike   = sum(X);                   % initialize an at-the-money basket

nTrials  = 10;                      % # of independent trials
nPeriods = 63;   % # of simulation periods: 63/252 = 0.25 years = 3 months

% sub 

sigma       = std(returns) * sqrt(252);    % annualized volatility
correlation = corrcoef(returns);           % correlated Gaussian disturbances
GBM1        = gbm(diag(r(ones(1,nIndices))), diag(sigma), 'StartState', X, ...
                 'Correlation'             , correlation);
             
% create second brownian copula with identity matrix

GBM2 = gbm(diag(r(ones(1    ,nIndices))), eye(nIndices), 'StartState', X);

% sub 

f = LongstaffSchwartz(nPeriods, nTrials);

% sub 

reset(s)

simByEuler(GBM1, nPeriods, 'nTrials'  , nTrials, 'DeltaTime', dt, ...
                          'Processes', f.LongstaffSchwartz);

BrownianMotionCallPrice = f.CallPrice(strike, r);
BrownianMotionPutPrice  = f.PutPrice (strike, r);

reset(s)

z = Example_CopulaRNG(returns * sqrt(252), nPeriods, 'Gaussian');
f = LongstaffSchwartz(nPeriods, nTrials);

simByEuler(GBM2, nPeriods, 'nTrials'  , nTrials, 'DeltaTime', dt, ...
                          'Processes', f.LongstaffSchwartz, 'Z', z);

GaussianCopulaCallPrice = f.CallPrice(strike, r);
GaussianCopulaPutPrice  = f.PutPrice (strike, r);

% sub 

reset(s)

z = Example_CopulaRNG(returns * sqrt(252), nPeriods, 't');
f =LongstaffSchwartz(nPeriods, nTrials);

simByEuler(GBM2, nPeriods, 'nTrials'  , nTrials, 'DeltaTime', dt, ...
                          'Processes', f.LongstaffSchwartz, 'Z', z);

tCopulaCallPrice = f.CallPrice(strike, r);
tCopulaPutPrice  = f.PutPrice (strike, r);

% sub 

disp(' ')
fprintf('                    # of Monte Carlo Trials: %8d\n'    , nTrials)
fprintf('        `            # of Time Periods/Trial: %8d\n\n'  , nPeriods)
fprintf(' Brownian Motion American Call Basket Price: %8.4f\n'  , BrownianMotionCallPrice)
fprintf(' Brownian Motion American Put  Basket Price: %8.4f\n\n', BrownianMotionPutPrice)
fprintf(' Gaussian Copula American Call Basket Price: %8.4f\n'  , GaussianCopulaCallPrice)
fprintf(' Gaussian Copula American Put  Basket Price: %8.4f\n\n', GaussianCopulaPutPrice)
fprintf('        t Copula American Call Basket Price: %8.4f\n'  , tCopulaCallPrice)
fprintf('        t Copula American Put  Basket Price: %8.4f\n'  , tCopulaPutPrice)
