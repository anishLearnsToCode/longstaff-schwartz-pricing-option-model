function f = LongstaffSchwartz(numTimes, numPaths)
%EXAMPLE_LONGSTAFFSCHWARTZ Price American basket option
%
% Syntax:
%
%   f = Example_LongstaffSchwartz(numTimes,numPaths)
%
% Description:
%
%   End-of-period processing function to price American basket options on a
%   portfolio of assets by Monte Carlo simulation, assuming a constant
%   risk-free rate. The valuation is based on the technique of Longstaff
%   and Schwartz [1], and approximates the continuation function by least
%   squares regression using a 3rd-order polynomial: y = F(x) = a + b*x +
%   c*x^2 + d*x^3. The function also illustrates how to update, access, and
%   share information among several nested functions. 
%
% Input Arguments:
%
%   numTimes - Number of simulation times steps (numPeriods*numSteps).
%   numPaths - Number of independent sample paths (simulation trials).
%
% Output Argument:
%
%   f - Structure of nested function handles to price American options.
%
% Reference:
%
%   [1] Longstaff F.A., and E.S. Schwartz. "Valuing American Options by
%   Simulation: A Simple Least-Squares Approach." The Review of Financial 
%   Studies. Vol. 14, 2001, pp. 113-147.

% Copyright 1999-2010 The MathWorks, Inc.
% $Revision: 1.1.4.1 $   $Date: 2013/10/09 06:17:39 $

prices = zeros(numPaths,numTimes); % Pre-allocate price matrix
times = zeros(numTimes,1);         % Pre-allocate sample time vector
iTime = 0;                         % Counter for the period index
iPath = 1;                         % Counter for the trial index
tStart = 0;                        % Initialize starting time
weights = 1;                       % Initialize portfolio weights

f.LongstaffSchwartz = @saveBasketPrices; % End-of-period processing
f.CallPrice = @getCallPrice;             % Call option pricing utility
f.PutPrice = @getPutPrice;               % Put option pricing utility
f.Prices = @getBasketPrices;             % Portfolio price access utility

function X = saveBasketPrices(t,X) % The actual workhorse, f(t,X)
  if iTime == 0 % Account for initial validation
     iTime = 1;
     tStart = t;                 % Record time of the initial condition
     weights = ones(1,numel(X)); % Assume 1 share of each asset
  else % Simulation is live

     % The following line forms the portfolio price series by adding the 
     % current prices of the individual assets, converting an N-D portfolio
     % of equities into a 1-D basket upon which a basket option is priced.
     % The portfolio price series represents the total purchase price of a
     % single share of each individual asset, although the weight vector
     % could be anything.

     prices(iPath,iTime) = weights*X;

     if iPath == 1
        times(iTime) = t;  % Save the sample time
     end
     if iTime < numTimes
        iTime = iTime + 1; % Update the period counter
     else % The last period of the current trial
        iTime = 1;         % Re-set the time period counter
        iPath = iPath + 1; % A new trial has begun
     end
  end
end

function value = getOptionPrice(strike,rate,optionClass)

% Model the continuation function as a 3rd-order polynomial.

  F = @(x,a,b,c,d)(a + b*x + c*x.^2 + d*x.^3);

% Pre-allocate option cash flow matrix, and initialize the terminal value
% at expiration to the cash flow of its European counterpart:

  V = zeros(numPaths, numTimes); % Option cash flow matrix

  if (nargin <= 2) || strcmpi(optionClass,'Call') 
     isCallOption = true; % Call option
     V(:,end) = max(prices(:,end)-strike, 0);
  else
     isCallOption = false; % Put option
     V(:,end) = max(strike-prices(:,end),0);
  end

% Step backward through time, successively updating the option cash flow
% matrix via OLS regression of in-the-money sample paths:

  for iTime = (numTimes-1):-1:1

      % Find all in-the-money sample paths, and format the regression
      % arrays. If no in-the-money paths are found, then assume the option
      % is not exercised:

      if isCallOption
         inTheMoney = find(prices(:,iTime) > strike);  % Call option
      else
         inTheMoney = find(prices(:,iTime) < strike);  % Put option
      end

      if ~isempty(inTheMoney)

         % For all in-the-money sample paths, identify any positive option
         % cash flows that occur after the current sample time, the sample
         % times at which the cash flows occur, and the column indices of
         % the cash flow matrix associated with them. If there is no
         % subsequent cash flow for a given path, then set the column index
         % to one just so indexing operations do not produce an error.

         iCashFlows = V(inTheMoney,(iTime+1):end) > 0;    % Find positive CFs
         tNext = iCashFlows*times((iTime+1):numTimes);    % Time of next CF
         iNext = max(iCashFlows*((iTime+1):numTimes)',1); % Index of next CF

         % Format the regression matrices, which include the current prices
         % of the underlier of all in-the-money sample paths (X), and the
         % discounted value of subsequent option cask flows associated with
         % the same paths (Y). If there is no subsequent cash flow for a
         % given path, then set the discount factor to zero just to be
         % clear. The following code segment assumes a constant risk-free
         % discount rate.

         % To improve numerical stability, normalize X and Y by the strike
         % price:

         X = prices(inTheMoney,iTime)/strike; % In-the-money prices
         D = exp(-rate*(tNext-times(iTime))); % Discount factors
         D(D > 1) = 0;                        % Allow for no future CFs
         Y = (V(sub2ind(size(V),inTheMoney,iNext)).*D)/strike;

         % Perform the OLS regression:

         OLS = [ones(numel(X),1) X  X.^2 X.^3]\Y;

         % Determine the intrinsic value of immediate exercise and the
         % value of continuation (scaled back to actual, unnormalized CFs):

         continuationValue = F(X,OLS(1),OLS(2),OLS(3),OLS(4))*strike;
         X = X*strike;

         if isCallOption
            intrinsicValue = max(X-strike,0);
         else
            intrinsicValue = max(strike-X,0);
         end

         iExercise = intrinsicValue > continuationValue;

         % Update the option cash flow matrix if immediate exercise is more 
         % valuable than continuation. Note that continuation values are
         % not inserted into the option cash flow matrix, but rather
         % determine whether or not intrinsic values are. Also, since the
         % option can only be exercised once, overwrite all non-zero cash
         % flows after the current time.

         V(inTheMoney(iExercise),iTime) = intrinsicValue(iExercise);
         V(inTheMoney(iExercise),(iTime+1):end) = 0;

      end
  end

  % Value the option by discounting the cash flows in the option cash flow
  % matrix back to the initial time (tStart), assuming a constant risk-free
  % discount rate:

  stoppingTimes = (V > 0)*times;
  value = mean(sum(V,2).*exp(-rate.*(stoppingTimes-tStart)));

end

function value = getCallPrice(strike,rate)
  value = getOptionPrice(strike,rate,'Call');
end

function value = getPutPrice(strike,rate)
  value = getOptionPrice(strike,rate,'Put');
end

function value = getBasketPrices()
  value = permute(prices,[2 3 1]); % Re-order to time-series format
end

end % End of outer/primary function
