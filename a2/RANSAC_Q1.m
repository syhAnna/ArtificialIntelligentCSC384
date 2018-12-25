function S = RANSAC_Q1(p, P, k)
% In Q3.1, p = 0.7, P = 0.99, k = 1:20
S = log(1 - P) / log(1 - p^k);
end