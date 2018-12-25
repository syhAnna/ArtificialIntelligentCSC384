% Script for q3.1
% In Q3.1, p = 0.7, P = 0.99, k = 1:20
figure

for x = 1:20
    y = RANSAC_Q1(0.7, 0.99, x);
    plot(x, y, 'o')
    hold on
end

title('Q3.1 Discrete Dot Plot')
xlabel('minimum number of sample points')
ylabel('required number of RANSAC iterations (S)')