% check whether horizontal derivative of Gaussian separable
[dx, dy] = gradient(fspecial('gaussian'));
disp(rank(dx)); % output rank = 1