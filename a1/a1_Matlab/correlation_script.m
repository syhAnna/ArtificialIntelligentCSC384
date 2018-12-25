I = 'iris.jpg';
f = zeros(51);
f(26, 51) = 1;
mode = 'same';  % string: 'valid', 'same', 'full;

output = correlation(I, f, mode);
imshow(output);