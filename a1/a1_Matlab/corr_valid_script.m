im = 'iris.jpg';

% Read image and convert to grayscale
imRGB = imread(im);
I = double(rgb2gray(imRGB));

f = zeros(101);
f(51, 100) = 1;

output = corr_valid(I, f);
imshow(output);