im = imread('portrait.jpg');

% Laplacian
H = fspecial('log');
out = imfilter(im, H);
imshow(out);
pause;

% Gradient
H = fspecial('gaussian');
h = gradient(H);
out = imfilter(im, h);
imshow(out);
pause;