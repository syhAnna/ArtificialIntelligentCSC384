function output = correlation(I, f, mode)
% Implements the correlation (for grayscale or color images 
% and 2D filters) between an input image and a given correlation filter. 
% The function must take as input: an input image 'I', a filter 'f', 
% and a string 'mode', that can either be 'valid', 'same' or 'full'. 
% The output must match what is specified by 'mode'.

% Read image and convert it to grayscale
imRGB = imread(I);
im = double(rgb2gray(imRGB));

% Different mode
switch mode
    case 'valid'
        output = corr_valid(im, f);
    case 'same'
       output = corr_same(im, f);
    case 'full'
       output = corr_full(im, f);
end
        
end