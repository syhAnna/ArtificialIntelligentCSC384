% Script
im = imread('whereswaldo.jpg');
filter = imread('waldo.jpg');
output = findMan(im, filter);