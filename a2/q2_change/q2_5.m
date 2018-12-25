img1 = 'colourTemplate.png';
img2 = 'colourSearch.png';
[fC1, dC1] = sift_colour(img1);
[fC2, dC2] = sift_colour(img2);
matchC = matching(dC1, dC2, 0.5);
affineC = affine(fC1,fC2,matchC,4);
visualize_aff(img1, img2, affineC);
