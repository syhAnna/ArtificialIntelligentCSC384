ig1 = 'colourTemplate.png';
ig2 = 'colourSearch.png';
[fC1, dC1] = sift_colour(ig1);
[fC2, dC2] = sift_colour(ig2);
matchC = matching(dC1, dC2, 0.5);
affineC = affine(fC1,fC2,matchC,4);
visualize_aff(ig1, ig2, affineC);