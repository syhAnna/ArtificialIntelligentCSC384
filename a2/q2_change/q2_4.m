im1= imread('book.jpeg');
img1=single(rgb2gray(im1));
[f1, d1] = vl_sift(img1);
im2= imread('findBook.png');
img2=single(rgb2gray(im2));
[f2, d2] = vl_sift(img2);

match = matching(d1, d2, 0.8);
aff = affine(f1,f2,match,4);
visualize_aff('book.jpeg', 'findBook.png', aff);
