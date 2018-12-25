function [f,d] = sift_colour(img)
    im=rgb2hsv(imread(img));
    hue = single(im(:,:,1));
    [f,d]=vl_sift(hue);
end

