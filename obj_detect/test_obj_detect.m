init_obj_detect;

im_fname = '000034.jpg'; % cars, person, buses
im = imread(im_fname);

thresh = -0.5; % Todo: understand the ranges


obj_detect_frame(im, thresh);


