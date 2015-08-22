init_obj_detect;

im_fname = '000034.jpg'; % cars, person, buses
im = imread(im_fname);

thresh = -1; % Todo: understand the ranges

tic
[boxes, classes, scores, classes_names]  = obj_detect_frame(im, thresh);
toc

imshow(im);
line([boxes(:,1) boxes(:,1) boxes(:,2) boxes(:,2) boxes(:,1)]', [boxes(:,3) boxes(:,4) boxes(:,4) boxes(:,3) boxes(:,3)]', 'color', 'r', 'linewidth', 3, 'linestyle', '-');
for k=1:size(boxes,1)
    x1 = boxes(k,1);
    x2 = boxes(k,2);
    y1 = boxes(k,3);
    y2 = boxes(k,4);
    label = classes_names{classes(k)};
    line([x1 x1 x2 x2 x1]', [y1 y2 y2 y1 y1]', 'color', 'r', 'linewidth', 3, 'linestyle', '-');
    text(x1, y1, label, 'Color', 'white');    
end
shg
