init_obj_detect;

thresh = -1; % Todo: understand the ranges

vid_fname = '../optical_flow/videos/outfile.avi'; % person approaches a chair
obj = VideoReader(vid_fname);
video = obj.read();

[boxes, classes, scores ] = deal({});

for k=1:3:90 % TODO - add counter and save frames one after the other
    im=video(:,:,:,k);
    tic
    [boxes{k}, classes{k}, scores{k}, classes_names]  = obj_detect_frame(im, thresh);
    toc
    boxes{k} = ceil(boxes{k});
    
    imshow(im);
%     line([boxes{k}(:,1) boxes{k}(:,1) boxes{k}(:,2) boxes{k}(:,2) boxes{k}(:,1)]', [boxes{k}(:,3) boxes{k}(:,4) boxes{k}(:,4) boxes{k}(:,3) boxes{k}(:,3)]', 'color', 'r', 'linewidth', 3, 'linestyle', '-');
    for d=1:size(boxes{k},1)
        x1 = boxes{k}(d,1);
        x2 = boxes{k}(d,2); 
        y1 = boxes{k}(d,3);
        y2 = boxes{k}(d,4);
        label = classes_names{classes{k}(d)};
        line([x1 x1 x2 x2 x1]', [y1 y2 y2 y1 y1]', 'color', 'r', 'linewidth', 3, 'linestyle', '-');
        text(x1, y1, label, 'Color', 'white');
    end
    drawnow
    shg

end

th_str = regexprep(num2str(thresh), '-', 'm');
th_str = regexprep(th_str, '\.', '_');
res_fname = ['vid1_detections_th' th_str];
save(res_fname, 'boxes', 'classes', 'scores', 'classes_names');


