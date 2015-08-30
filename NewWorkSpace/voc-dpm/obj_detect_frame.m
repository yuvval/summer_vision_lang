function [boxes, classes, scores, classes_names] = obj_detect_frame(im, thresh, pca)

if nargin < 3
    pca = 5;
end
%% init
boxes = [];
classes = [];
scores = [];
classes_names = {};

%% get all detectors names
path = 'VOC2010/';
detectors_fnames = dir([ path '*.mat']);

N = length(detectors_fnames);

%% run detectors
for n=1:N
    load([path detectors_fnames(n).name])
    model_year = '2010';
    classes_names{n} = model.class;
%     if strcmp(model.class, 'inriaperson')
%         model_year = '2007';
%         classes_names{n} = 'person'; % renaming 'inriaperson' to 'person'
%     end
    csc_model = cascade_model(model, model_year, pca, thresh);
    pyra = featpyramid(double(im), csc_model);
    [dCSC, bCSC] = cascade_detect(pyra, csc_model, csc_model.thresh);
    [x1, x2, y1, y2, s] = getboxes(csc_model, im, dCSC, bCSC);
    n_boxes = length(s);
    if n_boxes > 0
        b = [x1, x2, y1, y2];
        
        boxes(end+1:end+n_boxes, 1:4) = b;        
        classes(end+1:end+n_boxes) = n;
        scores(end+1:end+n_boxes) = s;
        
        %     if ~isempty(x1)
        %         c = 'r'; %[160/255 0 0];
        %         line([x1 x1 x2 x2 x1]', [y1 y2 y2 y1 y1]', 'color', c, 'linewidth', 3, 'linestyle', '-');
        %         text(x1(1), y1(1), num2str(s(1)));
        %         shg
        % %         keyboard
        %     end
    end
end



function [x1, x2, y1, y2, s] = getboxes(model, image, det, all)
[x1, x2, y1, y2, s] = deal([]);
if ~isempty(det)
    try
        % attempt to use bounding box prediction, if available
        bboxpred = model.bboxpred;
        [det all] = clipboxes(image, det, all);
        [det all] = bboxpred_get(bboxpred, det, all);
    catch
        warning('no bounding box predictor found');
    end
    [det all] = clipboxes(image, det, all);
    I = nms(det, 0.5);
    det = det(I,:);
    all = all(I,:);
    boxes = det(:,1:4) ;

    x1 = boxes(:,1);
    y1 = boxes(:,2);
    x2 = boxes(:,3);
    y2 = boxes(:,4);
    s = all(:,end);
    % remove unused filters
    del = find(((x1 == 0) .* (x2 == 0) .* (y1 == 0) .* (y2 == 0)) == 1);
    x1(del) = [];
    x2(del) = [];
    y1(del) = [];
    y2(del) = [];
end
