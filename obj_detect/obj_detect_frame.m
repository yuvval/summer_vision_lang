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
path = '~/externals/voc-dpm/VOC2010/';
detectors_fnames = dir([ path '*.mat']);

N = length(detectors_fnames);
N=3; % only for debug

%% run detectors
for n=1:N
    load([path detectors_fnames(n).name])
    classes_names{n} = model.class;
    csc_model = cascade_model(model, '2010', pca, thresh);
    pyra = featpyramid(double(im), csc_model);
    [dCSC, bCSC] = cascade_detect(pyra, csc_model, csc_model.thresh);
    b = getboxes(csc_model, im, dCSC, bCSC);
    
    % boxes = [];
    % classes = [];
    % scores = [];
    % classes_names = {};
    
end



function b = getboxes(model, image, det, all)
b = [];
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
    b = [det(:,1:4) all];
end
