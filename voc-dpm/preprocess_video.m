function preprocess_video(vid_fname, detection_thresh, frame_sample_interval)
%% init
if nargin<1
    vid_fname = '../optical_flow/videos/outfile.avi'; % Person approaches a chair.
end
if nargin < 2
    detection_thresh = -1.2;
end
if nargin < 3
    frame_sample_interval = 3; % Sample a frame from video once every X frames.
end

addpath ../optical_flow
addpath ../optical_flow/algorithms/CLG-TV/

init_obj_detect;
fname_split = regexp(vid_fname, '[\./]', 'split');
vid_name = fname_split{end-1};

obj = VideoReader(vid_fname);
video = obj.read();

%% Iterate on frames.
[boxes, classes, scores, centers, projected_centers ] = deal({});

Nframes = size(video,4);
t = 1; % Sampled frames counter.
for k=1:frame_sample_interval:(Nframes-frame_sample_interval)
    
    % Evaluate detections.
    im=video(:,:,:,k);
    tic
    [boxes{t}, classes{t}, scores{t}, classes_names]  = obj_detect_frame(im, detection_thresh);
    toc
    boxes{t} = ceil(boxes{t}); % convert to integer
    
    %% Evaluate optical flow.
    im2 = video (:,:,:,k+frame_sample_interval);
    [im1gray, im2gray] = acquistionSeq(im, im2);
    
    uvOF = OpticalFlowCLG_TV(im1gray, im2gray);
    fname_OF = ['../OF_frames/' vid_name '_OF_' num2str(t) '_' num2str(frame_sample_interval)];
    save (fname_OF, 'uvOF');
    
    % Evaluate detection boxes mean optical flow.
    
    n_detections = size(boxes{t},1);
    [centers{t}, projected_centers{t}] = deal(nan(n_detections ,2));
    
    for d=1:n_detections
        x1 = boxes{t}(d,1);
        x2 = boxes{t}(d,2);
        y1 = boxes{t}(d,3);
        y2 = boxes{t}(d,4);
        
        u = uvOF(:,:,1).';
        v = uvOF(:,:,2).';
        u_box = u(x1:x2, y1:y2);
        u_avg_box = mean(u_box(:));
        
        v_box = v(x1:x2, y1:y2);
        v_avg_box = mean(v_box(:));
        
        center_x = (x1+x2)/2;
        center_y = (y1+y2)/2;
        centers{t}(d,:) = [center_x, center_y];
        
        projected_center_x = center_x + u_avg_box;
        projected_center_y = center_y + v_avg_box;
        projected_centers{t}(d,:) = [projected_center_x, projected_center_y];
        
    end
    
    if false % (don't) visualize detections
        imshow(im);
        %     line([boxes{k}(:,1) boxes{k}(:,1) boxes{k}(:,2) boxes{k}(:,2) boxes{k}(:,1)]', [boxes{k}(:,3) boxes{k}(:,4) boxes{k}(:,4) boxes{k}(:,3) boxes{k}(:,3)]', 'color', 'r', 'linewidth', 3, 'linestyle', '-');
        for d=1:size(boxes{t},1)
            x1 = boxes{t}(d,1);
            x2 = boxes{t}(d,2);
            y1 = boxes{t}(d,3);
            y2 = boxes{t}(d,4);
            label = classes_names{classes{t}(d)};
            line([x1 x1 x2 x2 x1]', [y1 y2 y2 y1 y1]', 'color', 'r', 'linewidth', 3, 'linestyle', '-');
            text(x1, y1, label, 'Color', 'white');
        end
        drawnow
        shg
    end
    
    t=t+1; % Increase sampled frames counter.
end

%% save results
th_str = regexprep(num2str(detection_thresh), '-', 'm');
th_str = regexprep(th_str, '\.', '_');

res_fname = ['../preprocessed_videos/' vid_name '_detections_th' th_str];
save(res_fname, 'boxes', 'classes', 'scores', 'classes_names', 'centers', 'projected_centers' );


function uvOF = OpticalFlowCLG_TV(im1gray, im2gray)
%%settings
settings.lambda = 2200; % the weighting of the data term
settings.pyramid_factor = 0.5;
settings.resampling_method = 'bicubic'; % the resampling method used to build pyramids and upsample the flow
settings.warps = 5; % the number of warps per level
settings.interpolation_method = 'cubic'; % the interpolation method used for warping
settings.its = 10; % the number of iterations used for minimization
settings.use_diffusion = 1; % apply a weighting factor to the regularization term (the diffusion coefficient)
settings.use_bilateral = 1; % the data term weighting: bilateral or gaussian
settings.wSize = 5; % the window's size for the data fidelity term (Lukas-Kanade)
settings.sigma_d = settings.wSize/6; % sigma for the distance gaussian of the bilateral filter
settings.sigma_r = 0.1; % sigma for the range gaussian of the bilateral filter
settings.use_ROF_texture = 0; % apply ROF texture to the images (1 yes, 0 no)
settings.ROF_texture_factor = 0.95; % ROF texture; I = I - factor*ROF(I);
show_flow = 0; % display the flow during computation

tic
if show_flow
    fig_hnd = figure('Name', 'Optical flow');
    [u, v] = coarse_to_fine(im1gray, im2gray, settings, show_flow, fig_hnd);
    close all
else
    fig_hnd = nan;
    [u, v] = coarse_to_fine(im1gray, im2gray, settings, show_flow, fig_hnd);
end
toc

uvOF(:, :, 1) = u;
uvOF(:, :, 2) = v;

function [im1gray, im2gray] = acquistionSeq(im1, im2)
im1gray=mat2gray(im1);
im2gray=mat2gray(im2);

im1gray=rgb2gray(im1gray);
im2gray=rgb2gray(im2gray);
