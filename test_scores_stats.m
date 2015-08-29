% function [emission_scores, transition_scores] = generate_scores_from_2d_preprocessed_video(ppvid)

close all
ppvid = load('preprocessed_videos/outfile_detections_thm1.mat');
% 'vid_fname', 'boxes', 'classes', 'scores', 'classes_names', 'centers', 'projected_centers' 


figure
tx = [];
scores_history = [];
for t = 1:length(ppvid.projected_centers);
%     ppvid.scores{t}(ppvid.classes{t} ~= 15) = -inf; %% ONLY for DEBUG: nulling prob all except given class % person class is 15, chair class is 9.
    n_det0 = size(ppvid.boxes{t},1);
    n_det1 = size(ppvid.boxes{t+1},1);
    crossp_ids = allcomb(1:n_det0, 1:n_det1);
    proj_centers0 = ppvid.projected_centers{t}(crossp_ids(:,1), :);
    centers1 = ppvid.centers{t+1}(crossp_ids(:,2), :);
    centers_diff = centers1-proj_centers0;
    centers_dist = sqrt(sum(centers_diff.^2,2));
    minus_dist = -centers_dist;
    tx = [tx;minus_dist];
    scores_tmp = ppvid.scores{t}(ppvid.classes{t} == 3);
    scores_history = [scores_history; scores_tmp(:)];
end
hist(scores_history);

% tx(tx<-50) = []; %trimming all distances above 50 pixels
% hist(tx);
% hist(tx, -50:1:0);
% title ('transitions (minus) distances histogram')
% 
% figure
% b = -4;
% a = 0.2;
% x = -50:1e-2:0;
% y = 1./(1+exp(-a*(x-b)));
% plot(x,y);shg
% title ('transitions scores sigmoid')
% 
%% detection scores sigmoid (person)
% % person sigmoid params
% b = -0.4;
% a = 5;

% chair sigmoid params
b = -0.87;
a = 10;

x = -1.5:1e-2:1;
y = 1./(1+exp(-a*(x-b)));
figure
plot(x,y);shg
title ('detection scores sigmoid')

% Nframes = length(boxes);