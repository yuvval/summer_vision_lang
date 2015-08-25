% function [emission_scores, transition_scores] = generate_scores_from_2d_preprocessed_video(ppvid)

close all
ppvid = load('preprocessed_videos/outfile_detections_thm1.mat');
% 'vid_fname', 'boxes', 'classes', 'scores', 'classes_names', 'centers', 'projected_centers' 


figure
tx = [];
for t = 1:length(ppvid.projected_centers);
    n_det0 = size(ppvid.boxes{t},1);
    n_det1 = size(ppvid.boxes{t+1},1);
    crossp_ids = allcomb(1:n_det0, 1:n_det1);
    proj_centers0 = ppvid.projected_centers{t}(crossp_ids(:,1));
    centers1 = ppvid.centers{t+1}(crossp_ids(:,2));
    centers_diff = centers1-proj_centers0;
    centers_dist = sqrt(sum(centers_diff.^2,2));
    minus_dist = -centers_dist;
    tx = [tx;minus_dist];
end
tx(tx<-50) = []; %trimming all distances above 50 pixels
hist(tx, -50:1:0);
title ('transitions (minus) distances histogram')

figure
b = -4;
a = 0.5;
x = -20:1e-2:0;
y = 1./(1+exp(-a*(x-b)));
plot(x,y);shg
title ('transitions scores sigmoid')

%% detection scores sigmoid
b = -.8;
a = 10;
x = -1.5:1e-2:1;
y = 1./(1+exp(-a*(x-b)));
figure
plot(x,y);shg
title ('detection scores sigmoid')

% Nframes = length(boxes);