close all
clear

%% visualize sequence
if true

ppvid = load('preprocessed_videos/outfile_detections_thm0_9.mat');

% setting the tuning params for probabilities and features binning / sigmoiding
% emission probablities sigmoid params
tuning_params.other.sig_a = 10;
tuning_params.other.sig_b = -0.8;

tuning_params.person.sig_a = 5;
tuning_params.person.sig_b = -0.4;

tuning_params.chair.sig_a = 10;
tuning_params.chair.sig_b = -0.87;

% transition probablities sigmoid params
tuning_params.sig_a_trans = 0.3;
tuning_params.sig_b_trans = -4;

[tracker_scores.em, tracker_scores.tr, tracker_feats] = generate_scores_from_2d_preprocessed_video(ppvid, tuning_params);

verb = 'approach';
noun1 = 'person';
noun2 = 'chair';
[ cross_em_scores, cross_tr_scores_mat, cross_p_all_hmms_states ] = eval_cross_prod_trellis( verb, noun1, noun2, tracker_scores, tracker_feats);

seq = viterbi_yuval(tracker_scores.em, tracker_scores.tr, 0, 1);
    
frame_sample_interval = 3;
obj = VideoReader(['voc-dpm/' ppvid.vid_fname]);
video = obj.read();

boxes = ppvid.boxes;

t=1;
feat_history = [];
for k = 1:frame_sample_interval:size(video,4)
    
    im=video(:,:,:,k);
    imshow(im);
    
    if t>1
        d_prev = seq(t-1);
    else
        d_prev = 1;
    end
    
    d = seq(t);
    x1 = boxes{t}(d,1);
    x2 = boxes{t}(d,2);
    y1 = boxes{t}(d,3);
    y2 = boxes{t}(d,4);
    label = ppvid.classes_names{ppvid.classes{t}(d)};
%     label = sprintf('%s, %2.3f', label, ppvid.scores{t}(d));
    feat_name = 'velocity_abs';
    feat_id = find(ismember(tracker_feats.names, feat_name));
    
%     feat_val = ppvid.scores{t}(d);
    feat_val = tracker_feats.values{t}(d_prev, d, feat_id);
    feat_history(end+1) = feat_val;
    label = sprintf('%s, %2.3f', label, feat_val);
    line([x1 x1 x2 x2 x1]', [y1 y2 y2 y1 y1]', 'color', 'r', 'linewidth', 3, 'linestyle', '-');
    text(x1, y1, label, 'Color', 'white');
    drawnow;
    shg;
    pause(0.5)
    t=t+1;
end
end

figure
% hist(feat_history, 0:0.25:5);shg