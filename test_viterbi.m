

%% compare with matlab's hmmviterbi
rng(50)
% tr = [0.95,0.05;
%     0.10,0.90];
% 
% e = [1/6,  1/6,  1/6,  1/6,  1/6,  1/6;
%     1/10, 1/10, 1/10, 1/10, 1/10, 1/2;];
% 
% [seq, states] = hmmgenerate(5,tr,e);
% estimatedStates = hmmviterbi(seq,tr,e);

tr = [0.6 0.4; 0.5 0.5];
e = [0.3, 0.2, 0.2, 0.3; 0.2, 0.3, 0.3, 0.2];
seq = [3 3 2 1 2 4 3 1 1];

[e_scores, tr_scores] = deal({});
for t=1:length(seq)
    e_scores{t} = log2(e(:,seq(t))).';
    
    if t<length(seq)
       tr_scores{t} = log2(tr);
    end
end

estimatedStates = hmmviterbi(seq,tr,e);
my_viterbi_estimatedStates = viterbi_yuval(e_scores, tr_scores, 0, 1);
% seq
% estimatedStates
% my_viterbi_estimatedStates.'
% frames_scores
if ~all(my_viterbi_estimatedStates == estimatedStates.')
    error('a bug in my viterbi implementation')
end

%% visualize sequence
if true

ppvid = load('preprocessed_videos/outfile_detections_thm1.mat');
[s_em, s_tr] = generate_scores_from_2d_preprocessed_video(ppvid);

seq = viterbi_yuval(s_em, s_tr, 0, 1);
    
frame_sample_interval = 3;
obj = VideoReader(['voc-dpm/' ppvid.vid_fname]);
video = obj.read();

boxes = ppvid.boxes;

t=1;
for k = 1:frame_sample_interval:size(video,4)
    
    im=video(:,:,:,k);
    imshow(im);
    
    d = seq(t);
    x1 = boxes{t}(d,1);
    x2 = boxes{t}(d,2);
    y1 = boxes{t}(d,3);
    y2 = boxes{t}(d,4);
    label = ppvid.classes_names{ppvid.classes{t}(d)};
    label = sprintf('%s, %2.3f', label, ppvid.scores{t}(d));
    line([x1 x1 x2 x2 x1]', [y1 y2 y2 y1 y1]', 'color', 'r', 'linewidth', 3, 'linestyle', '-');
    text(x1, y1, label, 'Color', 'white');
    drawnow;
    shg;
    pause(0.2)
    t=t+1;
end
end

