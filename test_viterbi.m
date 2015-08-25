
ppvid = load('preprocessed_videos/outfile_detections_thm1_1.mat');
[s_em, s_tr] = generate_scores_from_2d_preprocessed_video(ppvid);

seq = viterbi_yuval(s_em, s_tr, 1);


%% visualize sequence

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


