function seq = viterbi_yuval(em_scores, tr_scores, prev_frame_scores, t)

Nframes = length(em_scores);

n_states = length(em_scores{t});
assignin('base', 'one', em_scores);
assignin('base', 'Two', tr_scores);
assignin('base', 'Three', prev_frame_scores);
assignin('base', 'Four', t);

if t>1
%     prev_frame_scores
    curr_tr_scores = tr_scores{t-1};
else    
    curr_tr_scores = -1*ones(1, n_states); % uniform prior distribution
end

frame_scores = nan(1, n_states);
for s=1:n_states
    frame_scores(s) = em_scores{t}(s) + max(prev_frame_scores(:) + curr_tr_scores(:,s));
end

[~, best_state] = max(frame_scores);

if t < Nframes
    seq = viterbi_yuval(em_scores, tr_scores, frame_scores, t+1);
end

if t == Nframes
    seq = best_state;
%     frames_scores = frame_scores;
else
    seq = [best_state; seq ];
%     frames_scores = [frame_scores; frames_scores];
end
