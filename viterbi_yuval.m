function seq = viterbi_yuval(em_scores, tr_scores, t)

Nframes = length(em_scores);

if t>1
    curr_tr_scores = sum(tr_scores{t-1},1);
else
    curr_tr_scores = 0; % uniform prior distribution
end

frame_scores = em_scores{t} + curr_tr_scores;

if t < Nframes
    seq = viterbi_yuval(em_scores, tr_scores, t+1);
end

[~, best_state] = max(frame_scores);
if t == Nframes
    seq = best_state;
else
    seq = [best_state; seq ];
end
