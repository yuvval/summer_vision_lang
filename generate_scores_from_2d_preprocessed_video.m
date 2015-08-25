function [emission_scores_vec, transition_scores_mat] = generate_scores_from_2d_preprocessed_video(ppvid)

% parameters for sigmoid
sig_a_emis = 10;
sig_b_emis = -0.8;

sig_a_trans = 0.5;
sig_b_trans = -4;

% number of frames
Nframes = length(ppvid.boxes);

[emission_scores_vec, transition_scores_mat] = deal({});

% iterating per frame and generating emission prob. and transition prob.
for t=1:Nframes
    % eval emissions scores per frame
    emission_scores_vec{t} = log(sigmoid(ppvid.scores{t}, sig_a_emis, sig_b_emis));
    
    if t<Nframes
        % eval transtion scores per frame
        n_det0 = size(ppvid.boxes{t},1); % Number of detections for current frame.
        n_det1 = size(ppvid.boxes{t+1},1); % Number of detections for next frame.
        crossp_ids = allcomb(1:n_det0, 1:n_det1);
        proj_centers0 = ppvid.projected_centers{t}(crossp_ids(:,1));
        centers1 = ppvid.centers{t+1}(crossp_ids(:,2));
        centers_diff = centers1-proj_centers0;
        centers_dist = sqrt(sum(centers_diff.^2,2));
        minus_dist = -centers_dist;
        
        s_tran_vec = log(sigmoid(minus_dist, sig_a_trans, sig_b_trans));
        transition_scores_mat{t} = sparse(crossp_ids(:,1), crossp_ids(:,2), s_tran_vec);
        if t>1
            assert(size(transition_scores_mat{t-1},2) == length(emission_scores_vec{t})); % sanity check, error if false
        end
    end
    
end

function s = sigmoid(x, a, b)
s = 1./(1+exp(-a*(x-b)));

