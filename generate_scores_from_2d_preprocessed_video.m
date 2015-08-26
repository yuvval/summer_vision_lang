function [emission_scores_vec, transition_scores_mat, features_per_transition] = generate_scores_from_2d_preprocessed_video(ppvid, tuning_params)

tp = tuning_params;
% parameters for sigmoid

% number of frames
Nframes = length(ppvid.boxes);


[emission_scores_vec, transition_scores_mat, features_per_transition.values] = deal({});

% iterating per frame and generating emission prob. and transition prob.
for t=1:Nframes
    %ppvid.scores{t}(ppvid.classes{t} ~= 9) = -inf; %% ONLY for DEBUG: nulling prob all except given class % person class is 15, chair class is 9.

    % eval emissions scores per frame
    emission_scores_vec{t} = log(sigmoid(ppvid.scores{t}, tp.sig_a_emis, tp.sig_b_emis));
    
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
        
        s_tran_vec = log(sigmoid(minus_dist, tp.sig_a_trans, tp.sig_b_trans));
        transition_scores_mat{t} = sparse(crossp_ids(:,1), crossp_ids(:,2), s_tran_vec);
        if t>1
            assert(size(transition_scores_mat{t-1},2) == length(emission_scores_vec{t})); % sanity check, error if false
        end
    else 
        % generate last features vector
        % TBD
    end
    
end

function [all_comb_features_per_frame] = get_all_combinations_of_features_per_frame(t, crossp_ids, ppvid)

% features per transition (per frame) names
features_per_transition_names = {'class', 'center', 'velocity_binned', 'velocity_orientation', 'prev_class'};

f_num = length(features_per_transition_names);

if t == 1
    n_det0 = 1; % Number of detections for current frame.
else
    n_det0 = size(ppvid.boxes{t-1},1); % Number of detections for previous frame.
end
n_det1 = size(ppvid.boxes{t},1); % Number of detections for current frame.
all_comb_features_per_frame = nan(n_det0, n_det1, f_num);

%% evaluating features values

% class
feat_name = 'class';
feat_id = find(ismember(features_per_transition_names, feat_name));
all_comb_features_per_frame(crossp_ids(:,1), crossp_ids(:,2), feat_id) = ppvid.classes{t}(crossp_ids(:,2));

% center
feat_name = 'center';
feat_id = find(ismember(features_per_transition_names, feat_name));
all_comb_features_per_frame(crossp_ids(:,1), crossp_ids(:,2), feat_id) = ppvid.centers{t}(crossp_ids(:,2));


% velocity_binned
feat_name = 'velocity_binned';
feat_id = find(ismember(features_per_transition_names, feat_name));

if t == 1
    velocity_binned = 0;
    velocity_orientation = 0; % 0 = no orientation, 1 = E, 2 = NE, 3 = N, 4 = NW, ... , 8 = SE
else
    proj_centers0 = ppvid.projected_centers{t-1}(crossp_ids(:,1));
    centers1 = ppvid.centers{t}(crossp_ids(:,2));
    velocity = centers1-proj_centers0;
    velocity_abs = sqrt(sum(velocity.^2,2));
    velocity_angle = angle(velocity(:,1) + 1i*velocity(:,2))*180/pi;
    
    velocity_binned = nan;

    % velocity angle
    % binning velocity orientation
    orientation_bins = -22.5:45:(360-22.5);
    abs_vel_bins = [0, 3, 10, 1e5];
    velocity_orientation = nan(n_det1,1);
    velocity_binned = nan(n_det1,1);
    for k=1:n_det1
        [~, velocity_orientation(k)] = min(abs(velocity_angle(k) - orientation_bins))
        [~, velocity_binned(k)] = min(abs(velocity_abs(k) - velocity_binned))
    end

end
all_comb_features_per_frame(crossp_ids(:,1), crossp_ids(:,2), feat_id) = velocity_binned;

feat_name = 'velocity_binned';
feat_id = find(ismember(features_per_transition_names, feat_name));
all_comb_features_per_frame(crossp_ids(:,1), crossp_ids(:,2), feat_id) = velocity_binned;

feat_name = 'velocity_orientation';
feat_id = find(ismember(features_per_transition_names, feat_name));
all_comb_features_per_frame(crossp_ids(:,1), crossp_ids(:,2), feat_id) = velocity_orientation;

% previous class
feat_name = 'prev_class';
feat_id = find(ismember(features_per_transition_names, feat_name));
all_comb_features_per_frame(crossp_ids(:,1), crossp_ids(:,2), feat_id) = ppvid.classes{t}(crossp_ids(:,1));


function s = sigmoid(x, a, b)
s = 1./(1+exp(-a*(x-b)));

