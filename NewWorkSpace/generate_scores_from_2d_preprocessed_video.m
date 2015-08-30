function [emission_scores_vec, transition_scores_mat, features_per_transition] = generate_scores_from_2d_preprocessed_video(ppvid, tuning_params)

tp = tuning_params;
% parameters for sigmoid

% number of frames
Nframes = length(ppvid.boxes);


[emission_scores_vec, transition_scores_mat, features_per_transition.values] = deal({});

% iterating per frame and generating emission prob. and transition prob.
for t=1:Nframes
%     ppvid.scores{t}(ppvid.classes{t} ~= 15) = -inf; %% ONLY for DEBUG: nulling prob all except given class % person class is 15, chair class is 9.

    % eval emissions scores per frame

    n_detections = length(ppvid.classes{t});
    emission_scores_vec{t} = nan(n_detections,1);
    for d = 1:n_detections
        if ppvid.classes{t}(d) == 15 % person
            sig_a = tp.person.sig_a;
            sig_b = tp.person.sig_b;
        elseif ppvid.classes{t}(d) == 9 % chair
            sig_a = tp.chair.sig_a;
            sig_b = tp.chair.sig_b;
        else
            sig_a = tp.other.sig_a;
            sig_b = tp.other.sig_b;
        end
        emission_scores_vec{t}(d) = log(sigmoid(ppvid.scores{t}(d), sig_a, sig_b));
    end
%     emission_scores_vec{t} = log(sigmoid(ppvid.scores{t}, tp.sig_a_emis, tp.sig_b_emis));
    
    if t<Nframes
        % eval transtion scores per frame
        n_det0 = size(ppvid.boxes{t},1); % Number of detections for current frame.
        n_det1 = size(ppvid.boxes{t+1},1); % Number of detections for next frame.
        crossp_ids = allcomb(1:n_det0, 1:n_det1);
        proj_centers0 = ppvid.projected_centers{t}(crossp_ids(:,1), :);
        centers1 = ppvid.centers{t+1}(crossp_ids(:,2), :);
        centers_diff = centers1-proj_centers0;
        centers_dist = sqrt(sum(centers_diff.^2,2));
        minus_dist = -centers_dist;
        
        s_tran_vec = log(sigmoid(minus_dist, tp.sig_a_trans, tp.sig_b_trans));
        transition_scores_mat{t} = full(sparse(crossp_ids(:,1), crossp_ids(:,2), s_tran_vec));
        if t>1
            assert(size(transition_scores_mat{t-1},2) == length(emission_scores_vec{t})); % sanity check, error if false
        end
    end
    
    % generate features vector
    [features_per_transition.values{t}, features_per_transition.names] = get_all_combinations_of_features_of_frame(t, ppvid);
    features_per_transition.classes_names = ppvid.classes_names;
end

function [all_comb_features_of_frame, features_per_transition_names] = get_all_combinations_of_features_of_frame(t, ppvid)

% features per transition (per frame) names
features_per_transition_names = {'class', 'center_x', 'center_y', 'velocity_binned', 'velocity_orientation', 'velocity_abs', 'velocity_angle'};

f_num = length(features_per_transition_names);

if t == 1
    n_det0 = 1; % Number of detections for current frame.
else
    n_det0 = size(ppvid.boxes{t-1},1); % Number of detections for previous frame.
end
n_det1 = size(ppvid.boxes{t},1); % Number of detections for current frame.
crossp_ids = allcomb(1:n_det0, 1:n_det1);
all_comb_features_of_frame = nan(n_det0, n_det1, f_num);

%% evaluating features values

% class
feat_name = 'class';
feat_id = find(ismember(features_per_transition_names, feat_name));
% all_comb_features_of_frame(crossp_ids(:,1), crossp_ids(:,2), feat_id) = ppvid.classes{t}(crossp_ids(:,2));
for k=1:length(crossp_ids(:,2))
    all_comb_features_of_frame(:, crossp_ids(k,2), feat_id) = ppvid.classes{t}(crossp_ids(k,2));
end


% center x coordinate
feat_name = 'center_x';
feat_id = find(ismember(features_per_transition_names, feat_name));
% all_comb_features_of_frame(crossp_ids(:,1), crossp_ids(:,2), feat_id) = ppvid.centers{t}(crossp_ids(:,2));
for k=1:length(crossp_ids(:,2))
    all_comb_features_of_frame(:, crossp_ids(k,2), feat_id) = ppvid.centers{t}(crossp_ids(k,2), 1);
end

% center y coordinate
feat_name = 'center_y';
feat_id = find(ismember(features_per_transition_names, feat_name));
% all_comb_features_of_frame(crossp_ids(:,1), crossp_ids(:,2), feat_id) = ppvid.centers{t}(crossp_ids(:,2));
for k=1:length(crossp_ids(:,2))
    all_comb_features_of_frame(:, crossp_ids(k,2), feat_id) = ppvid.centers{t}(crossp_ids(k,2), 2);
end

% velocity_binned
feat_name = 'velocity_binned';
feat_id = find(ismember(features_per_transition_names, feat_name));

[velocity_binned, velocity_orientation, velocity_abs,velocity_angle] = deal(nan(n_det0*n_det1,1));
n_frames = length(ppvid.boxes);

if t == n_frames

    velocity_binned(:) = 0;
    velocity_orientation(:) = 0; % 0 = no orientation, 1 = E, 2 = NE, 3 = N, 4 = NW, ... , 8 = SE
    velocity_abs(:) = 0;
    velocity_angle(:) = nan;
else
%     centers0 = ppvid.centers{t-1}(crossp_ids(:,1),:);
    centers = ppvid.centers{t}(crossp_ids(:,2),:);
    projected_centers = ppvid.projected_centers{t}(crossp_ids(:,2),:);
    velocity = projected_centers - centers;
    velocity_abs = sqrt(sum(velocity.^2,2));
    velocity_angle = angle(velocity(:,1) + 1i*velocity(:,2))*180/pi;
   
    % velocity angle
    % binning velocity orientation
    orientation_bins = [-180, -135, -45, 45, 180-45, 180];
    abs_vel_bins = [0, 1.5, 15, 1e5];
%     velocity_orientation = nan(n_det0*n_det1,1);
%     velocity_binned = nan(n_det0*n_det1,1);
    for k=1:length(crossp_ids(:,2))
        velocity_binned(k) = bin_real_number(velocity_abs(k),abs_vel_bins);
        if velocity_binned(k) > 1
            velocity_orientation(k) = bin_real_number(velocity_angle(k),orientation_bins);
            
        else % if velocity is too slow, then set angle / orientation to n/a
            velocity_orientation(k) = 0;
            velocity_angle(k) = nan;
        end
        
    end
    velocity_orientation(velocity_orientation == 5) = 1; % unwrap angles around +-180 deg

end

feat_name = 'velocity_binned';
feat_id = find(ismember(features_per_transition_names, feat_name));
% all_comb_features_of_frame(crossp_ids(:,1), crossp_ids(:,2), feat_id) = velocity_binned;
for k=1:length(crossp_ids(:,2))
    all_comb_features_of_frame(crossp_ids(k,1), crossp_ids(k,2), feat_id) = velocity_binned(k);
end

feat_name = 'velocity_orientation';
feat_id = find(ismember(features_per_transition_names, feat_name));
% all_comb_features_of_frame(crossp_ids(:,1), crossp_ids(:,2), feat_id) = velocity_orientation;
for k=1:length(crossp_ids(:,2))
    all_comb_features_of_frame(crossp_ids(k,1), crossp_ids(k,2), feat_id) = velocity_orientation(k);
end



feat_name = 'velocity_angle'; % saving for debug and statitics, probably won't use this feature directly.
feat_id = find(ismember(features_per_transition_names, feat_name));
% all_comb_features_of_frame(crossp_ids(:,1), crossp_ids(:,2), feat_id) = velocity_angle;
for k=1:length(crossp_ids(:,2))
    all_comb_features_of_frame(crossp_ids(k,1), crossp_ids(k,2), feat_id) = velocity_angle(k);
end

feat_name = 'velocity_abs'; % saving for debug and statitics, probably won't use this feature directly.
feat_id = find(ismember(features_per_transition_names, feat_name));
% all_comb_features_of_frame(crossp_ids(:,1), crossp_ids(:,2), feat_id) = velocity_abs;
for k=1:length(crossp_ids(:,2))
    all_comb_features_of_frame(crossp_ids(k,1), crossp_ids(k,2), feat_id) = velocity_abs(k);
end

function bin_id = bin_real_number(r, bins_partition)
    bin_id = find((r <= bins_partition),1)-1;
function s = sigmoid(x, a, b)
s = 1./(1+exp(-a*(x-b)));

