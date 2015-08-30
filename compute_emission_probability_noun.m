function [em_prob_noun] = compute_emission_probability_noun(noun_name, tracker_feats, n_frame, n_tracker_state)
%tracker_feats=feat_per_tr;
%I should find if the class of each state (detection) of each frame is a certain noun

feat_name = 'class';
feat_id = find(ismember(tracker_feats.names, feat_name));
class_num_tracker = tracker_feats.values{n_frame}(1, n_tracker_state, feat_id)  %it gives the class of this detection

class_num = find(ismember(tracker_feats.classes_names, noun_name));  %it maps the noum_name to the class_num
    
if class_num_tracker== class_num            %e.g. if the class number of the detection is equal to the class number of tha name; 1= index to find the class in the features vector
   em_prob_noun = 1;
else
    em_prob_noun = 10^-6;
end
