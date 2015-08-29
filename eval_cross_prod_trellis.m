function [ cross_em_scores, cross_tr_scores, all_crossp_states_t1_t2_v_n1_n2 ] = eval_cross_prod_trellis( verb, noun1, noun2, tracker_scores, tracker_feats)

n_frames = length(tracker_feats.values);

[ cross_em_scores, cross_tr_scores, noun1_em_scores, noun2_em_scores, verb_em_scores, cross_p_all_hmms_states ] = deal({});

for t = 1:n_frames
    n_det0 = size(feat_per_tr.values{t}, 1);
    n_det1 = size(feat_per_tr.values{t}, 2);
    n_det_next = size(feat_per_tr.values{t+1}, 2);
    
    % caching scores per word and its observations upon the tracker
    for tracker_state = 1:length(tracker_scores{t}.em)
        noun1_em_scores{t} = error('todo'); % log(em_prob_noun(noun1, tracker_feats, t, tracker_state));
        noun2_em_scores{t} = error('todo'); % log(em_prob_noun(noun2, tracker_feats, t, tracker_state));
    end

    verb_tr_scores_mat = log(get_verb_tr_mat(verb));
    cross_p_trackers_states = allcomb(1:n_det1, 1:n_det1).';
    n_verb_states = size(verb_tr_scores_mat,1);
    for verb_state = 1:n_verb_states
        for states = cross_p_trackers_states
            verb_em_scores{t}(verb_state, states(1), states(2)) = error('todo'); % log(em_prob_verb(verb, verb_state, tracker_feats, t, states(1), states(2))); 
        end
    end
    
    cross_p_all_hmms_states{t} = allcomb(1:n_det1, 1:n_det1, 1:n_verb_states, 1, 1).';
    cross_p_all_hmms_states_next = allcomb(1:n_det_next, 1:n_det_next, 1:n_verb_states, 1, 1).';
    
    cross_tr_scores{t} = nan(length(cross_p_all_hmms_states{t}), length(cross_p_all_hmms_states_next));
    for comb_state = cross_p_all_hmms_states
        tracker1_state = comb_state(1);
        tracker2_state = comb_state(2);
        verb_state = comb_state(3);
        cross_em_scores{t} = tracker_scores{t}.em(tracker1_state) + tracker_scores{t}.em(tracker2_state) ...
            + verb_em_scores{t}(verb_state, tracker1_state, tracker2_state) ...
            + noun1_em_scores{t} + noun2_em_scores{t};
        
        
        %% TODO: Need to eval scores for transitions (x-products) and 
        if t<Nframes
%             cross_tr_scores{t} = trkxscor{t}(tracker1_state) + trkxscor{t}(tracker2_state) + verb_tr_scores_mat(verb_state)
        end

            
        
    end
end

end

