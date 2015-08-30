function [ cross_em_scores, cross_tr_scores_mat, cross_p_all_hmms_states ] = eval_cross_prod_trellis( verb, noun1, noun2, tracker_scores, tracker_feats)

Nframes = length(tracker_feats.values);

[ cross_em_scores, cross_tr_scores_mat, noun1_em_scores, noun2_em_scores, verb_em_scores, cross_p_all_hmms_states ] = deal({});

for t = 1:Nframes
    n_det1 = size(tracker_feats.values{t}, 2);
    
    % caching scores per word and its observations upon the tracker
    [noun1_em_scores{t}, noun2_em_scores{t}] = deal(nan(length(tracker_scores.em{t}),1));
    for tracker_state = 1:length(tracker_scores.em{t})
        noun1_em_scores{t}(tracker_state) = log( compute_emission_probability_noun(noun1, tracker_feats, t, tracker_state));
        noun2_em_scores{t}(tracker_state) = log( compute_emission_probability_noun(noun2, tracker_feats, t, tracker_state));
    end
    
    verb_tr_scores_mat = log(verb_transition_probability(verb));
    cross_p_trackers_states = allcomb(1:n_det1, 1:n_det1).';
    n_verb_states = size(verb_tr_scores_mat,1);
    for verb_state = 1:n_verb_states
        for states = cross_p_trackers_states
            verb_em_scores{t}(states(1), states(2), verb_state) = log(compute_emission_probability_verb(verb, verb_state, tracker_feats, t, states(1), states(2)));
        end
    end
    
    cross_p_all_hmms_states{t} = allcomb(1:n_det1, 1:n_det1, 1:n_verb_states, 1, 1).';
    n_curr_all_states = size(cross_p_all_hmms_states{t},2);
    
    cross_em_scores{t} = nan(n_curr_all_states, 1);
    cnt = 1;
    for comb_state = cross_p_all_hmms_states{t}
        tracker1_state = comb_state(1);
        tracker2_state = comb_state(2);
        verb_state = comb_state(3);
        cross_em_scores{t}(cnt) = tracker_scores.em{t}(tracker1_state) + tracker_scores.em{t}(tracker2_state) ...
            + verb_em_scores{t}(tracker1_state, tracker2_state, verb_state) ...
            + noun1_em_scores{t}(tracker1_state) + noun2_em_scores{t}(tracker2_state);
        cnt = cnt+1;
    end
    assert(all(~isnan(cross_em_scores{t}))); % sanity check. error if fails.
    
    
    %% Eval scores for transitions (x-products) and
    if t<Nframes
        n_det_next = size(tracker_feats.values{t+1}, 2);
        cross_p_all_hmms_states_next = allcomb(1:n_det_next, 1:n_det_next, 1:n_verb_states, 1, 1).';
        n_next_all_states = size(cross_p_all_hmms_states_next,2);
        
        cross_tr_scores_mat{t} = nan(n_curr_all_states, n_next_all_states);
        
        all_transitions = allcomb(1:n_curr_all_states, 1:n_next_all_states).';
        for transition = all_transitions
            comb_state_curr = cross_p_all_hmms_states{t}(:, transition(1));
            trkr1_state_curr = comb_state_curr(1);
            trkr2_state_curr = comb_state_curr(2);
            verb_state_curr = comb_state_curr(3);
            
            comb_state_next = cross_p_all_hmms_states_next(:, transition(2));
            trkr1_state_next = comb_state_next(1);
            trkr2_state_next = comb_state_next(2);
            verb_state_next = comb_state_next(3);
            
            curr_tr_score = tracker_scores.tr{t}(trkr1_state_curr, trkr1_state_next) ...
                + tracker_scores.tr{t}(trkr2_state_curr, trkr2_state_next) ...
                + verb_tr_scores_mat(verb_state_curr, verb_state_next);
            cross_tr_scores_mat{t}(transition(1), transition(2)) = curr_tr_score;
        end
    end
    if t>1
        assert(size(cross_tr_scores_mat{t-1},2) == length(cross_em_scores{t})); % sanity check, error if false
    end
    
    
    
    
end

end

