function verb_tr_prob = verb_transition_probability(verb)

switch verb
    case 'approach'
        verb_tr_prob = [0.9 0.1 0;0 0.9 0.1; 0 0 1];
    otherwise
        error('unknown verb')
end

        