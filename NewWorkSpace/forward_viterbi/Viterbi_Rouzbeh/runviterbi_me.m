
states = classes;
obs = classes;

%Uniform distribution for start probability
start_p = (1/length(classes{1}))*ones(1,length(classes{1}));

%emit_p = centers-projected_centers;
emit_p=cell(1,length(projected_centers));
for i=1:length(projected_centers)
    a=[];
    for j=1:length(projected_centers{i})
        for jj=1:length(centers{i+1})
            a(j,jj)=norm(abs(centers{i+1}(jj,:)-projected_centers{i}(j,:)));
        end
    end
    emit_p{i}=a;
end
%Transition probability
trans_p=scores;
[total, argmax, valmax] = forward_viterbi_edit(obs,states,start_p,trans_p,emit_p);