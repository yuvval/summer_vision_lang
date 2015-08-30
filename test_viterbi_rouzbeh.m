close all
clear

%% compare with matlab's hmmviterbi
rng(50)

tr = [0.6 0.4; 0.5 0.5];
e = [0.3, 0.2, 0.2, 0.3; 0.2, 0.3, 0.3, 0.2];
seq = [3 3 2 1 2 4 3 1 1];

[e_scores, tr_scores] = deal({});
for t=1:length(seq)
    e_scores{t} = log2(e(:,seq(t))).';
    
    if t<length(seq)
       tr_scores{t} = log2(tr);
    end
end

estimatedStates = hmmviterbi(seq,tr,e);
my_viterbi_estimatedStates = viterbi_yuval(e_scores, tr_scores, 0, 1);
% seq
% estimatedStates
% my_viterbi_estimatedStates.'
% frames_scores
if ~all(my_viterbi_estimatedStates == estimatedStates.')
    error('a bug in my viterbi implementation')
end

%% visualize sequence
if true

ppvid = load('preprocessed_videos/outfile_detections_thm1_1.mat');

% setting the tuning params for probabilities and features binning / sigmoiding
% emission probablities sigmoid params
tuning_params.other.sig_a = 10;
tuning_params.other.sig_b = -0.8;

tuning_params.person.sig_a = 5;
tuning_params.person.sig_b = -0.4;

tuning_params.chair.sig_a = 10;
tuning_params.chair.sig_b = -0.87;

% transition probablities sigmoid params
tuning_params.sig_a_trans = 0.3;
tuning_params.sig_b_trans = -4;

[s_em, s_tr, feat_per_tr] = generate_scores_from_2d_preprocessed_video(ppvid, tuning_params);

seq = viterbi_yuval(s_em, s_tr, 0, 1);
    
frame_sample_interval = 3;
obj = VideoReader(['voc-dpm/' ppvid.vid_fname]);
video = obj.read();

boxes = ppvid.boxes;

t=1;
feat_history = [];
for k = 1:frame_sample_interval:size(video,4)
    
    im=video(:,:,:,k);
    imshow(im);
    
    if t>1
        d_prev = seq(t-1);
    else
        d_prev = 1;
    end
    
    d = seq(t);
    x1 = boxes{t}(d,1);
    x2 = boxes{t}(d,2);
    y1 = boxes{t}(d,3);
    y2 = boxes{t}(d,4);
    label = ppvid.classes_names{ppvid.classes{t}(d)};
%     label = sprintf('%s, %2.3f', label, ppvid.scores{t}(d));
    feat_name = 'velocity_abs';
    feat_id = find(ismember(feat_per_tr.names, feat_name));
    
%     feat_val = ppvid.scores{t}(d);
    feat_val = feat_per_tr.values{t}(d_prev, d, feat_id);
    feat_history(end+1) = feat_val;
    label = sprintf('%s, %2.3f', label, feat_val);
    line([x1 x1 x2 x2 x1]', [y1 y2 y2 y1 y1]', 'color', 'r', 'linewidth', 3, 'linestyle', '-');
    text(x1, y1, label, 'Color', 'white');
    drawnow;
    shg;
    pause(0.5)
    t=t+1;
end
end
%figure
% hist(feat_history, 0:0.25:5);shg

%%
% s_em is for the emission score of each frame of the video
% s_tr is for the transmission score of each frame of the video
%feat_per_tr.value{1}...{30} is the feature vector for each frame of the
%video, in which the first element the class, the second and third are x
%and y centers, the forth element is the velocity bin which can be 1, 2, or
%3


% Calculating emission and transmission coefficients for person and chair
% which have the same values

em_person=log10(ones(1,30));
tr_person=log10(ones(1,29));
em_chair=log10(ones(1,30));
tr_chair=log10(ones(1,29));

% Calulating emission and transmission coefficients for the Word "Approach"
% in 3 different stages

% Transmission coefficient
epsilon=0.00001;
for i=1:29
Approach_tr{i}(1,1)=log10(.9);
Approach_tr{i}(1,2)=log10(.1-epsilon);
Approach_tr{i}(1,1)=log10(epsilon);

Approach_tr{i}(2,1)=log10(epsilon);
Approach_tr{i}(2,2)=log10(0.9);
Approach_tr{i}(2,3)=log10(.1-epsilon);

Approach_tr{i}(3,1)=log10(epsilon);
Approach_tr{i}(1,2)=log10(epsilon);
Approach_tr{i}(1,1)=log10(1-2*epsilon);
end

% Emission Coeffiecients

epsilon=0.00001;
for i=1:30
  a_1=find(feat_per_tr.values{1}(1,:,1)==15);
  a_2=find(feat_per_tr.values{1}(1,:,1)==9);
  Abs=sqrt((feat_per_tr.values{1}(1,a_1(1),2)-feat_per_tr.values{1}(1,a_2(1),2))^2 ...
      + (feat_per_tr.values{1}(1,a_1(1),3)-feat_per_tr.values{1}(1,a_2(1),3))^2);
  Velocity_person=feat_per_tr.values{1}(1,a_1(1),4);
  Velocity_chair=feat_per_tr.values{1}(1,a_2(1),4);
  
  if Velocity_person == 1 && Velocity_chair==1 && Abs>100
      Approach_em{i}(1)=log10(1-epsilon);
  else
      Approach_em{i}(1)=log10(epsilon);
  end
  
  if Velocity_person == 2 && Velocity_chair==1 && Abs>100
      Approach_em{i}(2)=log10(1-epsilon);
  else
      Approach_em{i}(2)=log10(epsilon);
  end
  
  if Velocity_person == 1 && Velocity_chair==1 && Abs<100
      Approach_em{i}(3)=log10(1-epsilon);
  else
      Approach_em{i}(3)=log10(epsilon);
  end
  
end

% so far I have Approach_em{1:30}(1:3), Approach_tr{1:29}(1:3,1:3),em_person,
% tr_person, em_chair, tr_chair, s_em, and s_tr, Now we need to unify them
% into one set of vector for each one of transmisiion and emission of 30
% frames

% Total Emission Coefficient
k=0;
K=ones(1,30);
for i=1:30
    K(i)=k;
    k=0;
    for j=1:size(s_em{i},1)
        for jj=1:size(s_em{i},1)
            for jjj=1:3
                k=k+1;
                Em{i,k,1}=s_em{i}(j,1)+s_em{i}(jj,1)+Approach_em{i}(jjj)+em_person(i)+em_chair(i);
                %here I try to keep all of the indices to help me in the
                %transition phase
                Em{i,k,2}=[j,jj,jjj];
            end
        end
    end
end


% Total Transmission coefficient

for i=1:29
    for I=1:K(i+1)
        for II=1:K(i)
            %tracker number one indice for the first frame
            Tk1_1=Em{i,II,2}(1);
            %tracker number two indice for the first frame
            Tk2_1=Em{i,II,2}(2);
            %Word(Approach) Indice for the first frame
            AR1=Em{i,II,2}(3);
            %tracker number one indice for the second frame
            Tk1_2=Em{i+1,I,2}(1);
            %tracker number two indice for the second frame
            Tk2_2=Em{i+1,I,2}(2);
            %Word(Approach) Indice for the second frame
            AR2=Em{i+1,I,2}(3);
            Tr{i,II,I}=Approach_tr{i}(AR1,AR2)+tr_chair(i)+tr_person(i)...
            +s_tr{i}(Tk1_1,Tk1_2)+s_tr{i}(Tk2_1,Tk2_2);
        end
    end   
end

