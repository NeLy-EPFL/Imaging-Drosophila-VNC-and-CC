function w = compute_motion(I1,I2,I1Match,I2Match,fnDeepMatching,param,t)
%% Description
% Compute motion with a brigthness constancy data term defined on (I1,I2)
% and a feature matching similarity term defined on (I1Match,I2Match). The
% feature matches are computed by the DeepMatching method [Revaud et al. 2016], with the
% source code provided by the authors
% For more details see the paper: "Imaging neural activity in the ventral
% nerve cord of behaving adult Drosophila", bioRxiv
%
% [Revaud et al. 2016] J. Revaud, P. Weinzaepfel, Z. Harchaoui and C. Schmid (2016) "DeepMatching:
% Hierarchical Deformable Dense Matching. Int J Comput Vis 120:300–323
%% Input
% I1,I2: Images used for the brightness constancy term
% I1Match,I2Match: Images used by the feature matching similarity term
% fnDeepMatching: filename of the deepmatching code
% param: parameters of the algorithm (see 'default_parameters.m')
% t: number of the frame in the image sequence
%
%     Copyright (C) 2017 D. Fortun, denis.fortun@epfl.ch
%
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
%
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.

sz0=size(I1);
I1=padarray(I1,[15,15],'replicate');
I2=padarray(I2,[15,15],'replicate');

%% Feature matching
mkdir('tmp');
fnI1=['tmp/I1_',num2str(t),'.png'];
fnI2=['tmp/I2_',num2str(t),'.png'];
fnMatch=['tmp/match_',num2str(t),'.txt'];
imwrite(uint8(I1Match),fnI1);
imwrite(uint8(I2Match),fnI2);
command = [fnDeepMatching,'/deepmatching ', fnI1,' ', fnI2, ' -out ',fnMatch]; s = system(command);
command = ['rm ',fnI1]; s = system(command);
command = ['rm ',fnI2]; s = system(command);
formatSpec = '%u %u %u %u %f %u';
sizeCorresp = [6 Inf];
f=fopen(fnMatch);
corresp = fscanf(f,formatSpec,sizeCorresp);
command = ['rm ',fnMatch]; s = system(command);

thresh=param.threshMatch;
Iseg=segment(I1Match,'variance',thresh);

matches = zeros(5,1);
k=0;
for i=1:size(corresp,2)
    if corresp(1,i)>0 && corresp(1,i)<=sz0(2) && corresp(2,i)>0 && corresp(2,i)<=sz0(1)
        if Iseg(corresp(2,i),corresp(1,i)) == 1
            k = k+1;
            matches(1,k) = corresp(1,i);
            matches(2,k) = corresp(2,i);
            matches(4,k) = corresp(4,i)-corresp(2,i);
            matches(3,k) = corresp(3,i)-corresp(1,i);
        end
    end
end

scores = corresp(5,:);
idx = find(corresp(5,:));
scores = scores(scores~=0);
scores = (scores-min(scores))/(max(scores)-min(scores));
for j=1:length(idx)
    matches(5,idx(j)) = scores(j);
end

%% Optical flow
print = 0;  % if print =1:verbose mode
disp = 0;   % if disp=1; displays the results at each iteration
w = of_l1_l2_fm_admm(I1,I2,sz0,matches,param,print,disp);

w = crop_fit_size_center(w,[sz0(1),sz0(2),2]);





