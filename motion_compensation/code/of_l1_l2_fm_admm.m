function w = of_l1_l2_fm_admm(I1,I2,sz0,matches,param,print,disp)
%% Description
% Compute motion with a brigthness constancy data term defined on (I1,I2)
% and a feature matching similarity term defined by the 'matches' variable
% For more details see the paper: "Imaging neural activity in the ventral
% nerve cord of behaving adult Drosophila", bioRxiv
%
%% Input
% I1,I2: Images used for the brightness constancy term
% sz0: original size of the images (before padding)
% matches: list of pixel correspondences used by the feature matching similarity term
% param: parameters of the algorithm (see 'default_parameters.m')
% print: verbose mode 
% disp: display mode
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

sigmaPreproc = 0.9;
preProcFilter = fspecial('gaussian', 2*round(1.5*sigmaPreproc) +1, sigmaPreproc);
I1 = imfilter(I1, preProcFilter, 'corr', 'symmetric', 'same');
I2 = imfilter(I2, preProcFilter, 'corr', 'symmetric', 'same');

deriv_filter = [1 -8 0 8 -1]/12; 

% coarse-to-fine parameters
minSizeC2f = 10;
c2fLevels = ceil(log(minSizeC2f/max(size(I1)))/log(1/param.c2fSpacing));
factor = sqrt(2);
smooth_sigma      = sqrt(param.c2fSpacing)/factor;
c2fFilter = fspecial('gaussian', 2*round(1.5*smooth_sigma) +1, smooth_sigma);
I1C2f = compute_image_pyramid(I1,c2fFilter,c2fLevels,1/param.c2fSpacing);
I2C2f = compute_image_pyramid(I2,c2fFilter,c2fLevels,1/param.c2fSpacing);
matchesC2f = cell(c2fLevels,1);
matchesl = matches;
for i=c2fLevels:-1:1
    lambdaC2f(i) = param.lambda;%/(param.c2fSpacing^((i-1)));
    sigmaSSegC2f(i) = param.sigmaS/(param.c2fSpacing^((i-1)));
    gammaC2f(i) = param.gamma*((c2fLevels-i+1)/c2fLevels)^(1.8);
    
    matchesl(1,:) = max(round(matches(1,:)./(param.c2fSpacing^(i-1))),1); 
    matchesl(2,:) = max(round(matches(2,:)./(param.c2fSpacing^(i-1))),1); 
    matchesl(3,:) = matches(3,:)./(param.c2fSpacing^(i-1)); 
    matchesl(4,:) = matches(4,:)./(param.c2fSpacing^(i-1)); 
    matchesl(5,:) = matches(5,:); 
    matchesC2f{i} = matchesl;    
end

%% Initialization
wl = zeros(size(I1,1), size(I1,2),2);
occ = ones(size(I1));

%% Coarse to fine
for l=c2fLevels:-1:1
if print
    fprintf('\nScale %d\n',l)
end

% Scaled data
I1 = I1C2f{l};
I2 = I2C2f{l};
sigmaS = sigmaSSegC2f(l);
matchesl = matchesC2f{l};
lambda = lambdaC2f(l);
param.gamma = gammaC2f(l);

% Rescale flow
ratio = size(I1,1) / size(wl(:,:,1),1);
ul     = imresize(wl(:,:,1), size(I1), 'bicubic')*ratio;
ratio = size(I1,2) / size(wl(:,:,2),2);
vl     = imresize(wl(:,:,2), size(I1), 'bicubic')*ratio;
wl = cat(3,ul,vl);
sz0 = floor(sz0*ratio);

% Create binary and  motion vectors fields
c = zeros(size(I1,1),size(I1,2));
m = zeros(size(I1,1),size(I1,2),3);
for i=1:size(matches,2) 
    c(matchesl(2,i),matchesl(1,i)) = 1;
    if matchesl(5,i) > m(matchesl(2,i),matchesl(1,i),3)
        m(matchesl(2,i),matchesl(1,i),1) = matchesl(3,i);
        m(matchesl(2,i),matchesl(1,i),2) = matchesl(4,i);
        m(matchesl(2,i),matchesl(1,i),3) = matchesl(5,i);
    end
end        
% no weights
m(:,:,3) = ones(size(I1,1),size(I1,2));

occ = imresize(occ, size(I1), 'nearest');
occ = logical(occ);

mu = param.mu;
nu = param.nu;

G=LinOpGrad(size(I1),[],'circular');      
fGtG=fftn(G.fHtH);
eigsDtD = lambda * fGtG.^2 + mu;

for iWarp=1:param.nbWarps
    if print
        fprintf('Warp %d\n',iWarp)
    end
    
    wPrev = wl;
    dul = zeros(size(wl,1), size(wl,2));
    dvl = zeros(size(wl,1), size(wl,2));
    uu1 = zeros(size(wl,1), size(wl,2));
    uu2 = zeros(size(wl,1), size(wl,2));
    dwl = zeros(size(wl));
    % ADMM variables
    alpha  = zeros(size(I1,1),size(I1,2),2);
    beta  = zeros(size(I1,1),size(I1,2),2);
    z  = zeros(size(I1,1),size(I1,2),2);
    u  = zeros(size(I1,1),size(I1,2),2);
    % No occlusion handling
    occ = ones(size(I1));

    % Pre-computations
    [It,Ix,Iy] = partial_deriv(cat(3,I1,I2), wl, 'bi-cubic', deriv_filter);
    Igrad = Ix.^2 + Iy.^2 + 1e-3;
    idocc = occ == 0;

    %% Main iterations loop
    for it=1:param.maxIters
    %    rnorm_prev  = rnorm;
        thresh = Igrad/(mu+nu);
        thresh2 = param.gamma.*m(:,:,3)./nu;

        %% Data update 
        r1 = z-wl-alpha/mu;
        r2 = u-wl+beta/nu;
        t = (mu*r1+nu*r2)/(mu+nu);
        t1 = t(:,:,1);
        t2 = t(:,:,2);

        rho = It + t1.*Ix + t2.*Iy;
        idx1 = rho      < - thresh;
        idx2 = rho      >   thresh;
        idx3 = abs(rho) <=  thresh;

        dul(idx1) = t1(idx1) + Ix(idx1)/(mu+nu);
        dvl(idx1) = t2(idx1) + Iy(idx1)/(mu+nu);

        dul(idx2) = t1(idx2) - Ix(idx2)/(mu+nu);
        dvl(idx2) = t2(idx2) - Iy(idx2)/(mu+nu);

        dul(idx3) = t1(idx3) - rho(idx3).*Ix(idx3)./Igrad(idx3);
        dvl(idx3) = t2(idx3) - rho(idx3).*Iy(idx3)./Igrad(idx3);

        dul(idocc) = t1(idocc);
        dvl(idocc) = t2(idocc);

        dwl = cat(3,dul,dvl);
        
        %% Regularization update
        z(:,:,1) = real(ifft2(fft2(mu*(dwl(:,:,1)+wl(:,:,1))+alpha(:,:,1))./eigsDtD));
        z(:,:,2) = real(ifft2(fft2(mu*(dwl(:,:,2)+wl(:,:,2))+alpha(:,:,2))./eigsDtD));

        %% Matching update
        u0 = wl+dwl - beta/nu;
        u01 = u0(:,:,1);
        u02 = u0(:,:,2);
        m1 = m(:,:,1);
        m2 = m(:,:,2);
        idx = (c == 0);

        rho = u0-m(:,:,1:2);
        idx1 = logical((rho(:,:,1)      < - thresh2) .* (c == 1));
        idx2 = logical((rho(:,:,1)      >   thresh2) .* (c == 1));
        idx3 = logical((abs(rho(:,:,1)) <=  thresh2) .* (c == 1));

        uu1(idx1) = u01(idx1) + thresh2(idx1);
        uu1(idx2) = u01(idx2) - thresh2(idx2);
        uu1(idx3) = m1(idx3);
        uu1(idx) = u01(idx);
%       uu1(idocc) = z01(idocc);

        idx1 = logical((rho(:,:,2)      < - thresh2) .* (c == 1));
        idx2 = logical((rho(:,:,2)      >   thresh2) .* (c == 1));
        idx3 = logical((abs(rho(:,:,2)) <=  thresh2) .* (c == 1));

        uu2(idx1) = u02(idx1) + thresh2(idx1);
        uu2(idx2) = u02(idx2) - thresh2(idx2);
        uu2(idx3) = m2(idx3);
        uu2(idx) = u02(idx);
%       uu2(idocc) = z02(idocc);

        u = cat(3,uu1,uu2);
        
        %% Lagrange parameters update
        alpha = alpha + mu*(wl+dwl - z);
        beta= beta + nu*(u-wl-dwl);

        %% Post-processing
        if mod(it,param.iWM) == 0
            w0 = wl+dwl;
            [w0,occTmp] = post_process(w0,I1,I2,sigmaS,param.sigmaC);
            dwl = w0-wl;
        end

        %% End of iterations checking
        w = wl+dwl;
        change = norm(w(:)-wPrev(:))/norm(wPrev(:));
        if change<param.changeTol
            break;
        end
        wPrev = w;
    end
    wl = wl + dwl;
    [wl,occ] = post_process(wl,I1,I2,sigmaS,param.sigmaC);
    occ = im2bw(occ,param.occThresh);
    if disp
        figure(11);imagesc(crop_fit_size_center(flowToColor(wl),[sz0(1),sz0(2),3]));colormap gray; axis image; drawnow;
    end
end
end
w = wl;

end
