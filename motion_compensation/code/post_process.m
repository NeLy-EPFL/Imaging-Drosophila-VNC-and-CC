function [wOut,occ] = post_process(w,I1,I2,sigmaS,sigmaC)

occ = detect_occlusion(w, cat(3,I1,I2));
wOut = denoise_color_weighted_medfilt2(w,cat(3,I1,I2), medfilt2(occ,[9,9]),ceil(sigmaS),[3 3], ceil(sigmaC), ceil(sigmaS), false);
occ = detect_occlusion(wOut, cat(3,I1,I2));
occ = max(occ,0.01);

vectX = 1:size(I1,1);
vectY = 1:size(I1,2);
[gridY,gridX] = meshgrid(vectY,vectX);
correspX = gridX + w(:,:,2);
correspY = gridY + w(:,:,1);
idxOut1 = correspX > size(I1,1);
idxOut2 = correspX < 1;
idyOut1 = correspY > size(I1,2);
idyOut2 = correspY < 1;
idOut = idxOut1+idxOut2+idyOut1+idyOut2;
id=find(idOut);
occ(id) = 0;

