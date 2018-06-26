function warpedImg = warpImg(img,u,v)

%% Translation- coordinates
ii = zeros(size(img));
idx = find( ~ii );
[x,y] = ind2sub (size(img),idx) ;

xw = x(:)+v(:); yw = y(:)+u(:);

%% Warping
warpedImg = interp2(img, yw, xw, 'linear');
warpedImg = reshape(warpedImg, size(img));

end
