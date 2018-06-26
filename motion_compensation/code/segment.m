function Iseg=segment(I,method,param)

sI = size(I);
Iseg = zeros(sI);

if strcmp(method,'variance')
    sP = 5;
    thresh = param;
    for i=1:sI(1)
        for j=1:sI(2)
            p = I(max(i-sP,1):min(i+sP,sI(1)),max(j-sP,1):min(j+sP,sI(2)));
            p = 1/(size(p,1)*size(p,2)) * (p-mean(p(:))).^2;
            var = sum(p(:));
            if var>thresh
                Iseg(i,j)=1;
            end
        end
    end
end