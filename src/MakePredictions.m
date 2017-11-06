function [ accMat_pred ] = MakePredictions( expMat, accMat, L0, L1, L2, expMat_new )
%MAKEPREDICTIONS 
%   Makes predictions given the training data 'expMat' and 'accMat' (for
%   re-computing the mahalanobis distance to the training dataset), the L0,
%   L1 and L2 classifiers as well as new data for prediction ('expMat_new').


%   Step 2: Create empirical reference distributions for gene expression
%   values associated to accessible and inaccessible regions
oneDist = expMat(accMat==1);
%Just for generalizability, not necessary for the example data
oneDist(isnan(oneDist)) = [];

zeroDist = expMat(accMat==0);
%Just for generalizability, not necessary for the example data
zeroDist(isnan(zeroDist)) = [];

[r,c] = size(expMat_new);

accMat_pred = NaN*zeros(r,c);

for i = 1:c
    idx = find(~isnan(expMat_new(:,i)));
    [l,~] = MakeCombinedLevel2Predictions(L0,L1,L2, [expMat_new(idx,i) mahal(expMat_new(idx,i),zeroDist) mahal(expMat_new(idx,i),oneDist)]);
    accMat_pred(idx,i) = l;
end


end

