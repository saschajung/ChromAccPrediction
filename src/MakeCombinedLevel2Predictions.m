function [ l,s ] = MakeCombinedLevel2Predictions( Trees_L0, Trees_L1, Tree_L2, predictors )
%MAKECOMBINEDLEVEL2PREDICTIONS Summary of this function goes here
%   Detailed explanation goes here

L1Predictions = [];
%Compute Level0 predictions
for tree=1:numel(Trees_L0)
    L0Prediction = [];
    for subtree=1:numel(Trees_L0{tree})
        [~,s] = predict(Trees_L0{tree}{subtree},predictors);
        L0Prediction = [L0Prediction s(:,2)];
    end
    
   [~,s] = Trees_L1{tree}.predictFcn(L0Prediction);
   L1Predictions = [L1Predictions s];
end

[l,s] = Tree_L2{1}.predictFcn(L1Predictions);

end

