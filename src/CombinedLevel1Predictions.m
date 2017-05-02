function [ L1Predictions ] = CombinedLevel1Predictions( Level0Trees, Level1Trees, expression, observations)
%COMBINEDLEVEL0PREDICTIONS Summary of this function goes here
%   Detailed explanation goes here
L1Predictions = [];
%Compute Level0 predictions
for tree=1:numel(Level0Trees)
    L0Prediction = [];
    for subtree=1:numel(Level0Trees{tree})
        [~,s] = predict(Level0Trees{tree}{subtree},expression);
        L0Prediction = [L0Prediction s(:,2)];
    end
    
   [~,s] = Level1Trees{tree}.predictFcn(L0Prediction);
   L1Predictions = [L1Predictions s];
end

L1Predictions = [L1Predictions observations];

end

