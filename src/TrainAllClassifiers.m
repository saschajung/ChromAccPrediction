function [ L0Classifiers, L1Classifiers, L2Classifiers ] = TrainAllClassifiers( allSamples )
%TRAINL0L1L2CLASSIFIERS
% Trains all three layers of classifiers.
% To make predictions use
% allSamples has to be a cell array of matrices of the form [Predictor1 Predictor2 ... PredictorN Observation]

display('Train L0 Classifiers')
L0Classifiers = TrainL0Classifiers(allSamples);
display('Train L0 Classifiers (done)')

L1Classifiers = cell(numel(allSamples(1,:)),1);
parfor tree=1:numel(L0Classifiers)
    display(strcat('Train L1 classifier for sample ', num2str(tree)));
    L0Prediction = [];
    for subtree=1:numel(L0Classifiers{tree})
        [~,s] = predict(L0Classifiers{tree}{subtree},allSamples{tree}(:,1:end-1));
        L0Prediction = [L0Prediction s(:,2)];
    end
    L0Prediction = [L0Prediction allSamples{tree}(:,end)];
    [c,~] = TrainSingleL1Classifier(L0Prediction,false);
    L1Classifiers{tree} = c;
end
display('Train L1 Classifiers (done)')

display('Make L1 Predictions')
L1Predictions = [];
for sample=1:numel(allSamples)
   display(strcat('\tPrediction ',num2str(sample),' made'));
   L1Predictions = [L1Predictions;CombinedLevel1Predictions(L0Classifiers,L1Classifiers,allSamples{sample}(:,1:end-1),allSamples{sample}(:,end))]; 
end
display('Make L1 Predictions (done)')

display('Train L2 Classifiers')
L2Classifiers = cell(1,1);
[c, acc] = TrainSingleL2Classifier(L1Predictions,false);
display(acc)
L2Classifiers{1} = c;
display('Train L2 Classifiers (done)')



end

