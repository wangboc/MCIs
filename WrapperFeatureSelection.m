function [SelectedTrainData] = WrapperFeatureSelection(Matrix)
%% Implement sequentialfs Matlab function
% HDU, Bocheng Wang 2018.10
%%
X = Matrix(:, 2:size(Matrix, 2));
y = Matrix(:, 1);

c = cvpartition(y,'KFold',10);
opts = statset('display', 'iter',  'TolTypeFun','abs');
fun = @(train_data,train_labels,test_data,test_labels) ...
       sum(predict(fitcsvm(train_data,train_labels,'KernelFunction','rbf'), test_data) ~= test_labels); 

[fs,history] = sequentialfs(fun,X,y,'cv',c,'options',opts);
SelectedLabel = 1:size(Matrix, 2);
SelectedLabel = SelectedLabel(fs);
SelectedTrainData = [X(:, SelectedLabel), y];
%% temp results 
% HC_vs_EMCI 
% mapstd;
% [7 70 151 311 1529 ]; Accuracy: 83.3%
% [7 94 141 147]; Accuracy: 91.7%
% [7 112 216 444 541]; Accuracy: 91.7%

% mapminmax;
% [1 27 38 58 291 ]; Accuracy:93.8%
% [1 24 27 73 264 633 1341 ]; Accuracy:95.8%
% [1 151 252 452 832 1075 1554]; Accuracy: 93.8%
% [1 73 401 422 654]; Accuracy: 91.7%
% [1 27 38 48 291 ]; Accuracy: 87.5%
% [1 73 87 401 633 750 ]; Accuracy: 87.5%
% [1 25 38 59 422 ]; Accuracy: 93.8%
% [1 4 73 87 174 633]; Accuracy: 85.4%
%% temp results 
% HC_vs_MCI
%  
%% temp results 
% HC_vs_LMCI
%% temp results 
% HC_vs_AD
%% temp results 
% EMCI_vs_MCI
%% temp results 
% EMCI_vs_LMCI
% 
%% temp results 
% EMCI_vs_AD
%  
%% temp results 
% MCI_vs_LMCI
% 
%% temp results 
% MCI_vs_AD
%  
%% temp results 
% LMCI_vs_AD
%  


