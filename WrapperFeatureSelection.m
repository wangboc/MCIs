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
%  
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


