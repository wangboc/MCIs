function [SelectedTrainData, SelectedFeatures_in_RankImportanceOrder] = WrapperFeatureSelection(Matrix, use_predefined_sequence)
%% Implement sequentialfs Matlab function
% 
% Suitable features tested may be used to train model.
% See the following temp results

% HDU, Bocheng Wang 2018.10.12
% HDU, Bocheng Wang 2018.10
%%

X = Matrix(:, 2:size(Matrix, 2));
y = Matrix(:, 1);

SelectedLabel = [];
if use_predefined_sequence == 1
    SelectedLabel = [3 56 116 341 646 980];
else
    c = cvpartition(y,'KFold',10);
    opts = statset('display', 'iter',  'TolTypeFun','abs');
    fun = @(train_data,train_labels,test_data,test_labels) ...
       sum(predict(fitcsvm(train_data,train_labels,'KernelFunction','rbf'), test_data) ~= test_labels); 
    [fs,history] = sequentialfs(fun,X,y,'cv',c,'options',opts);
    SelectedLabel = 1:size(Matrix, 2);
    SelectedLabel = SelectedLabel(fs);
end
SelectedTrainData = [X(:, SelectedLabel), y];
SelectedFeatures_in_RankImportanceOrder = SelectedLabel;
%% temp results  HC_vs_EMCI 
% 
% [1 27 38 58 291 ];                                        Accuracy: 93.8%
% [1 24 27 73 264 633 1341 ];                               Accuracy: 95.8%
% [1 151 252 452 832 1075 1554];                            Accuracy: 93.8%
% [1 73 401 422 654];                                       Accuracy: 91.7%
% [1 27 38 48 291 ];                                        Accuracy: 87.5%
% [1 73 87 401 633 750 ];                                   Accuracy: 87.5%
% [1 25 38 59 422 ];                                        Accuracy: 93.8%
% [1 4 73 87 174 633];                                      Accuracy: 85.4%
%% temp results  HC_vs_MCI
% 
% [2 3 470 575 626 784];                                    Accuracy: 93.8%
% [3 64 96 250 357 598 ];                                   Accuracy: 87.5%
% [29 219 988 1050 ];                                       Accuracy: 89.6%
% [3 232 505 519 748 985 ];                                 Accuracy: 91.7%
% [3 7 297 1412 ];                                          Accuracy: 91.7%
% [3 131 173 186 313 598 627 ];                             Accuracy: 89.6%
% [3 56 116 341 646 980 ];                                  Accuracy: 91.7%
% [2 3 7 13 95 96 222];                                     Accuracy: 87.5%
%% temp results  HC_vs_LMCI
%
% [32 142 270 297 375];                                     Accuracy: 91.7%
% [11 58 140 149 1174];                                     Accuracy: 83.3%
% [4 9 23 33 87 175 176 1054 ];                             Accuracy: 95.8%
% [5 6 27 161 172 282 ];                                    Accuracy: 85.4%
% [40 51 58 67 143 337];                                    Accuracy: 87.5%
% [8 112 172 418 541 1014 ];                                Accuracy: 85.4%
% [5 6 112 167 172 197 239 1504 ];                          Accuracy: 97.9%
% [5 172 191 355 492 1267 ];                                Accuracy: 89.6%
%% temp results  HC_vs_AD
%
% [83 208 223 238 350 772 ];                                Accuracy: 91.7%
% [21 249 308 548 1381];                                    Accuracy: 87.5%
% [83 238 384 558 652];                                     Accuracy: 93.8%
% [21 22 79 185 ];                                          Accuracy: 95.8%
% [21 732 907 ];                                            Accuracy: 91.7%
% [21 79 185 862 ];                                         Accuracy: 91.7%
% [19 216 408 484 ];                                        Accuracy: 85.4%
% [21 79 185 862 ];                                         Accuracy: 89.6%
%% temp results  EMCI_vs_MCI
% 
%
%% temp results  EMCI_vs_LMCI
% 
% 
%% temp results  EMCI_vs_AD
% 
%  
%% temp results  MCI_vs_LMCI
% 
% 
%% temp results  MCI_vs_AD
% 
%  
%% temp results  LMCI_vs_AD
% 
%  


