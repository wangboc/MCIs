function [strength, ...
          clustering_coef, ...
          local_efficiency, ...
          betweenness, ...
          eigenvector, ...
          subgraph, ...
          kcoreness, ...
          flow_efficiency, ...
          pagerank...
] = BCT_calculation(data)
%% 利用BCT工具箱，计算网络拓扑值
%   
%   Data 是相关矩阵

%% Search for maximum GCE with the optimal threshold value
GCE_max = 0;
PSW_optimal = 0;
GCEs = zeros(40, 1);
for index = 1:100
    PSW = index/100;
    w = threshold_proportional(data, PSW);
    w = weight_conversion(w, 'binarize');
    GCE = efficiency_bin(w) - PSW;
    GCEs(index) = GCE;
    if GCE > GCE_max
        GCE_max = GCE;
        PSW_optimal = PSW;
    end
end
%% 阈值处理
data_threshed = threshold_proportional(data, PSW_optimal);
%% 二值化
% binarized_network = weight_conversion(data_threshed, 'binarize');
%% 计算特性
[   strength, ...
    clustering_coef, ...
    local_efficiency, ...
    betweenness, ...
    eigenvector, ...
    subgraph, ...
    kcoreness, ...
    flow_efficiency, ...
    pagerank...
] = Features_calculation(data_threshed);


