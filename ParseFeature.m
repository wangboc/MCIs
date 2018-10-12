function [name, area] = ParseFeature(feature_name_index, feature_location)


if feature_name_index     == 0
    name = 'Strength';
elseif feature_name_index == 1
    name = 'clustering_coef';
elseif feature_name_index == 2
    name = 'local_efficiency';  
elseif feature_name_index == 3
    name = 'betweenness'; 
elseif feature_name_index == 4
    name = 'eigenvector';  
elseif feature_name_index == 5
    name = 'subgraph';  
elseif feature_name_index == 6
    name = 'kcoreness';  
elseif feature_name_index == 7
    name = 'flow_coefficiency'; 
elseif feature_name_index == 8
    name = 'pagerank';     
end

areas = {'L-V1';...
    'L-MST';...
    'L-V6';...
    'L-V2';...
    'L-V3';...
    'L-V4';...
    'L-V8';...
    'L-4';...
    'L-3b';...
    'L-FEF';...
    'L-PEF';...
    'L-55b';...
    'L-V3A';...
    'L-RSC';...
    'L-POS2';...
    'L-V7';...
    'L-IPS1';...
    'L-FFC';...
    'L-V3B';...
    'L-L01';...
    'L-L02';...
    'L-PIT';...
    'L-MT';...
    'L-A1';...
    'L-PSL';...
    'L-SFL';...
    'L-PCV';...
    'L-STV';...
    'L-7Pm';...
    'L-7m';...
    'L-POS1';...
    'L-23d';...
    'L-v23ab';...
    'L-d23ab';...
    'L-31pv';...
    'L-5m';...
    'L-5mv';...
    'L-23c';...
    'L-5L';...
    'L-24dd';...
    'L-24dv';...
    'L-7AL';...
    'L-SCEF';...
    'L-6ma';...
    'L-7Am';...
    'L-7Pl';...
    'L-7PC';...
    'L-LIPv';...
    'L-VIP';...
    'L-MIP';...
    'L-1';...
    'L-2';...
    'L-3a';...
    'L-6d';...
    'L-6mp';...
    'L-6v';...
    'L-p24pr';...
    'L-33pr';...
    'L-a24pr';...
    'L-p32pr';...
    'L-a24';...
    'L-d32';...
    'L-8BM';...
    'L-p32';...
    'L-10r';...
    'L-47m';...
    'L-8Av';...
    'L-8Ad';...
    'L-9m';...
    'L-8BL';...
    'L-9p';...
    'L-10d';...
    'L-8C';...
    'L-44';...
    'L-45';...
    'L-47l';...
    'L-a47r';...
    'L-6r';...
    'L-IFJa';...
    'L-IFJp';...
    'L-IFSp';...
    'L-IFSa';...
    'L-p946v';...
    'L-46';...
    'L-a946v';...
    'L-9-46d';...
    'L-9a';...
    'L-10v';...
    'L-a10p';...
    'L-10pp';...
    'L-11l';...
    'L-13l';...
    'L-OFC';...
    'L-47s';...
    'L-LIPd';...
    'L-6a';...
    'L-i6-8';...
    'L-s6-8';...
    'L-43';...
    'L-OP4';...
    'L-OP1';...
    'L-OP2-3';...
    'L-52';...
    'L-RI';...
    'L-PFcm';...
    'L-Pol2';...
    'L-TA2';...
    'L-FOP4';...
    'L-MI';...
    'L-Pir';...
    'L-AVI';...
    'L-AAIC';...
    'L-FOP1';...
    'L-FOP3';...
    'L-FOP2';...
    'L-PFt';...
    'L-AIP';...
    'L-EC';...
    'L-PreS';...
    'L-H';...
    'L-ProS';...
    'L-PeEc';...
    'L-STGa';...
    'L-PBelt';...
    'L-A5';...
    'L-PHA1';...
    'L-PHA3';...
    'L-STSda';...
    'L-STSdp';...
    'L-STSvp';...
    'L-TGd';...
    'L-TE1a';...
    'L-TE1p';...
    'L-TE2a';...
    'L-TF';...
    'L-TE2p';...
    'L-PHT';...
    'L-PH';...
    'L-TPOJ1';...
    'L-TPOJ2';...
    'L-TPOJ3';...
    'L-DVT';...
    'L-PGp';...
    'L-IP2';...
    'L-IP1';...
    'L-IP0';...
    'L-PFop';...
    'L-PF';...
    'L-PFm';...
    'L-PGi';...
    'L-PGs';...
    'L-V6A';...
    'L-VMV1';...
    'L-VMV3';...
    'L-PHA2';...
    'L-V4t';...
    'L-FST';...
    'L-V3CD';...
    'L-LO3';...
    'L-VMV2';...
    'L-31pd';...
    'L-31a';...
    'L-VVC';...
    'L-25';...
    'L-s32';...
    'L-pOFC';...
    'L-Pol1';...
    'L-Ig';...
    'L-FOP5';...
    'L-p10p';...
    'L-p47r';...
    'L-TGv';...
    'L-MBelt';...
    'L-LBelt';...
    'L-A4';...
    'L-STSva';...
    'L-TE1m';...
    'L-PI';...
    'L-a32pr';...
    'L-p24';...
    ...
    'R-V1';...
    'R-MST';...
    'R-V6';...
    'R-V2';...
    'R-V3';...
    'R-V4';...
    'R-V8';...
    'R-4';...
    'R-3b';...
    'R-FEF';...
    'R-PEF';...
    'R-55b';...
    'R-V3A';...
    'R-RSC';...
    'R-POS2';...
    'R-V7';...
    'R-IPS1';...
    'R-FFC';...
    'R-V3B';...
    'R-L01';...
    'R-L02';...
    'R-PIT';...
    'R-MT';...
    'R-A1';...
    'R-PSL';...
    'R-SFL';...
    'R-PCV';...
    'R-STV';...
    'R-7Pm';...
    'R-7m';...
    'R-POS1';...
    'R-23d';...
    'R-v23ab';...
    'R-d23ab';...
    'R-31pv';...
    'R-5m';...
    'R-5mv';...
    'R-23c';...
    'R-5L';...
    'R-24dd';...
    'R-24dv';...
    'R-7AL';...
    'R-SCEF';...
    'R-6ma';...
    'R-7Am';...
    'R-7Pl';...
    'R-7PC';...
    'R-LIPv';...
    'R-VIP';...
    'R-MIP';...
    'R-1';...
    'R-2';...
    'R-3a';...
    'R-6d';...
    'R-6mp';...
    'R-6v';...
    'R-p24pr';...
    'R-33pr';...
    'R-a24pr';...
    'R-p32pr';...
    'R-a24';...
    'R-d32';...
    'R-8BM';...
    'R-p32';...
    'R-10r';...
    'R-47m';...
    'R-8Av';...
    'R-8Ad';...
    'R-9m';...
    'R-8BL';...
    'R-9p';...
    'R-10d';...
    'R-8C';...
    'R-44';...
    'R-45';...
    'R-47l';...
    'R-a47r';...
    'R-6r';...
    'R-IFJa';...
    'R-IFJp';...
    'R-IFSp';...
    'R-IFSa';...
    'R-p946v';...
    'R-46';...
    'R-a946v';...
    'R-9-46d';...
    'R-9a';...
    'R-10v';...
    'R-a10p';...
    'R-10pp';...
    'R-11l';...
    'R-13l';...
    'R-OFC';...
    'R-47s';...
    'R-LIPd';...
    'R-6a';...
    'R-i6-8';...
    'R-s6-8';...
    'R-43';...
    'R-OP4';...
    'R-OP1';...
    'R-OP2-3';...
    'R-52';...
    'R-RI';...
    'R-PFcm';...
    'R-Pol2';...
    'R-TA2';...
    'R-FOP4';...
    'R-MI';...
    'R-Pir';...
    'R-AVI';...
    'R-AAIC';...
    'R-FOP1';...
    'R-FOP3';...
    'R-FOP2';...
    'R-PFt';...
    'R-AIP';...
    'R-EC';...
    'R-PreS';...
    'R-H';...
    'R-ProS';...
    'R-PeEc';...
    'R-STGa';...
    'R-PBelt';...
    'R-A5';...
    'R-PHA1';...
    'R-PHA3';...
    'R-STSda';...
    'R-STSdp';...
    'R-STSvp';...
    'R-TGd';...
    'R-TE1a';...
    'R-TE1p';...
    'R-TE2a';...
    'R-TF';...
    'R-TE2p';...
    'R-PHT';...
    'R-PH';...
    'R-TPOJ1';...
    'R-TPOJ2';...
    'R-TPOJ3';...
    'R-DVT';...
    'R-PGp';...
    'R-IP2';...
    'R-IP1';...
    'R-IP0';...
    'R-PFop';...
    'R-PF';...
    'R-PFm';...
    'R-PGi';...
    'R-PGs';...
    'R-V6A';...
    'R-VMV1';...
    'R-VMV3';...
    'R-PHA2';...
    'R-V4t';...
    'R-FST';...
    'R-V3CD';...
    'R-LO3';...
    'R-VMV2';...
    'R-31pd';...
    'R-31a';...
    'R-VVC';...
    'R-25';...
    'R-s32';...
    'R-pOFC';...
    'R-Pol1';...
    'R-Ig';...
    'R-FOP5';...
    'R-p10p';...
    'R-p47r';...
    'R-TGv';...
    'R-MBelt';...
    'R-LBelt';...
    'R-A4';...
    'R-STSva';...
    'R-TE1m';...
    'R-PI';...
    'R-a32pr';...
    'R-p24'};
if feature_location == 0
    feature_location = 360;
end
area = areas(feature_location);
  
    


