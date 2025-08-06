% clear all
%
nmb_of_modules=40;
nmb_of_module_subsets=2;
channels_names={'R','G','B','RGg1','RBg1','GBg1','RGg2','RBg2','GBg2','RB','RG','GB','eRGB','BW','X','Y','Z'};
feature_RGB=[1 0 0
    0      1      0
    0      0      1
    0.618  0.382  0
    0.618  0      0.382
    0      0.618  0.382
    0.382  0.618  0
    0.382  0      0.618
    0      0.382  0.618
    0.5    0.5    0
    0.5    0      0.5
    0      0.5    0.5
    1/3    1/3    1/3    
    0.299  0.587  0.114
    0.4125 0.3576  0.1804
    0.2126 0.7152  0.0722
    0.0193 0.1192  0.9502]; 
nmb_of_colors=length(channels_names);
patch=0;
nmb_of_labs_per_module=25;
cross_entropy=1;

param.image_size=[64,64];
param.downsizing=2;
param.x_trim=1;
param.y_trim=1;
param.compute_decimal_place=4;
param.dwnsz_on=1;

param.patch=patch;
param.nmb_of_modules=nmb_of_modules;
param.nmb_of_module_subsets=nmb_of_module_subsets;

param.cross_entropy=cross_entropy;
param.nmb_of_labs_per_module=nmb_of_labs_per_module;

param.channels_names=channels_names;
param.nmb_of_colors=nmb_of_colors;
param.feature_RGB=feature_RGB;

parfor module=1:nmb_of_modules
    data_param=[];
    for subset=1:nmb_of_module_subsets
        reportname1 = sprintf('Transformed_IN1k_Data/Modularized_Data_for_SGD/modularized_data_patch_%d_module_%d_subset_%d_for_%d_labels_per_module.mat', ...
            patch,module,subset,nmb_of_labs_per_module);
        %     save(reportname1,'data','labels','label_ids','label_table','data_param');
        data_load=load(reportname1);
        data_0=data_load.data;
        output=fun_transform_data_rgbfeatures(data_0,param);
        data=output.transformed_image;
        data_param.mnsv=output.mnsv;
        data_param.maxsv=output.maxsv;
        data_param.ipvsz=output.ipvsz;
        labels=data_load.labels;
        label_ids=data_load.label_ids;
        label_table=data_load.label_table;
        out=fun_save_transformed_data(patch, module, subset,nmb_of_labs_per_module,data,labels,label_ids,label_table,data_param);
    end
    module
end
%%