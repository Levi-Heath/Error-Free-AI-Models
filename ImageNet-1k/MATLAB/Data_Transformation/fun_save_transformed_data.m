function out=fun_save_transformed_data(patch, module, subset,nmb_of_labs_per_module,data,labels,label_ids,label_table,data_param)
%
reportname1 = sprintf('Transformed_IN1k_Data/Transformed_Data_for_SGD/train_data_patch_%d_module_%d_subset_%d_for_%d_labels_per_module.mat', ...
    patch,module,subset,nmb_of_labs_per_module);
%     str=struct('data',data,'labels',labels,'label_ids',label_ids,'label_table',label_table,'data_param',data_param);
%     save(reportname1,"-fromstruct",str);
save(reportname1,'data','labels','label_ids','label_table','data_param');
out=[];
end
