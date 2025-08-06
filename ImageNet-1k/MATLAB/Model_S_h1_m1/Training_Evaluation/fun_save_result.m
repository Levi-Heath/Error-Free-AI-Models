function out=fun_save_result(model,patch, module, subset,color,channels_names,training_performance)
%
reportname1 = sprintf('../Evaluation_Data/Model_Performance/Trained_Model_%s_patch_%d_module_%d_subset_%d_ch_%s.mat',...
    model,patch, module, subset, char(channels_names(color)));
% save(reportname1, 'training_performance', '-append');
save(reportname1, 'training_performance');
out=[];
end
