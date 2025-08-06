function out=fun_save_model_performance(model,imgnt1kdataset,pf,...
        training_performance,image0_perf)
    reportname1 = sprintf('../Evaluation_Data/Model_Performance/training_performance_batch_%d_feature_%d_performance_%s_var.mat', imgnt1kdataset,pf,model);
        str=struct('training_performance',training_performance,'image0_perf',image0_perf);
    save(reportname1,"-fromstruct",str);
    out=[];
end