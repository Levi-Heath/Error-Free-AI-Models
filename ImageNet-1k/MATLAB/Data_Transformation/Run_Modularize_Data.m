nmb_of_labels=1000;
nmb_of_labs_per_module=25;
nmb_of_modules=(nmb_of_labels/nmb_of_labs_per_module);
relative_lab_seq=1:nmb_of_labs_per_module;
nmb_of_subsets=2;

patch=0;
parfor module=1:nmb_of_modules
    m_label_ids=[];
    m_labels=[];
    m_data=[];
    m_label_table=[relative_lab_seq;relative_lab_seq+(module-1)*nmb_of_labs_per_module];
    for imgnt1kdataset=1:10
        % change the path to the folder containing the ImageNet-1k mat
        % files
        reportname1 = sprintf('/work/mathbiology/lheath2/data/imagenet1k/mat/train_data_batch_%d.mat', imgnt1kdataset);
        temp_lpad=load(reportname1,'data','labels') %
        data=temp_lpad.data;
        labels=temp_lpad.labels;
        pos_seq=1:length(labels);
        for labs=relative_lab_seq
            idx=(labels==(labs+(module-1)*nmb_of_labs_per_module));
            aa=pos_seq(idx);
            bb=[0*aa+imgnt1kdataset;aa;labels(idx)];
            m_label_ids=[m_label_ids, bb];
            m_labels=[m_labels,0*aa+labs];
            m_data=[m_data;data(idx,:)];
        end
    end
    nmb_dt=length(m_labels);
    set_lng=fix(nmb_dt/nmb_of_subsets);
    for subset=1:nmb_of_subsets
        if subset<nmb_of_subsets
            set=(1:set_lng)+(subset-1)*set_lng;
        else
            set=(1+(subset-1)*set_lng):nmb_dt;
        end
        data=m_data(set,:);
        labels=m_labels(:,set);
        label_ids=m_label_ids(:,set);
        label_table=m_label_table;
        out=fun_save_modularized_data(patch, module, subset,nmb_of_labs_per_module,data,labels,label_ids,label_table)
    end
    module
end
