% clear all

model={'T_h1_m1';'S_h1_m1';'T_h1_m2';'S_h1_m2';'T_h2_m1';'S_h2_m1';};
nmb_of_models=length(model);
rtmax=[];
rtmin=[];
rtmed=[];
rtmean=[];
nmb_of_100rt_module=[];
nmb_of_100rt=[];
stat_100rt=[];
for ii=1:nmb_of_models
    md=char(model(ii));
    reportname1 = sprintf('Model_%s/Training_Evaluation/%s_1_performance.mat',md,md);
    load(reportname1)

    rtmax=[rtmax;max(pstvrt_model,[],'all')];
    rtmin=[rtmin;min(pstvrt_model,[],'all')];
    rtmed=[rtmed;median(pstvrt_model(:),'all')];
    rtmean=[rtmean;mean(pstvrt_model(:),'all')];

    aa=squeeze(pstvrt_model(2,:,1));
    bb=squeeze(pstvrt_model(2,:,2));
    cc=[aa,bb];
    idx=(cc==100);
    nmb_of_100rt_module=[nmb_of_100rt_module;sum(1*idx)];
    %     rt100=(sum(idx*1)-1)/length(cc);
    dd=[squeeze(pstvrt_model(:,:,1));squeeze(pstvrt_model(:,:,2))];
    idx2=(dd==100);
    nmb_of_100rt=[nmb_of_100rt;sum(1*idx)];
    stat_100rt=[stat_100rt;[sum(1*idx2,'all'),sum(1*idx)]];
end
%%

perfmn=table(model,rtmin,rtmean,rtmed,rtmax,stat_100rt)
