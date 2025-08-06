function [predicted_label,likelyhood]=fun_majority_rule_prediction(training_performance_temp,param)
% training_performance_temp=mpout;
% batch_training_performance([lab;dis],batch_of_data,module,subset,color)
% batch_training_performance(:,:,5,2,3)=[[1 2 0];[3 4 -1]];
var_on=1;%1; %0 for minmax, 1 for variance, which is the best; range-2nd best
% channels_names={'R','G','B','BW','RGB'};
% channel_sel=[1,2,3,4,5];
% channel_sel=[1,2,3];
% batch_training_performance=permute(training_performance,[1,5,2,3,4]);
channel_sel=param.channel_sel;

sz=size(training_performance_temp);
if length(sz)==5
    batch_training_performance=permute(training_performance_temp,[1,5,2,3,4]);
    aa=squeeze(batch_training_performance(:,:,:,1,channel_sel)); % to ([lab;dis],batch_of_data,module,color)
    bb=squeeze(batch_training_performance(:,:,:,2,channel_sel));
    cc=cat(3,aa,bb); % cat in module dimension to ([lab;dis],batch_of_data,module,color)
    trnpf=permute(cc,[3,4,2,1]); % to (module,color,batch_of_data,[lab,dis])
    %     AB=squeeze(range(trnpf(:,:,:,1),2)); % to (module,batch_of_data) in labels
    AB=trnpf(:,:,:,1); % to (module,color,batch_of_data) in labels
    dtsz=sz(5);
    mdzs=2*sz(2);
else
    batch_training_performance=training_performance_temp;
    aa=squeeze(batch_training_performance(:,:,1,:)); % to ([lab;dis],module,color)
    bb=squeeze(batch_training_performance(:,:,2,:));
    cc=cat(2,aa,bb); % cat in module dimension to ([lab;dis],module,color)
    trnpf=permute(cc,[2,3,1]); % to (module,color,[lab,dis])
    %     AB=squeeze(range(trnpf(:,:,1),2)); % to (module) in labels
    AB=trnpf(:,:,1); % to (module,color) in labels
    dtsz=1;
    mdzs=2*sz(2);
end
predicted_label=zeros(1,dtsz);
likelyhood=zeros(1,dtsz);
seq=1:mdzs;

channel_rel_seq=1:length(channel_sel);
if dtsz>1
    for m=1:dtsz
        [~,bb]=mode(squeeze(AB(:,channel_rel_seq,m)),2); %majority rule start
        aa=max(bb);
        idx=(bb==aa);
        seq_0=seq(idx); % majority rule end
        if var_on==1
            seq_1=var(trnpf(seq_0,channel_rel_seq,m,2),0,2);
        else
            % seq_1=max(trnpf(seq_0,channel_sel,m,2),[],2).*var(trnpf(seq_0,channel_sel,m,2),0,2); %var(trnpf(seq_0,channel_sel,m,2),0,2).*sum(trnpf(seq_0,channel_sel,m,2).^2,2);
            % seq_1=range(trnpf(seq_0,channel_sel,m,2),2);
            seq_1=range(trnpf(seq_0,channel_rel_seq,m,2),2).*var(trnpf(seq_0,channel_rel_seq,m,2),0,2);
        end
        [~,idx]=min(seq_1);
        ps=seq_0(idx);
        predicted_label(m)=trnpf(ps,1,m,1);
        likelyhood(m)=aa;
    end
else
    [~,bb]=mode(squeeze(AB(:,channel_rel_seq)),2);
    aa=max(bb);
    idx=(bb==aa);
    seq_0=seq(idx);
    if var_on==1
        seq_1=var(trnpf(seq_0,channel_rel_seq,2),0,2);
    else
        % seq_1=max(trnpf(seq_0,channel_sel,m,2),[],2).*var(trnpf(seq_0,channel_sel,m,2),0,2); %var(trnpf(seq_0,channel_sel,m,2),0,2).*sum(trnpf(seq_0,channel_sel,m,2).^2,2);
        % seq_1=range(trnpf(seq_0,channel_sel,m,2),2);
        seq_1=range(trnpf(seq_0,channel_rel_seq,2),2).*var(trnpf(seq_0,channel_rel_seq,2),0,2);
    end
    [~,idx]=min(seq_1);
    ps=seq_0(idx);
    predicted_label=trnpf(ps,1,1);
    likelyhood=aa;
end
end