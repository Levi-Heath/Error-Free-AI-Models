function []=fun_spoke_plot(vect_image,true_label,Ws,bs,spr,Wt,bt,tpr,nmb_of_labels)
%
%
lb=true_label;
nml=nmb_of_labels;
nn=3;
clr=colormap(cool(nn*(nml+1)));
lbnm={'a','b','c','d','e','f','g','h','i','j','k','l','m','n',...
    'o','p','q','r','s','t','u','v','w','x','y','z'};
gy=.33;
grdcl='w';
grlwt=2;
msz=4;
redge=1.1;

figure(1)
hold off

subplot(1,2,1)
prediction=fun_prediction(vect_image,Ws,bs);
predicted=prediction.A_end;
[~,nmd]=size(predicted);
offst=.07;
%%%%%%%%%%%%%%%
inc=2*pi/nml;
tht=(0:(nml-1))*inc;
A=zeros(2,nmb_of_labels);
for ii=1:nml %tht
    A(:,ii)=[tht(ii);1];
end
radii=1./abs(prediction.predicted.distance+offst);
%%%%%%%%%%%%%%%
pi_pt=zeros(2,nmd);
for mm=1:nmd
    pi_pt(:,mm)=A*predicted(:,mm);
end

aa=pi_pt(1,:);
theta_all=aa(:);
% aa=pi_pt(2,:);
% rho_all=aa(:);
%%%%%%%%%%%%%%%
% polarplot(twopi,unitclc,'linewidth',1,'Color','w');
% hold on
for kk=1:nml
    idx=(lb==kk);
    theta=theta_all(idx);
    rho=radii(idx);
    % rho=rho_all(idx);
    p=polarplot(theta,rho/max(rho));
    hold on

    p.Marker = 'square';
    p.MarkerSize = msz;
    p.LineStyle = "none";
    p.Color = clr(nn*kk,:);
    p.MarkerFaceColor = clr(nn*kk,:);
end
ax = gca;
ax.RTickLabel = {};
ax.ThetaTick = rad2deg(tht);
ax.ThetaTickLabel = lbnm;
axis([-inf, inf, 0,redge])
set(ax,'Color',[gy gy gy])
set(ax,'GridColor',grdcl,'LineWidth',grlwt)

% title("SGD - trained Model (" + SGDpr*epoch_size_SGD/epoch_sz_GDT + "%)",'fontsize',14)
title("SGD - trained Model (" + round(spr,2) + "%)",'fontsize',14)

%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(1,2,2)
prediction=fun_prediction(vect_image,Wt,bt);
predicted=prediction.A_end;
[~,nmd]=size(predicted);
% offst=.05;
%%%%%%%%%%%%%%%
inc=2*pi/nml;
tht=(0:(nml-1))*inc;
A=zeros(2,nmb_of_labels);
for ii=1:nml %tht
    A(:,ii)=[tht(ii);1];
end
radii=1./abs(prediction.predicted.distance+offst);
%%%%%%%%%%%%%%%
pi_pt=zeros(2,nmd);
for mm=1:nmd
    pi_pt(:,mm)=A*predicted(:,mm);
end

aa=pi_pt(1,:);
theta_all=aa(:);
% aa=pi_pt(2,:);
% rho_all=aa(:);
%%%%%%%%%%%%%%%
% polarplot(twopi,unitclc,'linewidth',1,'Color','w');
% hold on
for kk=1:nml
    idx=(lb==kk);
    theta=theta_all(idx);
    rho=radii(idx);
    % rho=rho_all(idx);
    p=polarplot(theta,rho/max(rho));
    hold on

    p.Marker = 'square';
    p.MarkerSize = msz;
    p.LineStyle = "none";
    p.Color = clr(nn*kk,:);
    p.MarkerFaceColor = clr(nn*kk,:);
end
ax = gca;
ax.RTickLabel = {};
ax.ThetaTick = rad2deg(tht);
ax.ThetaTickLabel = lbnm;
axis([-inf, inf, 0,redge])
set(ax,'Color',[gy gy gy])
set(ax,'GridColor',grdcl,'LineWidth',grlwt)

% title('Fully Trained Model (100%)','fontsize',14)
title("GDT - trained Model (" + tpr + "%)",'fontsize',14)

set(gcf,'Position',[10 80 900 440])
% set(gcf, 'MenuBar', 'None')
% sgtitle('Spoke-Plot for Training', 'fontsize', 16)
sgtitle('Confusion Wheel for Training', 'fontsize', 16)

    function out=fun_prediction(vect_image,W, b)
        %
        %
        nmb_of_hidden_layers=length(fieldnames(W))-1;
        W1=W.LayerName1;
        W2=W.LayerName2;
        b1=b.LayerName1;
        b2=b.LayerName2;

        a_0=vect_image;
        %             true_label=data_load.labels;
        %             dtsz=length(true_label);
        nmb_labels=length(b2);
        z1=W1*a_0+b1;
        [a1,~]=fun_activation(z1);
        z2=W2*a1+b2;

        if nmb_of_hidden_layers==1
            [a2,~]=fun_softmax(z2);
            predicted_vector=a2;
        else
            W3=W.LayerName3;
            b3=b.LayerName3;
            nmb_labels=length(b3);
            [a2,~]=fun_activation(z2);

            z3=W3*a2+b3;
            [a3,~]=fun_softmax(z3);
            predicted_vector=a3;
        end
        out.A_end=predicted_vector;
        out.predicted=fun_predicted_vector_2_label(predicted_vector,nmb_labels);
    end
end