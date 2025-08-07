function images_bw=fun_primecolor_2_features(images,data_param,color,feature_RGB)
%
%
sz=size(images);
vone=ones(sz(1),1);
ori_images=0*images;
% feature_RGB=[1 0 0
%     0      1      0
%     0      0      1
%     0.618  0.382  0
%     0.618  0      0.382
%     0      0.618  0.382
%     0.382  0.618  0
%     0.382  0      0.618
%     0      0.382  0.618
%     0.5    0.5    0
%     0.5    0      0.5
%     0      0.5    0.5
%     1/3    1/3    1/3
%     0.2126 0.7152 0.0722
%     0.299  0.587  0.114];
c1=feature_RGB(color,1);
c2=feature_RGB(color,2);
c3=feature_RGB(color,3);

for cl=1:3
    aa=squeeze(images(:,:,cl));
    bb=squeeze(data_param.maxsv(:,1,cl));
    cc=vone*(squeeze(data_param.mnsv(:,1,cl))');
    ori_images(:,:,cl)=aa*diag(bb)+cc;
end
if color>=4
    aa=c1*ori_images(:,:,1)+...
        c2*ori_images(:,:,2)+c3*ori_images(:,:,3);
    bb=mean(aa,1);
    aa=aa-vone*bb;
    cc=1./max(abs(aa),[],1);
    images_bw=aa*diag(cc);
else
    images_bw=squeeze(images(:,:,color));
end
end