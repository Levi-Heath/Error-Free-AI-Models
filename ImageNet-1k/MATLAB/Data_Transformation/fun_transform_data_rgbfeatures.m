function output=fun_transform_data_rgbfeatures(img_data,param)
%
%img_data=data_load.data;
output.transformed_image=[];
output.mnsv=[];
output.maxsv=[];
output.ipvsz=[];

image_size=param.image_size;
img_x_dim=image_size(1);
img_y_dim=image_size(1);

compute_decimal_place=param.compute_decimal_place;
feature_RGB=param.feature_RGB;
[nmb_of_colors,~]=size(feature_RGB);

[data_size,~]=size(img_data);
reshaped_images=zeros(data_size,img_x_dim,img_y_dim,3);
for m=1:data_size
    aa=double(img_data(m,:));
    reshaped_images(m,:,:,:)=reshape(aa,img_x_dim,img_y_dim,3);
end

if param.dwnsz_on==1
    x_trim=param.x_trim;
    y_trim=param.y_trim;
    downsizing=param.downsizing;
    x_dwnsz_dim=fix(img_x_dim/downsizing);
    y_dwnsz_dim=fix(img_y_dim/downsizing);

    ipvsz=(x_dwnsz_dim-2*x_trim)*(y_dwnsz_dim-2*y_trim);
    xsq=1:downsizing;
    ysq=1:downsizing;
    output.transformed_image=zeros(ipvsz,data_size,nmb_of_colors);
    output.mnsv=zeros(data_size,nmb_of_colors);
    output.maxsv=zeros(data_size,nmb_of_colors);
    output.ipvsz=ipvsz;

    temp_img=zeros(ipvsz,3);
    for m=1:data_size
        img=squeeze(reshaped_images(m,:,:,:));
        for color=1:3
            img1=squeeze(img(:,:,color));
            aa=zeros(ipvsz,1);
            for ii=(1+x_trim):(x_dwnsz_dim-x_trim)
                for jj=(1+y_trim):(y_dwnsz_dim-y_trim)
                    aa((jj-y_trim)+((ii-x_trim)-1)*(y_dwnsz_dim-2*y_trim),1)=mean(img1(xsq+2*(ii-1),ysq+2*(jj-1)),'all');
                end
            end
            temp_img(:,color)=aa;
        end
        for color=1:nmb_of_colors
            c1=feature_RGB(color,1);
            c2=feature_RGB(color,2);
            c3=feature_RGB(color,3);
            aa=c1*temp_img(:,1)+c2*temp_img(:,2)+c3*temp_img(:,3);
            mnsv=mean(aa);
            aa=aa-mnsv;
            maxsv=max(abs(aa));
            output.transformed_image(:,m,color)=round(aa/maxsv*10^compute_decimal_place)*10^(-compute_decimal_place);
            output.mnsv(m,color)=mnsv;
            output.maxsv(m,color)=maxsv;
        end
    end
else    
    temp_img=zeros(img_x_dim*img_y_dim,3);
    ipvsz=img_x_dim*img_y_dim;
    output.ipvsz=ipvsz;
    for m=1:data_size
        img=squeeze(reshaped_images(m,:,:,:));
        for color=1:3
            aa=img(:,:,color);
            aa=aa(:);
            temp_img(:,color)=aa;
        end
        for color=1:nmb_of_colors
            c1=feature_RGB(color,1);
            c2=feature_RGB(color,2);
            c3=feature_RGB(color,3);
            aa=c1*temp_img(:,1)+c2*temp_img(:,2)+c3*temp_img(:,3);
            mnsv=mean(aa);
            aa=aa-mnsv;
            maxsv=max(abs(aa));
            output.transformed_image(:,m,color)=round(aa/maxsv*10^compute_decimal_place)*10^(-compute_decimal_place);
            output.mnsv(m,color)=mnsv;
            output.maxsv(m,color)=maxsv;
        end             
    end    
end
if data_size==1
    output.transformed_image=squeeze(output.transformed_image);
end
end