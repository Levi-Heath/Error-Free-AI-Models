function image_batch=fun_proc_batch(nmb_of_images,nmb_of_batches,batch_nmb,image0)
%
%
image_batch=[];
nmb_of_batch_inc=floor(nmb_of_images/nmb_of_batches);
bstrt=1+(batch_nmb-1)*nmb_of_batch_inc;
bend=batch_nmb*nmb_of_batch_inc;
if batch_nmb<nmb_of_batches && image0<nmb_of_images
    if bstrt>=image0 && image0<bend
        image_batch=image0:bend;
    end
end
if batch_nmb==nmb_of_batches && image0<nmb_of_images
    if bstrt>=image0 
        image_batch=image0:nmb_of_images;
    end
end
end