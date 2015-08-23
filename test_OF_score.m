clear
clc

load voc-dpm/vid1_detections_thm1.mat

projected_centers = {};
for k=1:3:90
    fname = sprintf('me_CLG-TV_%08d.mat',k );
    load (['optical_flow/results/me_CLG-TV/', fname]);
    for d=1:size(boxes{k},1)
        x1 = boxes{k}(d,1);
        x2 = boxes{k}(d,2); 
        y1 = boxes{k}(d,3);
        y2 = boxes{k}(d,4);
        
        u = uv(:,:,1).';
        v = uv(:,:,2).';
        u_box = u(x1:x2, y1:y2);
        u_avg_box = mean(u(:));
        
        v_box = v(x1:x2, y1:y2);
        v_avg_box = mean(v(:));
        [u_avg_box, v_avg_box];
        
        center.x = (x1+x2)/2;
        center.y = (y1+y2)/2;
        
        projected_centers{k,d}.x = center.x + u_avg_box;
        projected_centers{k,d}.y = center.y + v_avg_box;
        
    end
end


save voc-dpm/vid1_detections_thm1.mat
