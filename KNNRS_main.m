% B. Tu, J. Wang. et al. KNN-based Representation of Superpixels for
% hyperspectral image classification[J]. IEEE Journal of Selected Topics in
% Applied Earth Observations and Remote Sensing. vol. 11, no. 11, pp:4032-4047, Nov. 2018 

close all;
clear;clc;
K_1 = 1; K_2 = 7; C = 16;
load Indian_pines_corrected
im_3d = indian_pines_corrected;
im_2d = ToVector(im_3d)';
load Indian_pines_gt
im_gt = indian_pines_gt;
[xi,yi] = find(im_gt==0);
xisize = size(xi);
[i_row, i_col] = size(im_gt);
im_gt_1d = reshape(im_gt,1,i_row*i_col);
index_map = reshape(1:length(im_gt_1d),[i_row,i_col]);
[r,c,b]=size(im_3d);
x=reshape(im_3d,[r*c b]);
[x] = scale_new(x);
x1=reshape(x,[r c size(im_3d,3)]);

fimage=spatial_feature(x1,204,0.5);           
im_2d = ToVector(fimage)';

index = [];label = [];num_class = [];
for i = 0:1: C
    index_t =  find(im_gt_1d == i);
    index = [index index_t];
    label_t = ones(1,length(index_t))*i;
    label = [label label_t];
    num_class_t = length(index_t);
    num_class = [num_class num_class_t];
end
num_tr = [1022,10,143,83,34,48,23,2,28,2,150,246,60,21,127,35,10];

D = [];D_label = [];tt_data = [];tt_label = [];tt_index = [];
temp_1 = [];D_index = [];temp_train = [];temp_test = [];
for i = 1:1:C
    label_c = find(label == i);
    random_index = label_c(randperm(length(label_c)));
    temp = index(random_index(1:num_tr(i+1)));
    temp_train_1{i} = temp;
    temp_train = [temp_train temp_train_1{i}];
    D_i = im_2d(:,temp_train_1{i});
    D_index = [D_index temp];
    D = [D D_i];
    D_label_i = ones(1,length(temp))*i;
    D_label = [D_label D_label_i];
    temp = index(random_index(num_tr(i+1)+1:end));
    temp_test_1{i} = temp;
    temp_test = [temp_test temp_test_1{i}];
    tt_data_i = im_2d(:,temp_test_1{i});
    temp_1 = [temp_1 temp];
    tt_data = [tt_data tt_data_i];
    tt_label_i = ones(1,length(temp))*i;
    tt_label = [tt_label tt_label_i];
    tt_index = [tt_index temp];
end

data_all = [D,tt_data ];
labels = [D_label,tt_label];
D = D./repmat(sqrt(sum(D.*D)),[size(D,1) 1]); 
label_result = zeros(size(tt_label));

superpixel_data =reshape(compute_mapping(im_2d','PCA',3),r,c,3);
number_superpixels = 3300;lambda_prime = 0.8;sigma = 10; conn8 = 1;
labels_1 = mex_ers(double(superpixel_data),number_superpixels,lambda_prime,sigma,conn8);

Z=data_all;
train_data_ori = Z(:, (1:num_tr(1)));
test_data_ori = Z(:, (num_tr(1)+1:end));
[distance] = KNN_Joint_rule(train_data_ori, test_data_ori, D_label,labels_1, K_1,K_2,tt_index,index_map);

for i = 1:1:size(tt_data,2)
    residual = distance(:,i)';
    temp = find(residual == min(residual));
    label_result(i) = temp(1);
end
[OA,AA,kappa,CA]=confusion(tt_label,label_result);
im_gt_1d(tt_index) = label_result;
im_gt_1d_reshape = reshape(im_gt_1d,i_row, i_col);
figure()
KJSRC_map = label2color(im_gt_1d_reshape,'india');
