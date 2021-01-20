function distance = KNN_Joint_rule(train_data_ori, test_data_ori, D_label,labels_1, K_1,K_2,tt_index,index_map)

[~, n] = size(test_data_ori);
distance = zeros(max(D_label), n);
i_row = size(index_map,1);i_col = size(index_map,2);
distance_old = [];sum_distance = [];
sum_distance_end = [];

for p = 1:n
    y_index = tt_index(p);
    y_jonit_index = index_map((labels_1==labels_1(y_index)));
    [~,tt_index_location,~] =...
        intersect(tt_index , y_jonit_index);
    y_jonit_data = test_data_ori( :,tt_index_location);
    for no = 1:size(y_jonit_data,2)
        y = y_jonit_data(:,no);
        for i = 1:max(D_label)
            X1 = train_data_ori(:, (D_label == i));
            d = sqrt(sum((repmat(y, 1, size(X1, 2)) - X1).^2));
            d_2 = sort(d,2, 'ascend');
            if size(d,2)<K_1
                K1 = size(d,2);
            else
                K1 = K_1;
            end
            sum_distance(i,no) = sum(d_2(1:K1)) / K1;
        end
    end
    sort_d_1 = sort(sum_distance,2, 'ascend');
    for i = 1:size(sum_distance,1)
        if size(sum_distance,2)<K_2
            K2 = size(sum_distance,2);
        else
            K2 = K_2;
        end
        sort_d = sort_d_1(i,:);
        distance_old_1(i,p) = sum(sort_d(1:K2)) / K2;
    end
end
distance = distance_old_1;
end
