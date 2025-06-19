function [UnLabel_Matrix_Fill] = fill_labels(Train_Matrix, Train_Label, UnLabel_Matrix, k)
    % Assigning Pseudo-Labels
    
    [label_num, ~] = size(Train_Matrix);
    
    [unlabel_num, ~] = size(UnLabel_Matrix);
    
    [~, label_dim] = size(Train_Label);
    
    Train_Label(Train_Label == -1) = 0;
    
    weight = linspace(k, 1, k);
    
    UnLabel_Matrix_Fill = [];
    
    temp_Train_Matrix = Train_Matrix;
    temp_Train_Label = Train_Label;
    
    for i = 1:unlabel_num

        dists = pdist2(UnLabel_Matrix(i,:), temp_Train_Matrix, 'Euclidean');
        
        [~, dists_rank] = sort(dists, 2, 'ascend');
        
        weight_labels = zeros(1, label_dim);
        count_labels = zeros(1, label_dim);
        labels = zeros(1, label_dim);
        
        for j = 1:k
            rk = dists_rank(j);
            tmp_weight_labels = temp_Train_Label(rk, :) * weight(j);
            weight_labels = weight_labels + tmp_weight_labels;
            count_labels = count_labels + temp_Train_Label(rk, :);
        end

        max_weight = max(weight_labels);
        
        for q = 1:label_dim
            if weight_labels(1, q) >= max_weight
                labels(:, q) = 1;
            else
                labels(:, q) = count_labels(:, q) / k;
            end
        end
        
        fill_labels = labels;
        
        temp_Train_Matrix = [temp_Train_Matrix; UnLabel_Matrix(i, :)];
        temp_Train_Label = [temp_Train_Label; fill_labels];
        UnLabel_Matrix_Fill = [UnLabel_Matrix_Fill; fill_labels];
    end
    
    mean_label_fill = mean(UnLabel_Matrix_Fill, 1);
    
    for m = 1:label_dim
        if mean_label_fill(:, m) ~= 0
            for n = 1:unlabel_num
                if UnLabel_Matrix_Fill(n, m) >= mean_label_fill(:, m)
                    UnLabel_Matrix_Fill(n, m) = 1;
                else
                    UnLabel_Matrix_Fill(n, m) = 0;
                end
            end
        end
    end

    UnLabel_Matrix_Fill(UnLabel_Matrix_Fill == 0) = -1;
end



