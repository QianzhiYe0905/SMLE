function [LE] = label_enhancement(Train_Matrix, Train_Label, UnLabel_Matrix, UnLabel_Matrx_Fill, k)
    % Ehancing Training Data
    
    Train_Matrix_All = [Train_Matrix; UnLabel_Matrix];
    Train_Label_All = [Train_Label; UnLabel_Matrx_Fill];
    [train_num,label_dim] = size(Train_Label_All);
    
    Train_Label_All(Train_Label_All == -1) = 0;    
    
    dists = pdist2(Train_Matrix_All,Train_Matrix_All, 'Euclidean');
    [sort_dists, dists_rank] = sort(dists, 2);
    
    weight = linspace(k, 1, k);
    
    LE = zeros(train_num,label_dim);
    
    for i = 1:train_num
        knn_labels = zeros(1,label_dim);
        for j = 1:k
            rk = dists_rank(j);
            temp_label = Train_Label_All(rk,:) * weight(j);
            knn_labels = knn_labels + temp_label;
        end
        LE(i,:) = knn_labels/k;
    end
   
    LE = LE + Train_Label_All;
    for i = 1 : train_num
        for j = 1 : label_dim
            val = LE(i, j);
            if val >= 1
                LE(i, j) = 1;
            elseif val > 0
                LE(i, j) = val;
            else
                LE(i, j) = -1;
            end
        end
    end
end

