function [Wd, Wf, bd, bf, objective] = SMLE(features, ldl, train_h_labels, parm)
    % Optimization

    loss_list = [];
    
    un_label_size = size(ldl, 1) - size(train_h_labels, 1);    
    [labeled_num, label_dim] = size(train_h_labels);    
    Y = [train_h_labels; zeros(un_label_size, label_dim)];
    [Wd, Wf, L, F, Q, bd, bf] = init(features, ldl, train_h_labels, labeled_num, parm);
    loss = calc_loss(features, ldl, F, L, Q, Wd, Wf, bd, bf, parm);
    iter = 1;
    obji = 1;
    objective(iter) = loss;
    iter = iter + 1;
    
    while 1
        bd = update_bd(features, ldl, Wd, Wf, L, F, Q, parm);
        bf = update_bf(features, ldl, Wd, Wf, L, F, Q, parm);
        Wd = update_Wd(features, ldl, Wd, Wf, L, F, Q, bd, parm);
        Wf = update_Wf(features, ldl, Wd, Wf, L, F, Q, bf, parm);
        F = update_F(features, ldl, Wd, Wf, bd, bf, Y, parm);
        
        [sample_size, label_size] = size(ldl);
        I1 = ones(sample_size, 1);
        res = (features * 1 / 2 * (Wd + Wf)) + (1 / 2 * I1 * (bd + bf)) + Y;
        for i = 1 : sample_size
            for j = 1 : label_size
                val = res(i, j);
                if val >= 1
                    ldl(i, j) = 1;
                elseif val > 0
                    ldl(i, j) = val;
                else
                    ldl(i, j) = -1;
                end
            end
        end
        
        loss = calc_loss(features, ldl, F, L, Q, Wd, Wf, bd, bf, parm);
        objective(iter) = loss;
        loss_list = [loss_list, loss];
        cver = abs((loss - obji)/obji);
        obji = objective(iter);
        iter = iter + 1;
        if (cver < 10^-3 || iter > parm.max_iter) , break, end
    end
end

function[Q] = init_Q(labels, labeled_num)
    label_size = size(labels, 1);
    Q = eye(label_size);
    for i = labeled_num + 1 : label_size
        Q(i, i) = 0.5;
    end
end

function [Wd, Wf, L, F, Q, bd, bf] = init(features, labels, train_h_labels, labeled_num, parm)
    [sample_size, feature_dim] = size(features);
    label_dim = size(labels, 2);
    Wd = rand(feature_dim, label_dim);
    Wf = rand(feature_dim, label_dim);
    tmp_S = pdist(features);
    S = squareform(tmp_S);
    
    S_normalized = zeros(size(S));

    for i = 1:size(S, 1)
        distances = S(i, :);

        [sorted_distances, idx] = sort(distances);
        k_nearest_idx = idx(2:parm.k+1);

        k_nearest_distances = sorted_distances(2:parm.k+1);

        sum_k_distances = sum(k_nearest_distances);
        if sum_k_distances > 0
            normalized_distances = k_nearest_distances / sum_k_distances;
        else
            normalized_distances = k_nearest_distances;
        end

        S_normalized(i, k_nearest_idx) = normalized_distances;
    end
    
    S = S_normalized;

    L = (eye(sample_size) - S)' * (eye(sample_size) - S);
    un_label_size = size(labels, 1) - size(train_h_labels, 1);
    F = [train_h_labels; zeros(un_label_size, label_dim)];
    Q = init_Q(labels, labeled_num);
    bd = rand(1, label_dim);
    bf = rand(1, label_dim);
end

function [res] = update_bd(features, ldl, Wd, Wf, L, F, Q, parm)
    sample_size = size(features, 1);
    I1 = ones(sample_size, 1);
    m = I1' * Q * I1;
    part1 = 1 / m * I1' * Q * ldl;
    part2 = 1 / m * I1' * Q * features * Wd;
    res = part1 - part2;
end

function [res] = update_bf(features, ldl, Wd, Wf, L, F, Q, parm)
    sample_size = size(features, 1);
    I1 = ones(sample_size, 1);
    
    m = I1' * Q * I1;
    part1 = 1 / m * I1' * Q * F;
    part2 = 1 / m * I1' * Q * features * Wf;
    res = part1 - part2;
end

function [res] = update_F(features, ldl, Wd, Wf, bd, bf, Y, parm)
    [sample_size, label_size] = size(ldl);
    I1 = ones(sample_size, 1);

    res = parm.alpha * (features * Wd + I1 * bd) + (1-parm.alpha) * (features * Wf + I1 * bf) + Y;
    
    for i = 1 : sample_size
        for j = 1 : label_size
            val = res(i, j);
            if val >= 1
                res(i, j) = 1;
            elseif val > 0
                res(i, j) = val;
            else
                res(i, j) = 0;
            end
        end
    end
end

function [res] = update_Wd(features, ldl, Wd, Wf, L, F, Q, bd, parm)
    [sample_size, feature_dim] = size(features);
    X = features;
    D = ldl;
    H = X' * Q;
    E = X' * L * X;
    I = eye(feature_dim);
    I1 = ones(sample_size, 1);

    part1 = (parm.alpha * H * X) + (parm.lambda1* I) + (parm.alpha * parm.alpha * parm.lambda3 * E);
    part2 = (parm.alpha * H * D) - (parm.alpha * X' * Q * I1 * bd) - (parm.alpha * (1 - parm.alpha) * parm.lambda3 * E * Wf);
    
    res = inv(part1) * part2;
end

function [res] = update_Wf(features, ldl, Wd, Wf, L, F, Q, bf, parm)
    [sample_size, feature_dim] = size(features);
    X = features;
    D = ldl;
    H = X' * Q;
    E = X' * L * X;
    I = eye(feature_dim);
    I1 = ones(sample_size, 1);

    part1 = (parm.beta * H * X) + (parm.lambda2* I) + ((1 - parm.alpha)^2 * parm.lambda3 * E);
    part2 = (parm.beta * H * F) - (parm.beta * X' * Q * I1 * bf) - (parm.alpha * (1 - parm.alpha) * parm.lambda3 * E * Wd);
    
    res = inv(part1) * part2;
end

function [res] = calc_loss(features, labels, F, L, Q, wd, wf, bd, bf, parm)
    sample_size = size(features, 1);
    X = features;
    D = labels;
    W = parm.alpha * wd + (1-parm.alpha) * wf;
    
    I1 = ones(sample_size, 1);
    part1 = parm.alpha * trace((X * wd + I1 * bd - D)' * Q * (X * wd + I1 * bd - D));
    part2 = parm.beta * trace((X * wf + I1 * bf - F)' * Q * (X * wf + I1 * bf - F));
    part3 = parm.lambda1 * trace(wd' * wd);
    part4 = parm.lambda2 * trace(wf' * wf);
    part5 = parm.lambda3 * trace(W' * X' * L * X * W);
    res = 1/2 * (part1 + part2 + part3 + part4 + part5);
end