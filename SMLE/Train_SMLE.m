function [result, std_result] = Train_SMLE(features, labels, labeled_proportion, parm)

    para.rep = 5;

    if exist('features', 'var') == 1
        data = features;
        target = labels;
        data = normalize(data, 2, 'norm');
        data(find(isnan(data))) = 0;
    end
    
    if exist('train_data','var') == 1
        data = [train_data; test_data];
        target = [train_target'; test_target'];
        clear train_data train_target test_data test_target;
        data(find(isnan(data))) = 0;
    end

    ind = find(target == -1);
    target(ind) = 0;
    num_data = size(data, 1);

    rng(123);
    randorder = randperm(num_data);
    PRO = zeros(4, 1);

    for t = 1 : 5
        SS = [];

        [X_train, Y_train, X_test, Y_test] = generateCVSet(data, target, randorder, t, para.rep);

        [train_num, ~] = size(X_train);

        Y_train(Y_train == 0) = -1;
        Y_test(Y_test == 0) = -1;

        R = randperm(train_num);

        labeled_num = round(train_num * labeled_proportion);
        unlabel_num = train_num - labeled_num;

        train_h_features = X_train(R(1:labeled_num), :);
        train_h_labels = Y_train(R(1:labeled_num), :);
        if unlabel_num == 0
            train_n_features = [];
        else
            train_n_features = X_train(R(labeled_num + 1 : train_num), :);
        end

        X_train = [train_h_features; train_n_features];

        [train_n_labels_Fill] = fill_labels(train_h_features, train_h_labels, train_n_features, parm.k);

        [LE] = label_enhancement(train_h_features, train_h_labels, train_n_features, train_n_labels_Fill, parm.k);

        [Wd, Wf, bd, bf, objective] = SMLE(X_train, LE, train_h_labels, parm);

        sample_size = size(X_test, 1);
        I1 = ones(sample_size, 1);
        Outputs = parm.alpha * (X_test * Wd + I1 * bd) + (1 - parm.alpha) * (X_test * Wf + I1 * bf);

        Pre_Labels = sign(Outputs - 0.5);
        Y_test(Y_test == 0) = -1;

        TT(:, 1) = EvaluationAll(Pre_Labels', Outputs', Y_test');
        SS = [SS, TT];

        FF = SS(1, :);
        tt = find(FF == max(FF));
        PRO(:, t) = SS(:, tt(1));
        % fprintf('fold: %d is finish\n', t);
    end

    result = mean(PRO, 2);
    std_result = std(PRO, 1, 2);
end



