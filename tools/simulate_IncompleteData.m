function simulate_IncompleteData( data_name, X_train, Y_train, X_test, Y_test, labeled_proportion, incomplete)
% load dataset
fprintf(data_name,'\n');

[train_num,~] = size(X_train);
[~,label_dim] = size(Y_train);

Y_train(Y_train == 0) = -1; Y_test(Y_test == 0) = -1;


R = randperm(train_num);

labeled_num = round(train_num*labeled_proportion);
unlabel_num = train_num-labeled_num;

random_train_Index = randperm(labeled_num*label_dim);

Train_Matrix = X_train(R(1:labeled_num),:);
Train_Label = Y_train(R(1:labeled_num),:);
UnLabel_Matrix = X_train(R(labeled_num+1:labeled_num + unlabel_num),:);
Test_Matrix = X_test;
Test_Label = Y_test;

 
        
Z = createIncomplete_Label(Train_Label, incomplete,random_train_Index);

save([data_name,'/',data_name,'Infor',num2str(incomplete),'.mat'],'Train_Matrix','Test_Matrix','Z','Train_Label','Test_Label','UnLabel_Matrix','-v7');
