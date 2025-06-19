function ResultAll = EvaluationAll(Pre_Labels,Outputs,test_target)
    
    ResultAll = zeros(4,1); 
    
    tmp_Y_test = test_target;
    tmp_Y_test(tmp_Y_test == -1) = 0;
    
    AvgAuc            = Avgauc(Outputs, test_target);
    Average_Precision = Average_precision(Outputs,test_target);
    RankingLoss       = Ranking_loss(Outputs,test_target);
    Coverage          = coverage(Outputs,test_target);

        
    ResultAll(1,1)    = AvgAuc;
    ResultAll(2,1)    = Average_Precision;
    ResultAll(3,1)    = RankingLoss;
    ResultAll(4,1)    = Coverage;
end