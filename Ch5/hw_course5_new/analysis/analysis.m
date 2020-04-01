function analysis()
    pos = load("../build/poses.txt");
    solver = load("../build/solver.txt");
    pos0 = [0,0,0,-1.0718,4,0.866025,-4,6.9282,0.866025];
    delta_pos = bsxfun(@minus,pos, pos0);
    
    pos_rmse = zeros(50,13);
    iteration_rmse = zeros(50,13);
    time_rmse = zeros(50,13);
    x=[0, 0.5, 5, 50, 5e2, 5e3, 5e4, 5e5, 5e6, 5e7, 5e8, 5e9, 5e10];
    for i=1:1:13
        rmse = 0;
        for j=1:1:50
            % pos rmse
            dist = (norm(delta_pos((i-1)*50 + j, 1:3))...
                +norm(delta_pos((i-1)*50 + j, 4:6))...
                +norm(delta_pos((i-1)*50 + j, 7:9)))/3;
            rmse = rmse + dist;
            pos_rmse(j,i) = dist;
            
        end 
        iteration_rmse(:, i) = solver((i-1)*50 +1:i*50,1);
        time_rmse(:, i) = solver((i-1)*50 +1:i*50,2);
    end
    figure(1)
    semilogx(x,mean(pos_rmse));
    hold on;
    xlabel("先验权重");
    ylabel("RMSE(m)");
    grid on
    figure(2)
    subplot(2,1,1)
    boxplot(iteration_rmse, 'Labels',{'0','0.5','5','50', '5e2', '5e3', '5e4', '5e5', '5e6', '5e7', '5e8', '5e9', '5e10'});
    hold on;
    plot(iteration_rmse(1,:));
    grid on;
    xlabel("先验权重");
    ylabel("迭代次数");
    subplot(2,1,2)
    boxplot(time_rmse(2:end,:), 'Labels',{'0','0.5','5','50', '5e2', '5e3', '5e4', '5e5', '5e6', '5e7', '5e8', '5e9', '5e10'});
    hold on;
    plot(mean(time_rmse(2:end,:)));
    grid on;
    xlabel("先验权重");
    ylabel("计算耗时(ms)");
end