function plottraj
    clear;clc;
    load("imupose.mat");
    
    figure(1)
    plot3(imupose(:,6),imupose(:,7),imupose(:,8));
    hold on;
    plot3(imuintpose0(:,6),imuintpose0(:,7),imuintpose0(:,8));
    plot3(imuintpose1(:,6),imuintpose1(:,7),imuintpose1(:,8));
    legend("gt","euler","mid")
    grid on;
    
    figure(2)
    subplot(3,1,1)
    plot(imupose(2:end,1),imuintpose0(:,6)-imupose(2:end,6));
    hold on;
    plot(imupose(2:end,1),imuintpose1(:,6)-imupose(2:end,6));
    grid on;
    legend("euler","mid");
    
    subplot(3,1,2)
    plot(imupose(2:end,1),imuintpose0(:,7)-imupose(2:end,7));
    hold on;
    plot(imupose(2:end,1),imuintpose1(:,7)-imupose(2:end,7));
    grid on;
    legend("euler","mid");
    
    subplot(3,1,3)
    plot(imupose(2:end,1),imuintpose0(:,8)-imupose(2:end,8));
    hold on;
    plot(imupose(2:end,1),imuintpose1(:,8)-imupose(2:end,8));
    grid on;
    legend("euler","mid");
end
