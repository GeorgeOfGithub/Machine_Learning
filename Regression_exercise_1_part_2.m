clearvars; clc; close all;
%% Model for training and test data gerenation
L = 200; % size of the data set
sigma_noise = 0.3; % standard deviation of measurement noise
X = 0 + (0+1).*rand(L,1); % input data
Y = sin(2*pi*X) + sigma_noise*randn(L,1); %output data
D=[X,Y]; % data set
error_model = @(Y_true,Y_est,N) sqrt(mean(abs(Y_true-Y_est)).^2/N);
M = 11; % order of the employed polynomial mode    
%% Splitting the data set in training and test sets
L_test = 100; % size of the data set used for testing
Q = L/L_test; 
for q = 1 : Q
X_test = X((q-1)*L_test + 1 : q*L_test);
Y_test = Y((q-1)*L_test + 1 : q*L_test);

X_train = X;
Y_train = Y;
X_train((q-1)*L_test + 1 : q*L_test) = [];
Y_train((q-1)*L_test + 1 : q*L_test) = [];
%% Learning the model
for a = 1 : M
    % Transforming input vector
    for k = 1 : a+1
        Phi_train(:,k) = X_train.^(k-1);
    end    
    %% Computing the coeffcients using Moore-Penrose pseudo inverse
    T = Phi_train'*Phi_train;
    [r,c] = size(T);
    lambda = 1e-3;  % tuning parameter
    W_MP = (lambda*eye(r) + T)\Phi_train'*Y_train;
    %% Testing the learned model
    for k = 1 : a+1
        Phi_test(:,k) = X_test.^(k-1);
    end
    
    Y_pred_test = W_MP'*Phi_test';
    Y_pred_train = W_MP'*Phi_train';
    
    test_error(a) = error_model(Y_test,Y_pred_test',length(X_test));
    train_error(a) = error_model(Y_train,Y_pred_train',length(X_train));
end

test_error_M(q,:) = test_error;
train_error_M(q,:) = train_error;


plot(train_error,'r','linewidth',2), hold on
plot(test_error,'b','linewidth',2), hold on
xlabel('Model order M')
ylabel('Root Mean Equare Error (RMSE)')
legend('Train error','Test error')
grid on
end

test_error_av = mean(test_error,1);
train_error_av = mean(train_error,1);

figure
plot(train_error_av,'r','linewidth',2), hold on
plot(test_error_av,'b','linewidth',2), hold off
xlabel('Model order M')
ylabel('Averaged Root Mean Square Error (RMSE)')
legend('Train error','Test error')
grid on

