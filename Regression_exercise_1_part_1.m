clearvars; clc; close all;
%% Model for training and test data gerenation
L_train = 100; % number of training data
sigma_noise = 0.3; % standard deviation for measurement noise
t = linspace(0,1,100); % time vector plotting the true model and tsting the learned model

X_train = 0 + (0+1).*rand(L_train,1); % generation of input training data to the model
Y_true = sin(2*pi*t); % true model
Y_train = sin(2*pi*X_train) + sigma_noise*randn(L_train,1); % generation of output training data
Train_set = [X_train,Y_train]; % complete data set
error_model = @(Y_true,Y_est) sqrt(mean((Y_true-Y_est).^2)); % inline function to compute the mean square error
%% Gradient descent parameters
L_iter = 3000; % number of iterations
eta = 0.25; % learning rate of the gradient descent (tuning parameter)

figure
plot(t,Y_true,'r','linewidth',2), hold on
plot(X_train,Y_train,'bo','linewidth',2), hold off
xlabel('X');ylabel('Y')
legend('True model','Training (observable) data')
grid on
%% Learning the model
M = 3; %polynomial order
W = zeros(1,M+1); % Weight initialization
% Transforming input training data to generate \phi_k. 
for k = 1 : M+1
    Phi(:,k) = X_train.^(k-1);
end
% Gradient descent learning of coefficients W
for i = 1 : L_iter
    for j = 1 : L_train
        e(j) = Y_train(j) - W*Phi(j,:)';
        W = W + eta*e(j)*Phi(j,:);
    end
    MSE(i) = mean(e.^2);
end

figure
plot(MSE,'linewidth',2)
xlabel('Number of iterations')
ylabel('Mean Square Error (MSE)')
grid on
%% Computing the weight coeffcients using Moore-Penrose pseudo inverse
W_MP = inv(Phi'*Phi)*Phi'*Y_train;
%% Testing the learned model
for k = 1 : M+1
    Phi_test(:,k) = t.^(k-1);
end

Y_test_gradient = W*Phi_test';
Y_test_MPinverse = W_MP'*Phi_test';
%% Evaluating the learned model 
figure
plot(t,Y_true,'r','linewidth',2), hold on
plot(X_train,Y_train,'bo','linewidth',2),
plot(t,Y_test_gradient,'gs-','linewidth',2),
plot(t,Y_test_MPinverse,'m+-','linewidth',2), hold off
xlabel('X');ylabel('Y')
legend('True model','Training (observable) data','Learned model (gradient descent)','Learned model (pseudo-inverse)')
grid on



