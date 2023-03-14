clear all
clc
close all


%% Learn AND, OR and XOR gates
clear all

AND = [0,0,-1;0,1,-1;1,0,-1;1,1,1];
OR = [0,0,-1;0,1,1;1,0,1;1,1,1]; 
XOR = [0,0,-1;0,1,1;1,0,1;1,1,-1];

M = 2; %input dim. 
w = zeros(1,M+1); 
eta = 1e-5; 
L_train = 2e3;
L_test = 2e3;

s_test = randi(4,1,L_test);
s_train = randi(4,1,L_train);

X_train_LG = OR(s_train,:);
X_test_LG = OR(s_test,:);


M = 2;  % number of inputs
eta = 1e-6; % learning rate
x = [ones(length(X_train_LG),1),X_train_LG];
x_t = [ones(length(X_test_LG),1),X_test_LG]; 
w = zeros(1,M+1); % initialize weights 

%Training non-batch
for k = 1 : length(X_train_LG)
    y = sign(w*x(k,1:3)');
    e(k) = x(k,4) - y;
    w = w + eta*e(k)*x(k,1:3);
end
%Test 
for k = 1 : length(X_test_LG)
    y = sign(w*x_t(k,1:3)');
    e_test(k) = x_t(k,4) - y;
end

figure
stem(e,'r'), hold on
stem(e_test,'b'), hold off
legend('Training (sample)','Test')
xlabel('Iterations')
ylabel('Error')
