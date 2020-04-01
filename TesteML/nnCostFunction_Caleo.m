function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
                               
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X,1);

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

a1 = [ones(m,1) X];
z1 = a1; %20X401
z2 = a1*Theta1'; %20X401 * 401X25
a2 = sigmoid(z2);%20X25
a2 = [ones(m,1) a2];%20X26
z3 = a2*Theta2';%20X26 * 26X2
a3 = sigmoid(z3);%20X2
h = a3;
y2 = y;
log1 = log(h);
log2 = log(1-h);


J1 = (((-y2)*log1') - ((1-y2)*log2'));
J2 = diag(J1);

reg = lambda*(sum(sum(Theta1(:,2:size(Theta1,2)).^2)) + sum(sum(Theta2(:,2:size(Theta2,2)).^2)))/(2*m);

J = sum(J2)/m + reg;

d3 = h-y2;



d2 = (Theta2(:,2:end)'*d3')'.*sigmoidGradient(z2);



Theta1_grad = (Theta1_grad + d2'*a1)/m;
Theta2_grad = (Theta2_grad + d3'*a2)/m;

Theta1_grad = Theta1_grad + [zeros(size(Theta1_grad,1),1) (lambda/m)*Theta1(:,2:end)];
Theta2_grad = Theta2_grad + [zeros(size(Theta2_grad,1),1) (lambda/m)*Theta2(:,2:end)];

grad = [Theta1_grad(:) ; Theta2_grad(:)];