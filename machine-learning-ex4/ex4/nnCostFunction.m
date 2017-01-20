function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1: Forward Propagation

a1 = X; % assign activation 1
a1 = [ones(m,1) a1]; % add in bias term

a2 = sigmoid(a1*Theta1'); % assign activation 2
a2 = [ones(size(a2,1),1) a2]; % add in bias term

a3 = sigmoid(a2*Theta2'); % assign activation 3, h(x)

% create y vector with recoded values
yVec = zeros(size(y,1), num_labels); % initialize matrix

for i = 1:m % loop through all rows
    label = y(i); % assigns y value of row
    yVec(i,label) = 1; % assigns column value of 1 to corresponding label
end

cost = sum(-yVec.*log(a3) - (1 - yVec).*log(1-(a3))); % calculate cost

temp1 = Theta1(:, 2:end); % removes bias column from theta
temp2 = Theta2(:, 2:end);

s1 = sum(temp1.^2); % calculates the inner sums
s2 = sum(temp2.^2);

reg = lambda/(2*m) * (sum(s1) + sum(s2)); % calculates regularization term

J = 1/m * sum(cost) + reg;

% initialize delta_1 and delta_2 with zeros
delta_1 = zeros(size(Theta1));
delta_2 = zeros(size(Theta2));

for t = 1:m
    % calculate delta_3
    delta_3 = a3(t,:) - yVec(t,:);
    
    % add 1 to include bias unit for feedforward pass
    z2 = [1 a1(t,:)*Theta1'];
    
    % calculate delta_2 for hidden layer l = 2
    delt_2 = delta_3 * Theta2 .* sigmoidGradient(z2);

    % Accumulate the gradient skipping delta2_0
    delta_1 = delta_1 + (delt_2(2:end))' * a1(t,:);
    delta_2 = delta_2 + (delta_3)' * a2(t,:);
end

% Obtain the (unregularized) gradient by dividing by m
temp1 = [zeros(size(Theta1,1),1) temp1];
temp2 = [zeros(size(Theta2,1),1) temp2];

Theta1_grad = (1 / m) * delta_1 + lambda/m * temp1; 
Theta2_grad = (1 / m) * delta_2 + lambda/m * temp2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
