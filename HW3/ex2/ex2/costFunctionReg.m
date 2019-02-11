function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i = 1:m
    %    fprintf('i = %i, ', i);
    h = sigmoid(X(i,:)*theta);
    %fprintf('h = %f, ', h);
    part1 = y(i)*log(h);
    part2 = (1 - y(i))*log(1 - h);
    whole = -part1 - part2;
    J = J + whole;
    %fprintf('J = %f\n', J);
    for j = 1:size(theta)
        grad(j) = grad(j) + (h - y(i))*X(i,j);
    end
end

J = J / m;
%fprintf('J antes de reg: %f\n', J);

% Falta el termino de regularizacion
JR = 0;
for j=2:size(theta)
    JR = JR + theta(j)^2;
end
JR = JR * lambda / (2 * m);

J = J + JR;
%fprintf('J final: %f\n', J);

for j = 1:size(theta)
    grad(j) = grad(j) / m;
end

% Regularizacion de grad
grad(1) = grad(1) / m;
for j = 2:size(theta)
    grad(j) = (grad(j) / m) + (lambda / m) * theta(j);
end



% =============================================================

end
