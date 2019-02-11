function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

rows = size(z,1);
cols = size(z,2);

if rows==1 && cols==1
    g =  1 / (1 + exp(-z));
elseif cols==1
    for i = 1:rows
        g(i) = 1 / (1 + exp(-z(i)));
    end
elseif rows==1
    for i = 1:cols
        g(1,i) = 1 / (1 + exp(-z(1,i)));
    end
else
    for i = 1:rows
        for j = 1:cols
            g(i,j) = 1 / (1 + exp(-z(i,j)));
        end
    end
endif


% =============================================================

end
