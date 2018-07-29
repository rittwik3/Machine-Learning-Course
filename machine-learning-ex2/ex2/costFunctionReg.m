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

% z=sigmoid(X*theta);
% grad(1)= (1/m)*(z(1)-y(1))'*X(1);
% J(1)=(1/m)*(-y(1)'*log(z(1)) - (1-y(1))'*log(1-z(1)));p=grad(1);q=J(1);
% theta(1)=0;
% grad= (1/m)*((z-y)'*X +lambda*theta);
% J=(1/m)*(-y'*log(z) - (1-y)'*log(1-z)) + (lambda/(2*m))*sum(theta.*theta);
% grad(1)=p;
% J(1)=q;

  
    
 
% h_theta = sigmoid(X*theta);
% J = (-1/m)*sum(y.*log(h_theta) + (1-y).*log(1-h_theta)) + (lambda/(2*m))*sum(theta(2:length(theta)).^2);
%grad(1) = (1/m)*(X')*(h_theta - y);
%grad(2:size(theta,1)) = 1/m * (X'(2:size(X',1),:)*(h_theta - y) + lambda*theta(2:size(theta,1),:));
%grad(:,2:length(grad)) = grad(:,2:length(grad)) + (lambda/m)*theta(2:length(theta))';

J = ( (1 / m) * sum(-y'*log(sigmoid(X*theta)) - (1-y)'*log( 1 - sigmoid(X*theta))) ) + (lambda/(2*m))*sum(theta(2:length(theta)).*theta(2:length(theta))) ;

 z=sigmoid(X*theta);
 grad(1)= (1/m)*(z(1)-y(1))'*X(1);
grad(:,2:length(grad)) = grad(:,2:length(grad)) + (lambda/m)*theta(2:length(theta))';

% =============================================================

end
