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
z=X*theta;
predictor=1./(1+exp(-z));


grad = zeros(size(theta));
grad=((1/m)*(X'*(predictor-y)))+((lambda/m)*theta);
grad(1)=grad(1)-((lambda/m)*theta(1));


J=((y'*log(predictor))+((1-y)'*log(1-predictor)));
J=-J/m;
J=J+(sum(theta(2:end).^2)*(lambda/(2*m)));

end

