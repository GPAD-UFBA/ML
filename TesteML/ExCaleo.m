load('DataImagem');
load('Pesos');
X = DataImagem;
lambda = 1;

Parametros = [Theta1(:) ; Theta2(:)];
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 2; %Triangulo ou Circulo

[J grad] = nnCostFunction_Caleo(Parametros, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

options = optimset('MaxIter', 100);

costFunction = @(p) nnCostFunction_Caleo(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
                               
[nn_params, cost] = fmincg(costFunction, Parametros, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
             
h = Previsao(X,Theta1,Theta2);
Largura = 20;
Altura = 20;
MatrizImagens(DataImagem,Largura, Altura, h);
close All;