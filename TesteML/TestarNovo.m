function TestarNovo(Imagem,Theta1,Theta2)

DataImagem = Imagem(:,:,1);
X = double(DataImagem(:))';

m = size(X,1);
a1 = [ones(m,1) X];

z1 = a1; %20X401
z2 = a1*Theta1'; %20X401 * 401X25
a2 = sigmoid(z2);%20X25
a2 = [ones(m,1) a2];%20X26
z3 = a2*Theta2';%20X26 * 26X2
a3 = sigmoid(z3);%20X2
h = a3;

[dummy h] = max(h,[],2);

colormap(gray);

image(DataImagem);
hold on
h
if h == 1
    title('Triângulo');
else
    title('Circulo');
end


