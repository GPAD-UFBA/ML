function M = MatrizImagens(DataImagem,Largura, Altura, h)


M = reshape(DataImagem',[Altura,Largura,size(DataImagem,1)]);
colormap(gray);

for i=1:size(h,1)
    image(M(:,:,i))
    hold on
    if h(i,1) == 1
        title('Triângulo');
    else
        title('Circulo');
    end
fprintf('Program paused. Press enter to continue.\n');
pause;

end

    


