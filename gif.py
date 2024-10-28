import os
from PIL import Image

# Caminho para o diretório onde estão as imagens para o GIF
caminho_imagens = "./imagens/imagens_gif"

# Lista todos os arquivos na pasta e filtra apenas os arquivos de imagem
imagens = [os.path.join(caminho_imagens, arquivo) for arquivo in os.listdir(caminho_imagens) if arquivo.endswith(('.png', '.jpg', '.jpeg'))]

# Ordena as imagens (opcional, caso queira uma ordem específica)
imagens.sort()

# Carrega todas as imagens
frames = [Image.open(imagem) for imagem in imagens]

# Verifica se há imagens suficientes para o GIF
if frames:
    # Salva o GIF com uma taxa de atualização de 500 ms por frame
    frames[0].save(
        "reta_de_regressao.gif",
        save_all=True,
        append_images=frames[1:],  # Adiciona o restante das imagens
        duration=500,              # Duração de cada frame em milissegundos
        loop=0                     # loop=0 faz o GIF repetir indefinidamente
    )
    print("GIF criado com sucesso!")
else:
    print("Nenhuma imagem encontrada no diretório especificado.")
