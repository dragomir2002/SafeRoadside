import pygame
import random

# Inicializa o Pygame
pygame.init()

# Configurações da janela
LARGURA = 1920
ALTURA = 1080
TAMANHO = (LARGURA, ALTURA)
FPS = 60

# Cores
BRANCO = (255, 255, 255)

# Configura a tela e o título
tela = pygame.display.set_mode(TAMANHO)
pygame.display.set_caption("Simulador de Movimentos")

# Carregar a imagem de fundo (certifique-se de que o caminho está correto)
background = pygame.image.load('road.png')
background = pygame.transform.scale(background, TAMANHO)

# Função para gerar cor aleatória
def cor_aleatoria():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Função para salvar as coordenadas no arquivo
def salvar_coordenadas(trajetorias):
    with open("trajetorias.txt", "a") as f:
        for traj in trajetorias:
            f.write("Traj: ")
            for coord in traj:
                f.write(f"({coord[0]}, {coord[1]}),")
            f.write("\n")  # Espaço entre trajetórias

# Função principal
def main():
    clock = pygame.time.Clock()

    # Lista para armazenar as trajetórias (listas de coordenadas)
    trajetorias = []
    movimento_ativo = False
    cor_atual = cor_aleatoria()
    coordenadas_atuais = []  # Usando um conjunto para a trajetória atual

    while True:
        tela.fill(BRANCO)  # Preenche o fundo com branco
        tela.blit(background, (0, 0))  # Coloca a imagem de fundo

        # Eventos
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                if coordenadas_atuais:  # Se houver coordenadas a salvar
                    trajetorias.append(coordenadas_atuais)
                salvar_coordenadas(trajetorias)  # Salva as trajetórias no arquivo
                pygame.quit()
                return

            if evento.type == pygame.MOUSEBUTTONDOWN:
                if evento.button == 1:  # Se o botão esquerdo do mouse for pressionado
                    movimento_ativo = True
                    cor_atual = cor_aleatoria()  # A cada nova interação, gera uma cor aleatória
                    coordenadas_atuais = []  # Reinicia o conjunto de coordenadas para essa nova interação

            if evento.type == pygame.MOUSEBUTTONUP:
                if evento.button == 1:  # Se o botão esquerdo do mouse for solto
                    movimento_ativo = False
                    trajetorias.append(coordenadas_atuais)  # Adiciona a trajetória final à lista de trajetórias
                    coordenadas_atuais = []  # Reinicia as coordenadas

            if evento.type == pygame.MOUSEMOTION:
                if movimento_ativo:
                    x, y = evento.pos
                    coordenadas_atuais.append((x, y))  # Adiciona a coordenada no conjunto

        # Desenha as bolas
        for traj in trajetorias:
            for coord in traj:
                pygame.draw.circle(tela, cor_atual, coord, 5)  # Desenha as bolas (em todas as trajetórias)

        pygame.display.flip()  # Atualiza a tela
        clock.tick(FPS)  # Controla o FPS

if __name__ == "__main__":
    main()