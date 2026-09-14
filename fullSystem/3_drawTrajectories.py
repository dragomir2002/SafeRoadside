"""
Step 3 - Draw reference trajectories over withguide.png.

Controls
    left-click + drag    freehand trajectory
    right-click twice    straight line between the two points
    u                    undo last trajectory
    s                    save now (without quitting)
    ESC / close window   save and quit

COORDINATE SPACE (important)
    The background is shown fit-to-window, but every saved point is scaled back
    to the FULL RESOLUTION of withguide.png. trajetorias.txt therefore always
    lives in the same pixel space as road.png / map.txt / the source video.

    The previous version of this script recorded raw 1920x1080 window
    coordinates, so on any scene whose imagery was not exactly 1920x1080 the
    reference trajectories silently disagreed with the detector's coordinates.
"""
import pygame
import random
import sys
import os
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="SafeRoadside step 3 - draw reference trajectories")
    p.add_argument("--background", default="withguide.png",
                   help="Background image. Default withguide.png.")
    p.add_argument("--out", default="trajetorias.txt",
                   help="Output trajectories file. Default trajetorias.txt.")
    p.add_argument("--max-window", type=int, default=1600,
                   help="Max window width in pixels. Default 1600.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite the output file instead of appending.")
    return p.parse_args()


def cor_aleatoria():
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))


def interpolar_pontos(p1, p2, num_pontos=30):
    """Return num_pontos+1 points along the straight line p1 -> p2."""
    x1, y1 = p1
    x2, y2 = p2
    pontos = []
    for i in range(num_pontos + 1):
        t = i / num_pontos
        pontos.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
    return pontos


def main():
    args = parse_args()

    if not os.path.exists(args.background):
        print(f"[ERRO] background not found: {args.background}")
        sys.exit(1)

    pygame.init()

    background_full = pygame.image.load(args.background)
    full_w, full_h = background_full.get_size()

    # Fit the image into a window no wider than --max-window, preserving aspect.
    scale = min(1.0, args.max_window / full_w)
    win_w, win_h = int(full_w * scale), int(full_h * scale)

    tela = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(
        f"Trajectories - {args.background} ({full_w}x{full_h} shown at {win_w}x{win_h})"
    )
    background = pygame.transform.smoothscale(background_full, (win_w, win_h))

    print(f"[INFO] background {full_w}x{full_h} displayed at {win_w}x{win_h} "
          f"(scale {scale:.4f})")
    print(f"[INFO] saved coordinates are in FULL {full_w}x{full_h} space")
    print("[INFO] left-drag = freehand | right-click x2 = line | u = undo | "
          "s = save | ESC = save+quit")

    def to_full(pt):
        """Window coords -> full-resolution image coords."""
        return (int(round(pt[0] / scale)), int(round(pt[1] / scale)))

    def salvar(trajetorias):
        if not trajetorias:
            print("[INFO] nothing to save")
            return
        mode = "w" if args.overwrite else "a"
        with open(args.out, mode) as f:
            for traj in trajetorias:
                pts = [to_full(p) for p in traj["pontos"]]
                f.write("Traj: " + ", ".join(f"({x}, {y})" for x, y in pts) + "\n")
        print(f"[OK] {len(trajetorias)} trajectories -> {args.out} "
              f"(mode={'overwrite' if args.overwrite else 'append'})")

    clock = pygame.time.Clock()
    trajetorias = []
    movimento_ativo = False
    coordenadas_atuais = []
    cor_atual = cor_aleatoria()
    ponto_reta = None
    saved = False

    while True:
        tela.blit(background, (0, 0))

        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                if coordenadas_atuais:
                    trajetorias.append({"pontos": coordenadas_atuais, "cor": cor_atual})
                salvar(trajetorias)
                pygame.quit()
                return

            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE:
                    if coordenadas_atuais:
                        trajetorias.append({"pontos": coordenadas_atuais, "cor": cor_atual})
                    salvar(trajetorias)
                    pygame.quit()
                    return
                if evento.key == pygame.K_u and trajetorias:
                    trajetorias.pop()
                    print(f"[INFO] undo - {len(trajetorias)} trajectories left")
                if evento.key == pygame.K_s:
                    salvar(trajetorias)
                    trajetorias = []
                    saved = True

            if evento.type == pygame.MOUSEBUTTONDOWN:
                x, y = evento.pos
                if evento.button == 1:
                    movimento_ativo = True
                    cor_atual = cor_aleatoria()
                    coordenadas_atuais = [(x, y)]
                elif evento.button == 3:
                    if ponto_reta is None:
                        ponto_reta = (x, y)
                        print(f"[INFO] line start at window {(x, y)} "
                              f"-> full {to_full((x, y))}")
                    else:
                        trajetorias.append({
                            "pontos": interpolar_pontos(ponto_reta, (x, y), 30),
                            "cor": cor_aleatoria(),
                        })
                        ponto_reta = None

            if evento.type == pygame.MOUSEBUTTONUP:
                if evento.button == 1:
                    movimento_ativo = False
                    if len(coordenadas_atuais) > 1:
                        trajetorias.append({"pontos": coordenadas_atuais, "cor": cor_atual})
                    coordenadas_atuais = []

            if evento.type == pygame.MOUSEMOTION and movimento_ativo:
                coordenadas_atuais.append(evento.pos)

        for traj in trajetorias:
            pts = traj["pontos"]
            for i in range(1, len(pts)):
                pygame.draw.line(tela, traj["cor"], pts[i - 1], pts[i], 3)

        if movimento_ativo and len(coordenadas_atuais) > 1:
            for i in range(1, len(coordenadas_atuais)):
                pygame.draw.line(tela, cor_atual,
                                 coordenadas_atuais[i - 1], coordenadas_atuais[i], 3)

        if ponto_reta is not None:
            pygame.draw.circle(tela, (255, 0, 0), ponto_reta, 5)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
