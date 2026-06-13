"""Integra entrada, processamento, detecção, avaliação e exibição."""

import sys
from collections import deque
from time import perf_counter

import cv2
import numpy as np

from avaliacao import estabilizar_resultado, salvar_resultados
from deteccao import classificar_estado
from processamento import (
    ler_argumentos,
    ler_quadros,
    preprocessar,
    redimensionar,
    segmentar,
    selecionar_roi,
)

CORES = {
    "VERMELHO": (0, 0, 255),
    "AMARELO": (0, 255, 255),
    "VERDE": (0, 255, 0),
    "DESCONHECIDO": (255, 255, 255),
}


def criar_painel(imagem, titulo, largura=320, altura=210):
    """Ajusta uma etapa do processamento para o painel final."""
    if imagem.ndim == 2:
        imagem = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)

    escala = min(largura / imagem.shape[1], (altura - 30) / imagem.shape[0])
    tamanho = tuple(max(1, int(valor * escala)) for valor in imagem.shape[1::-1])
    imagem = cv2.resize(imagem, tamanho, interpolation=cv2.INTER_AREA)

    painel = np.full((altura, largura, 3), 25, dtype=np.uint8)
    x = (largura - tamanho[0]) // 2
    y = 30 + (altura - 30 - tamanho[1]) // 2
    painel[y : y + tamanho[1], x : x + tamanho[0]] = imagem
    cv2.putText(
        painel, titulo, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
    )
    return painel


def mostrar_resultado(frame, recorte, processada, mascaras, limpas, roi, resultado):
    """Exibe a entrada, a detecção e as máscaras em uma janela."""
    x, y, largura, altura = roi
    texto = f"Estado: {resultado['estado']} ({resultado['confianca']:.0%})"
    cor = CORES[resultado["estado"]]

    original = frame.copy()
    cv2.rectangle(original, (x, y), (x + largura, y + altura), (255, 255, 255), 2)
    cv2.putText(original, texto, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)

    detectada = processada.copy()
    if resultado["regiao"]:
        rx, ry, rw, rh = resultado["regiao"]
        cv2.rectangle(detectada, (rx, ry), (rx + rw, ry + rh), cor, 2)

    etapas = [
        (original, "Quadro original"),
        (recorte, "Regiao selecionada"),
        (detectada, texto),
    ]
    etapas += [
        (mascara, f"{cor.capitalize()} - {tipo}")
        for tipo, grupo in (("segmentada", mascaras), ("limpa", limpas))
        for cor, mascara in grupo.items()
    ]
    paineis = [criar_painel(imagem, titulo) for imagem, titulo in etapas]
    linhas = [np.hstack(paineis[i : i + 3]) for i in range(0, 9, 3)]
    cv2.imshow("Detector de semaforo", np.vstack(linhas))


def processar_quadro(frame, roi, args):
    frame = redimensionar(frame)
    recorte, roi = selecionar_roi(frame, roi)
    processada = preprocessar(recorte, args.filtro, args.ampliacao)
    mascaras, limpas = segmentar(processada, args.kernel, args.iteracoes)
    resultado = classificar_estado(limpas)
    return frame, recorte, processada, mascaras, limpas, roi, resultado


def executar(args):
    roi = args.roi
    historico = deque(maxlen=5)
    registros = []

    try:
        for numero, (frame, imagem_unica) in enumerate(ler_quadros(args.entrada), 1):
            inicio = perf_counter()
            dados = processar_quadro(frame, roi, args)
            frame, recorte, processada, mascaras, limpas, roi, resultado = dados

            historico.append(resultado)
            if not imagem_unica:
                resultado = estabilizar_resultado(historico)

            tempo_ms = (perf_counter() - inicio) * 1000
            registros.append(
                {
                    "quadro": numero,
                    "estado": resultado["estado"],
                    "confianca": resultado["confianca"],
                    "tempo_ms": round(tempo_ms, 2),
                }
            )
            mostrar_resultado(
                frame, recorte, processada, mascaras, limpas, roi, resultado
            )

            if imagem_unica:
                print(f"Estado do semáforo: {resultado['estado']}")
            if cv2.waitKey(0 if imagem_unica else 25) & 0xFF in (ord("q"), 27):
                break
    except ValueError as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 1
    finally:
        salvar_resultados(registros, args.saida)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(executar(ler_argumentos()))
