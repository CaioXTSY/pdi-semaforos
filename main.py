"""Entrada, processamento e detecção do estado do semáforo."""

import sys

import cv2

from deteccao import classificar_estado, mostrar_resultado
from entrada import ler_argumentos, ler_quadros, redimensionar, selecionar_roi
from processamento import preprocessar
from segmentacao import segmentar


def processar_quadro(frame, roi, args):
    frame = redimensionar(frame)
    recorte, roi = selecionar_roi(frame, roi)
    imagem_processada = preprocessar(recorte, args.filtro, args.ampliacao)
    mascaras, mascaras_limpas = segmentar(
        imagem_processada, args.kernel, args.iteracoes
    )
    resultado = classificar_estado(mascaras_limpas)
    mostrar_resultado(
        frame,
        recorte,
        imagem_processada,
        mascaras,
        mascaras_limpas,
        roi,
        resultado,
    )
    return roi, resultado


def executar(args):
    roi = args.roi
    try:
        for frame, imagem_unica in ler_quadros(args.entrada):
            roi, resultado = processar_quadro(frame, roi, args)
            if imagem_unica:
                print(f"Estado do semáforo: {resultado['estado']}")
            if cv2.waitKey(0 if imagem_unica else 25) & 0xFF in (ord("q"), 27):
                break
    except ValueError as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(executar(ler_argumentos()))
