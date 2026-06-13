"""Funções de recorte e pré-processamento da região do semáforo."""

import cv2
import numpy as np

# Limites HSV do OpenCV: H entre 0-179; S e V entre 0-255.
LIMITES_HSV = {
    "vermelho": [
        ((0, 80, 60), (12, 255, 255)),
        ((168, 80, 60), (179, 255, 255)),
    ],
    "amarelo": [((15, 80, 70), (38, 255, 255))],
    "verde": [((38, 60, 50), (90, 255, 255))],
}


def redimensionar(frame, largura=960):
    """Redimensiona o quadro mantendo sua proporção."""
    if frame.shape[1] <= largura:
        return frame

    escala = largura / frame.shape[1]
    return cv2.resize(frame, None, fx=escala, fy=escala)


def selecionar_roi(frame, coordenadas=None):
    """Seleciona uma ROI pelo mouse ou usa (x, y, largura, altura)."""
    if coordenadas is None:
        coordenadas = cv2.selectROI("Selecione o semaforo", frame, False)
        cv2.destroyWindow("Selecione o semaforo")

    x, y, largura, altura = map(int, coordenadas)
    if largura == 0 or altura == 0:
        raise ValueError("Região de interesse inválida.")

    return frame[y : y + altura, x : x + largura], (x, y, largura, altura)


def filtro_gaussiano(imagem):
    """Suaviza ruídos gerais usando uma média ponderada."""
    return cv2.GaussianBlur(imagem, (5, 5), 0)


def filtro_mediana(imagem):
    """Reduz ruídos pontuais preservando melhor as bordas."""
    return cv2.medianBlur(imagem, 5)


def melhorar_brilho(imagem):
    """Melhora o contraste local do canal de brilho usando CLAHE."""
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.createCLAHE(2.0, (8, 8)).apply(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def preprocessar(imagem, filtro="mediana", ampliacao=4):
    """Amplia a ROI, melhora o brilho e reduz pequenos ruídos."""
    if ampliacao <= 0:
        raise ValueError("A ampliação deve ser maior que zero.")

    imagem = cv2.resize(
        imagem,
        None,
        fx=ampliacao,
        fy=ampliacao,
        interpolation=cv2.INTER_CUBIC,
    )
    imagem = melhorar_brilho(imagem)

    if filtro == "gaussiano":
        return cv2.GaussianBlur(imagem, (3, 3), 0)
    if filtro == "mediana":
        return cv2.medianBlur(imagem, 3)

    raise ValueError("Filtro deve ser 'gaussiano' ou 'mediana'.")


def converter_hsv(imagem):
    """Converte uma imagem BGR para o espaço de cores HSV."""
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)


def criar_mascaras(imagem_hsv, limites=None):
    """Cria máscaras binárias para vermelho, amarelo e verde."""
    limites = limites or LIMITES_HSV
    mascaras = {}

    for cor, faixas in limites.items():
        mascara = np.zeros(imagem_hsv.shape[:2], dtype=np.uint8)
        for minimo, maximo in faixas:
            faixa = cv2.inRange(
                imagem_hsv,
                np.array(minimo, dtype=np.uint8),
                np.array(maximo, dtype=np.uint8),
            )
            mascara = cv2.bitwise_or(mascara, faixa)
        mascaras[cor] = mascara

    return mascaras


def aplicar_regioes_esperadas(mascaras):
    """Mantém cada cor somente no terço vertical esperado do semáforo."""
    altura = next(iter(mascaras.values())).shape[0]
    cortes = {
        "vermelho": (0, altura // 3),
        "amarelo": (altura // 3, 2 * altura // 3),
        "verde": (2 * altura // 3, altura),
    }
    resultado = {}

    for cor, mascara in mascaras.items():
        filtrada = np.zeros_like(mascara)
        inicio, fim = cortes[cor]
        filtrada[inicio:fim] = mascara[inicio:fim]
        resultado[cor] = filtrada

    return resultado


def criar_kernel(tamanho=5):
    """Cria um elemento estruturante elíptico."""
    if tamanho <= 0:
        raise ValueError("O tamanho do kernel deve ser maior que zero.")
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamanho, tamanho))


def erodir(mascara, kernel, iteracoes=1):
    return cv2.erode(mascara, kernel, iterations=iteracoes)


def dilatar(mascara, kernel, iteracoes=1):
    return cv2.dilate(mascara, kernel, iterations=iteracoes)


def abertura(mascara, kernel, iteracoes=1):
    return cv2.morphologyEx(
        mascara, cv2.MORPH_OPEN, kernel, iterations=iteracoes
    )


def fechamento(mascara, kernel, iteracoes=1):
    return cv2.morphologyEx(
        mascara, cv2.MORPH_CLOSE, kernel, iterations=iteracoes
    )


def limpar_mascara(mascara, tamanho_kernel=3, iteracoes=1):
    """Remove pontos isolados e preenche pequenas falhas."""
    if iteracoes <= 0:
        raise ValueError("A quantidade de iterações deve ser maior que zero.")

    kernel = criar_kernel(tamanho_kernel)
    mascara = fechamento(mascara, kernel, iteracoes)
    return abertura(mascara, kernel, iteracoes)


def limpar_mascaras(mascaras, tamanho_kernel=3, iteracoes=1):
    """Aplica a limpeza morfológica a todas as máscaras."""
    return {
        cor: limpar_mascara(mascara, tamanho_kernel, iteracoes)
        for cor, mascara in mascaras.items()
    }
