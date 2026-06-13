"""Etapa 2: ampliação, iluminação e redução de ruído da ROI."""

import cv2


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
