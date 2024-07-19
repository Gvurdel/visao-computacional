# Detector de Olhos Abertos/Fechados em Vídeo

Autor: Gustavo Menezes Vurdel

## Definir Projeto
Este projeto utiliza técnicas de visão computacional para detectar se uma pessoa está com os olhos abertos ou fechados em um vídeo. Usando OpenCV e dlib, o script processa cada frame do vídeo para identificar rostos e verificar o estado dos olhos, exibindo a detecção em tempo real.

## Funcionalidades
- Detecção de rostos em vídeo.
- Verificação do estado dos olhos (abertos ou fechados) usando a relação de aspecto dos olhos (EAR).
- Exibição de caixas ao redor dos rostos com cores indicativas:
- Verde: Olhos abertos
- Vermelho: Olhos fechados
- Exibição de texto indicando o estado dos olhos ("Olhos Abertos" ou "Olhos Fechados").

## Modelo Pré-Treinado
O projeto utiliza um modelo pré-treinado para a detecção de pontos faciais, fornecido pelo dlib:
- `shape_predictor_68_face_landmarks.dat`: Este modelo é usado para identificar 68 pontos faciais, essenciais para a análise da posição dos olhos.

## Executando o Projeto
- Criar um ambiente virtual:
- `python3.8 -m venv myenv`

- Ativar o ambiente virtual:
- `source myenv/bin/activate` macOS

- Com o ambiente virtual ativado, instale as bibliotecas necessárias:
- `pip install opencv-python-headless dlib scipy`

- Baixe o Modelo Pré-Treinado:
- `curl -O http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2`
- `bzip2 -d shape_predictor_68_face_landmarks.dat.bz2`

- Execute o script:
- `python detector_eyes.py`

