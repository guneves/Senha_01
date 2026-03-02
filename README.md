Gesture Login System (MediaPipe + Flask)
Um sistema de autenticação biométrica baseado em visão computacional que utiliza sequências de gestos (contagem de dedos) para liberar o acesso a uma área secreta.

Funcionalidades:

Reconhecimento em Tempo Real: Processamento fluido utilizando a MediaPipe Tasks API no modo vídeo.

Interface Web Moderna: Dashboard desenvolvido com Flask e design responsivo (Dark Mode).

Segurança por Sequência: O acesso só é liberado após a inserção de uma sequência exata de 3 gestos.

Feedback Visual: Barra de progresso para confirmação de gestos e tela de "Acesso Negado" com auto-reset.

Área Secreta: Exibição de conteúdo restrito após a validação bem-sucedida da senha.

Tecnologias Utilizadas:

Linguagem: Python 3.12.10

Visão Computacional: OpenCV & MediaPipe (Hand Landmarker)

Backend Web: Flask

Frontend: HTML5, CSS3 (Modern UI) e JavaScript (Fetch API para sincronização assíncrona)