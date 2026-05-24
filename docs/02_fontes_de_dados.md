# Fontes de Dados

## Google Play Store

- Método principal: `google-play-scraper`
- Tipo: biblioteca Python
- Motivação: coleta pública de avaliações do aplicativo do Nubank sem assumir acesso à API oficial do Google Play Developer.

## Reclame Aqui

- Método principal: scraping com `BeautifulSoup`
- Método alternativo: `Selenium`
- Motivação: páginas públicas de reclamações, com possível dependência de conteúdo dinâmico.

## Consumidor.gov.br

- Método principal: ingestão de arquivo público/dados abertos
- Método alternativo: `Selenium` para automatizar download
- Motivação: priorizar fonte pública estruturada antes de scraping.

## Limitações

- Mudanças estruturais dos sites podem exigir manutenção do código.
- Alguns downloads podem precisar de etapa manual documentada.
- O projeto não assume APIs privadas ou corporativas como disponíveis.
