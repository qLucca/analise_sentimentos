# Dicionário de Dados

## Bronze unificado

- `id_registro`: identificador único do registro na origem ou gerado no pipeline
- `fonte`: origem do dado (`google_play`, `reclame_aqui`, `consumidor_gov`)
- `data_publicacao`: data do registro
- `texto_original`: texto bruto relevante para análise
- `nota`: nota, avaliação ou score quando disponível
- `status_reclamacao`: situação da reclamação
- `categoria_problema`: categoria pública do problema
- `uf`: unidade federativa, quando disponível
- `versao_app`: versão do aplicativo, quando disponível
- `sentimento_real`: rótulo conhecido ou derivado

## Silver

- `texto_limpo`: texto após limpeza e normalização
- `data_processamento`: timestamp do pipeline

## Gold

- `sentimento_previsto`: classe prevista pelo modelo
- `topico`: tópico associado ao registro
