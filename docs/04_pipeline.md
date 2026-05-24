# Pipeline

1. Coleta ou ingestão dos dados brutos em `data/raw/`
2. Padronização local e unificação em `data/bronze/`
3. Limpeza textual e geração da camada silver
4. Carga da silver no SQL Server
5. Treinamento e inferência de sentimentos
6. Extração de tópicos
7. Carga da gold no SQL Server
8. Consumo analítico no dashboard Streamlit
