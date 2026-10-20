# SRAG — MLP e Deep Learning

Projeto acadêmico de classificação binária para estudar o uso de redes neurais do tipo Multilayer Perceptron (MLP) na previsão do desfecho de casos hospitalizados de Síndrome Respiratória Aguda Grave (SRAG). A variável-alvo diferencia cura e óbito associado à SRAG, utilizando apenas informações consideradas disponíveis no momento da notificação/internação. **Este projeto é experimental e não é uma ferramenta clínica validada.**

## Integrantes

- Juliano Costa
- Victor Hugo Jesus

## Dados e protocolo experimental

- **Fonte:** SIVEP-Gripe / Ministério da Saúde, via OpenDataSUS.
- **Portal do conjunto de dados:** <https://dadosabertos.saude.gov.br/dataset/srag-2019-a-2026>
- **Portal Brasileiro de Dados Abertos:** <https://dados.gov.br/dados/conjuntos-dados/srag-2021-e-2022>
- **Partição temporal:** treinamento em 2020–2021, validação em 2022 e teste em 2023.
- **Atributos de entrada:** 188 após o pré-processamento, conforme o resumo salvo dos experimentos.

O notebook carrega os arquivos Parquet diretamente do servidor do Ministério da Saúde, se eles ainda não existirem no diretório de dados local. Por esse motivo, os arquivos Parquet não foram incluídos neste repositório: alguns ultrapassam o limite de upload via navegador do GitHub. A execução completa pode exigir bastante memória, espaço em disco e tempo; o notebook oferece a opção `RAPIDO = True` para um teste menor.

## Metodologia

1. Carregamento dos dados anuais e seleção dos registros com desfecho válido e hospitalização.
2. Limpeza e padronização dos campos, tratamento de datas inválidas, transformação da idade e criação de variáveis de atraso.
3. Tratamento dos valores ausentes e preparação das variáveis categóricas e numéricas.
4. Exclusão padrão de variáveis disponíveis somente durante ou após a internação, para reduzir vazamento do alvo.
5. Comparação de largura/profundidade, estratégias de regularização e diferentes configurações de MLP.
6. Busca de hiperparâmetros com Optuna, seleção de modelos finalistas e avaliação temporal no teste de 2023.

## Busca de hiperparâmetros (Optuna)

A busca registrada contém **40 tentativas: 17 completas e 23 podadas**. O objetivo foi maximizar a PR-AUC (Average Precision) de validação. O estudo usa amostrador TPE com semente 42 e podador MedianPruner para interromper tentativas que não demonstram progresso suficiente.

| Parâmetro | Espaço de busca registrado |
|---|---|
| Taxa de aprendizado (`lr`) | 1e-4 a 1e-2, escala logarítmica |
| Número de camadas ocultas | 1 a 5 |
| Largura-base | 32, 64, 128, 256, 512 ou 1024 neurônios |
| Formato da rede | constante ou funil |
| Ativação | ReLU, GELU ou SiLU |
| Dropout | 0 a 0,5 |
| `weight_decay` | 1e-6 a 1e-1, escala logarítmica |
| Batch normalization | habilitada ou desabilitada |
| Tamanho do lote | 256, 512, 1024, 2048 ou 4096 |
| Balanceamento de classes | ligado ou desligado |

A melhor configuração da busca foi registrada como **MLP-Optuna-Top1 (#26)**: 5 camadas em formato funil, largura-base 512, GELU, dropout aproximado de 0,2623, `lr` aproximado de 0,002184, `weight_decay` aproximado de 0,01505, lote de 512, sem batch normalization e com balanceamento de classes. A PR-AUC de validação registrada durante a busca foi aproximadamente **0,4893**. O resultado pode diferir da avaliação posterior dos finalistas, pois essa etapa possui protocolo de retreino e avaliação próprios.

## Comparação final de modelos

Métricas médias registradas no arquivo `tabela_comparativa_finalistas.csv` para o conjunto de teste (2023):

| Modelo | PR-AUC teste | ROC-AUC teste | F1 macro teste |
|---|---:|---:|---:|
| Regressão logística | 0,3699 | 0,8580 | 0,6693 |
| MLP base (2 × 128) | 0,3931 | 0,8675 | 0,6807 |
| MLP-Optuna-Top1 (#26) | **0,4003** | **0,8689** | **0,6823** |
| MLP-Compacta (#28) | 0,3951 | 0,8673 | 0,6804 |
| MLP-Alta capacidade (#20) | 0,3923 | 0,8669 | 0,6790 |

Os resultados estão sujeitos à mudança temporal da distribuição dos dados e ao desbalanceamento das classes. A PR-AUC é particularmente relevante para avaliar a classe minoritária, e a interpretação deve considerar também precisão, recall, especificidade e calibração. Os arquivos de resultados fornecem métricas adicionais e gráficos.

## Estrutura do repositório

- `docs/`: relatório técnico em DOCX e PDF e apresentação em PPTX e PDF.
- `notebooks/`: notebook atualizado e notebook com resultados de execução.
- `experimentos/`: registros da busca do Optuna, configurações, tabelas CSV, resumos JSON e gráficos.
- `requirements.txt`: dependências Python principais.
- `.gitignore`: evita incluir dados brutos e arquivos temporários.

## Como executar no Google Colab

1. Abra `notebooks/SRAG_MLP_DeepLearning_atualizado.ipynb` no GitHub.
2. Selecione a opção para abrir o notebook no Google Colab (ou carregue o arquivo `.ipynb` diretamente no Colab).
3. Em **Ambiente de execução → Alterar o tipo de ambiente de execução**, selecione GPU quando disponível.
4. Execute as células na ordem. O notebook instala Optuna e baixa os arquivos de dados necessários.
5. Para um teste rápido, defina `RAPIDO = True` na seção de configuração.

O notebook gera gráficos, CSVs, JSON e o banco SQLite do Optuna no diretório de resultados. O notebook completo pode consumir recursos consideráveis; os artefatos já incluídos em `experimentos/` documentam a execução que fundamenta o relatório.

## Dependências

As principais bibliotecas utilizadas são PyTorch, scikit-learn, pandas, NumPy, PyArrow, Matplotlib, Seaborn e Optuna. Veja `requirements.txt`. No Google Colab, PyTorch e várias bibliotecas já podem estar instaladas; o notebook instala Optuna quando necessário.

## Arquivos principais

- [Relatório técnico (PDF)](docs/Relatorio_SRAG_MLP.pdf)
- [Relatório editável (DOCX)](docs/Relatorio_SRAG_MLP.docx)
- [Apresentação (PPTX)](docs/Apresentacao_SRAG_MLP.pptx)
- [Resultados da busca Optuna](experimentos/exp3_optuna_tentativas.csv)
- [Configuração selecionada](experimentos/finalistas_config.json)
- [Resumo dos resultados](experimentos/resumo_resultados.json)
- [Tabela comparativa final](experimentos/tabela_comparativa_finalistas.csv)
