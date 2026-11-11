# Predição da tensão de células a combustível PEM com aprendizado de máquina

Código do Trabalho de Conclusão de Curso de Eduardo Gonzaga (Engenharia Mecânica, UFOP), orientado pela DSc. Elisângela Martins Leal.

O script `tcc_pipeline.py` compara Baseline, Regressão Linear, Random Forest, Gradient Boosting, regressão por processos gaussianos (GPR), MLP e um modelo semiempírico da curva de polarização, sob quatro esquemas de validação cruzada: GroupKFold por curva, Leave-One-Protocol-Out e exclusão de um nível de pressão ou de umidade relativa por vez.

## Dados
Base ECSIM de acesso aberto (Hamidi, Haghighi e Askari, 2020): https://github.com/ECSIM/pem-dataset1. O script clona a base automaticamente se ela não estiver no diretório de trabalho.

## Como executar
1. Abra um notebook no Google Colab e cole o conteúdo de `tcc_pipeline.py` em uma célula.
2. Execute. O tempo total é de cerca de uma hora, dominado pelo MLP.
3. Os resultados são gravados em `resultados_tcc_final/` e compactados em `resultados_tcc_final.zip`.

## Reprodutibilidade
Semente global 42; MLP com sementes 0 a 4. A divisão do GroupKFold por curva usa um algoritmo determinístico implementado no código, independente da versão da scikit-learn. O script confere os resultados com valores de referência e informa divergências ao final.
