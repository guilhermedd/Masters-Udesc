# R²C-AD em Python

Protótipo executável do núcleo da seção 25. A pergunta que ele permite investigar é se selecionar arestas por ganho preditivo em blocos melhora a detecção quando parte do treino está contaminada.

## Executar no Windows

Extraia o ZIP e abra o PowerShell na pasta que contém este README. Use Python 3.11 ou superior com NumPy e PyTorch. Os comandos usam o interpretador do ambiente virtual diretamente.

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m r2c demo --out exemplo
.\.venv\Scripts\python -m r2c treinar --csv exemplo/treino.csv --model modelo
.\.venv\Scripts\python -m r2c detectar --csv exemplo/teste.csv --model modelo --out resultado.csv
.\.venv\Scripts\python -m r2c avaliar --result resultado.csv --labels exemplo/rotulos.csv
.\.venv\Scripts\python -m unittest discover -s tests
```

Em Linux ou macOS, crie o ambiente com `python3 -m venv .venv` e use `.venv/bin/python`. A execução padrão usa CPU. Para usar uma GPU compatível, passe `--device cuda` ao treino e à detecção.

## Seus dados

Use um CSV separado por vírgulas, com uma linha por instante em ordem cronológica, frequência regular e cabeçalho com os nomes dos sensores. Informe somente as medições. Guarde timestamps e rótulos separadamente. O índice do resultado começa em zero e corresponde à posição da linha de dados. Treino e inferência exigem as mesmas colunas na mesma ordem. Mantenha o teste cronologicamente posterior. Campos vazios, NA e NaN representam medições ausentes. O comando de avaliação aceita um CSV separado com coluna `anomalia` e valores 0 ou 1 para cada linha do teste.

```csv
pressao,vazao,temperatura
1.22,4.21,26.3
1.24,,26.4
```

O treino divide a série cronologicamente em ajuste neural (50%), seleção do grafo (25%), referência de ranks (12,5%) e calibração da política (restante). As janelas são construídas dentro de cada trecho e dentro de cada bloco do grafo. Cada bloco precisa ter mais linhas que a janela. Referência e calibração exigem pelo menos 20 instantes observados após o aquecimento. Para estudar taxas raras, reserve exposição suficiente para observar vários eventos de alarme. As primeiras linhas de cada nova execução de detecção servem de histórico e recebem scores vazios.

## O que o modelo aprende

1. O ajuste estima a mediana e a MAD por sensor. Quando a MAD é zero, usa escala unitária. O encoder recebe valores normalizados e máscaras de presença. O Patch Transformer compartilha parâmetros entre canais e processa somente o histórico anterior ao instante avaliado.
2. Uma cabeça prevê o valor atual a partir do passado do próprio sensor. Outra usa atenção sobre mensagens dos vizinhos, inclusive suas medições contemporâneas. A diagonal do grafo é bloqueada, e o valor atual do alvo fica fora de sua própria reconstrução. A perda de Huber e pesos por janela, limitados entre 0,15 e 1, reduzem a influência de erros grandes. Os pesos são uma heurística baseada na mediana e MAD do minibatch.
3. O CRGL estima, em blocos reservados, quanto cresce a perda de cada alvo quando a fonte é removida. O preditor aprende com fontes aleatoriamente ocultadas. Uma aresta requer ganho superior a 0,0001 na maioria dos blocos. Entre as elegíveis, a seleção usa ganho mediano positivo dividido pela MAD acrescida de 0,0001. Cada alvo conserva até o limite de vizinhos. A segunda fase reajusta a rede com a máscara fixa e pesos de atenção contextuais. O ganho mede utilidade preditiva condicionada aos outros sensores disponíveis.
4. Cada score usa a média dos três maiores resíduos absolutos observados, ou de todos os disponíveis se houver menos de três. A fusão combina o máximo e a média dos ranks e normaliza ambos pela referência empírica. A política aceita evidência instantânea forte, dois instantes moderados consecutivos ou evidência acumulada. Um instante totalmente ausente interrompe a acumulação.

## Calibração e avaliação

`--budget 5` significa até cinco **inícios de alarme por mil amostras observadas** na calibração. A grade percorre níveis de sensibilidade e avalia a política completa sobre a sequência. Se nenhuma política ativa atender ao orçamento, o modelo registra a desativação dos alertas. O tempo normal é desconhecido na calibração sem rótulos. No teste rotulado, o denominador do FAR inclui somente amostras normais com score disponível. Para converter a taxa para alarmes por dia, multiplique o FAR por mil pelo número de amostras normais observáveis por dia e divida por mil.

Um evento real é uma sequência contígua de rótulos 1. Ele é detectado quando pelo menos um alarme começa dentro do intervalo. Um alarme que começa antes do evento conta como falso se começar em tempo normal. O atraso é medido em amostras desde o início real até o primeiro início de alarme no intervalo. Eventos inteiramente fora da cobertura são contados separadamente. A coluna `ativo` marca a duração dos alertas e `inicio_alerta` marca os inícios usados na avaliação.

O exemplo usa dois intervalos contaminados no trecho de ajuste e três inversões de sinal no teste. Em uma execução de desenvolvimento em CPU com semente 42, houve um início de alarme em 493 amostras na calibração (2,03 por mil). No teste, houve detecção dos três eventos e oito falsos alarmes em 1.463 amostras normais (5,47 por mil), com atraso mediano de zero amostras. O orçamento foi excedido no teste. Esse cenário serviu ao desenvolvimento da política. A validação científica exige dados independentes, múltiplas sementes e orçamentos definidos previamente.

## Reproduzir uma ablação

### Comparar com métodos clássicos

Na pasta do projeto, com o ambiente Python ativado, execute

```powershell
python -m pip install -r requirements.txt
python comparar.py --treino exemplo/treino.csv --teste exemplo/teste.csv --rotulos exemplo/rotulos.csv --out comparacao
```

O comparador treina o R²C-AD e três referências. Isolation Forest usa 100 árvores. PCA pontua pelo erro quadrático de reconstrução, retendo componentes até 90% da variância e reservando pelo menos uma direção para o resíduo. Ambos recebem o vetor atual e o anterior. O baseline MAD usa o maior desvio absoluto entre sensores após normalização pela mediana e MAD de treino. Valores ausentes são imputados pela mediana nos métodos clássicos. O R²C-AD usa suas próprias máscaras e toda a janela.

Os métodos compartilham trecho de ajuste, referência de ranks, sequência de calibração, período de teste avaliado, política de alarmes e definição de evento. O trecho reservado para escolha do grafo é específico do R²C-AD. Use `--budget` para ajustar o orçamento antes de executar. Em `resumo.csv`, confira primeiro `politica_ativa` e `dentro_orcamento_teste`. Compare o recall dos modelos que atenderam ao orçamento e, em caso de empate, examine o FAR e o atraso. Um FAR baixo acompanhado de recall zero indica que o detector perdeu todos os eventos. Os CSVs por método permitem localizar cada início de alarme.

### Comparar variantes do grafo

Repita os comandos com os mesmos CSVs, semente e orçamento. Em `treinar`, use `--graph consensus` para CRGL, `--graph attention` para atenção densa entre sensores, ou `--graph none` para desativar as mensagens entre canais. Use uma pasta `--model` e um arquivo de resultados distintos para cada variante. Cada modelo passa pela sua própria calibração. Compare recall, atraso e FAR efetivamente observado. Consulte `python -m r2c treinar --help` para alterar janela, tamanho do patch, largura, profundidade, épocas por fase, blocos e vizinhos.

## Arquivos e escopo

| Arquivo | Função |
|---|---|
| r2c/data.py | CSV, normalização robusta e janelas cronológicas |
| r2c/model.py | Encoder de patches, atenção de grafo e cabeças preditivas |
| r2c/learning.py | Ajuste neural, estimação de ganhos, seleção do grafo e scores |
| r2c/alarms.py | Ranks, fusão, política sequencial e calibração |
| r2c/cli.py | Dados sintéticos, treino, inferência e avaliação |
| tests/test_detector.py | Testes de mascaramento do alvo, histórico e política de alarme |

O treino salva pesos, parâmetros de normalização, distribuições de referência e configuração. Para o CRGL, `edges.csv` registra ganho, dispersão, confiança e seleção de cada aresta dirigida. Os padrões de execução usam uma janela de 32 amostras e uma camada para facilitar testes em CPU. A inferência percorre um arquivo em lotes e respeita o histórico disponível em cada instante.

Esta edição cobre uma rodada de descoberta do grafo seguida de refinamento, dois scores e uma política sequencial simples. Aprendizado contrastivo com protótipos, scores latente e estrutural, adaptação online, serviço de streaming e a bateria completa de benchmarks são extensões da dissertação. A robustez depende, entre outras condições, da distribuição da contaminação entre blocos. O projeto permite testar essa hipótese experimentalmente.
