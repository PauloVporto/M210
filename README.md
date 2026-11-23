# Simplex Tableau com análise de sensibilidade

Aplicação Streamlit para problemas de maximização e minimização com 2, 3 ou 4 variáveis de decisão não negativas, restrições ≤, ≥ e = e alterações sucessivas do lado direito (LD).

## Executar

Com Python 3.10 ou mais recente, execute na raiz do projeto:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Para rodar os testes:

```sh
python -m unittest discover -s tests -v
```

## Exemplo completo

A aba **Entrada do PPL** vem preenchida com a maximização de `Z = 3x1 + 2x2`, sujeita a `x1 + x2 ≤ 4`, `x1 ≤ 2` e `x2 ≤ 3`. Clique em **Resolver PPL**. A solução original é `(2, 2)`, com `Z = 10`, preços-sombra `(2, 1, 0)` e faixas individuais de LD `[2, 5]`, `[1, 4]` e `[2, +∞)`.

Na aba **Alterações do LD**, aplique cada linha abaixo em sequência com **Calcular novo Z**.

| Alteração | Vetor de LD após alteração | Ponto ótimo | Z | Método |
| --- | --- | --- | --- | --- |
| R1 passa a 4,5 | (4,5 ; 2 ; 3) | (2 ; 2,5) | 11 | Preço-sombra |
| R2 passa a 2,5 | (4,5 ; 2,5 ; 3) | (2,5 ; 2) | 11,5 | Preço-sombra |
| R3 passa a 1 | (4,5 ; 2,5 ; 1) | (2,5 ; 1) | 9,5 | Simplex Dual |

A primeira mudança dá `10 + 2 × 0,5 = 11`. A segunda parte dos LDs vigentes e dá `11 + 1 × 0,5 = 11,5`. A terceira deixa a antiga folga de R3 negativa. O Simplex Dual parte do quadro anterior com o LD atualizado, encontra o novo ótimo e recalcula os preços-sombra. A aba **Histórico** permite consultar as contas de cada tentativa.

## Alterar recursos

Em **Alterações do LD**, selecione os recursos e escolha uma operação por restrição. **Aumentar** e **Diminuir** aceitam porcentagem, unidades e multiplicadores. **Dobrar**, **Multiplicar**, **Definir um novo LD diretamente** e **Restaurar LD original** ficam no mesmo menu. Para aplicar uma porcentagem a todos os recursos de uma vez, use **Alterar todos os recursos**.

Use **Atualizar prévia** para enviar os campos sem aplicar a alteração. **Calcular novo Z** processa os valores do clique, inclusive quando são diferentes da última prévia. Selecione as restrições antes de preencher o formulário.

Cada cálculo usa o LD vigente. O percentual incide sobre o valor com seu sinal, inclusive para LDs negativos. O histórico guarda o tipo, a quantidade solicitada e a variação relativa ao LD anterior quando ele é diferente de zero. **Restaurar problema original** recupera o quadro ótimo inicial. Em **Histórico**, **Restaurar estado selecionado** reativa o quadro e os LDs de uma alteração que teve ótimo. As restaurações também ficam registradas.

Para reproduzir o exemplo com `Z = 40x1 + 30x2`, cadastre `2x1 + x2 ≤ 100` e `x1 + x2 ≤ 80`. Aumente R1 em 20% e diminua R2 em 10% na mesma proposta. A aplicação calcula o novo ponto `(48, 24)` e exibe `ZNOVO = 2600 + 200 − 160 = 2640`.

## Exportar PDF

Na barra lateral, abra **Exportar resultados**, selecione **Resolução completa**, **Quadros Simplex**, **Análise de sensibilidade** ou **Histórico das alterações**. Clique em **Preparar PDF** e depois em **Baixar**. Os relatórios usam os quadros e as alterações armazenados na sessão. O PDF de sensibilidade acompanha a referência selecionada nessa aba. A resolução completa inclui o modelo, a sequência de pivôs, o quadro vigente, os preços-sombra, as condições de viabilidade e as contas de cada proposta.

## Reutilização da base

A sessão guarda o modelo e o quadro ótimo originais, a solução vigente e o histórico. As operações de linha também transformam uma matriz identidade com os sinais da normalização inicial. Ao final, essa matriz é a inversa da base expressa nas coordenadas dos LDs informados pelo usuário.

Para cada proposta, o programa calcula `Δb = b_proposto − b_atual` e `xB_proposto = xB_atual + B⁻¹Δb`. Se o vetor básico preserva a viabilidade, calcula `Z_novo = Z_atual + yᵀΔb` com `yᵀ = CbᵀB⁻¹`, diretamente e sem pivôs. Os intervalos individuais mantêm fixos os demais LDs. A verificação matricial cobre mudanças simultâneas.

Quando a base perde a viabilidade, o Simplex Dual tenta recuperar uma base ótima. Uma proposta inviável fica registrada no histórico e a última solução ótima segue ativa. Em restrições linearmente dependentes, as artificiais remanescentes ficam fixas em zero e seus LDs devem conservar a dependência.

## Convenção dos quadros

A apresentação segue a resolução da NP2. O cabeçalho começa com **VB** e termina com **LD**. A primeira linha é **Z**, com `Zj − Cj` e o valor da função objetivo no LD. O preço-sombra aparece junto à coluna da qual foi lido. O sinal é ajustado para colunas originalmente de excesso. Em igualdades, ele vem da transformação da linha Z, pois a restrição não tem coluna de folga.

As abas de sensibilidade e alteração mostram as expressões em Δ, a substituição numérica e a verificação de cada linha. As matrizes e os custos ficam disponíveis em **Detalhes matemáticos** e na barra lateral, respectivamente. A Fase 1 maximiza o negativo da soma das artificiais. Na Fase 2 de problemas de minimização, a linha Z é exibida na escala da função objetivo original.

O critério padrão **Modelo do professor** escolhe o coeficiente mais negativo da linha Z interna e a menor razão LD/coeficiente positivo. Em empate nas razões, usa a primeira linha do quadro, como no exercício 1 da NP2. A regra de Bland continua disponível e é usada como proteção se uma base se repetir.

- **NP2, exercício 1.** Os quadros terminam em Z = 17.500 e preços-sombra (40, 0, 15). No item C, a última condição dá −250 e reprova a base anterior. Em seguida, o Dual determina 18.500 para o problema modificado. No item D, partindo do quadro original, as condições dão (225, 315, 0) e o lucro é 15.750.
- **NP2, exercício 2.** Entra X2 e depois X1. O ponto ótimo é (10/3, 1/3), com Z = 80 e preços-sombra (0, 0, 2,5). No item C, a base se mantém e o lucro passa a 64.

A aplicação mantém os rótulos das colunas durante os pivôs e calcula com a precisão interna, inclusive onde a apostila repete rótulos ou mostra coeficientes arredondados.

A barra lateral controla de 0 a 8 casas decimais para a interface e PDFs. Essa formatação remove resíduos visuais próximos de zero e preserva os números usados no solver.
