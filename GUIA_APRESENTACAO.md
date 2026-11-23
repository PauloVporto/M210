# Guia de apresentação ao professor

## Abertura

“Desenvolvi uma aplicação que resolve problemas de programação linear e registra o procedimento em quadros. Quando muda o lado direito de uma restrição, o programa reaproveita o quadro ótimo para mostrar as mesmas condições e contas de preço-sombra usadas na resolução manual.”

Python foi escolhido pela legibilidade do código e pelo suporte do NumPy a vetores e matrizes. O Streamlit permite montar formulários, tabelas e histórico no próprio Python. pandas organiza as tabelas e ReportLab compõe os relatórios. A biblioteca padrão fornece as dataclasses que organizam o modelo e o quadro e os testes com `unittest`. As operações de pivoteamento são implementadas no próprio projeto.

## Demonstração de cerca de oito minutos

| Tempo | O que fazer | O que explicar |
| --- | --- | --- |
| 0:00–0:50 | Abra **Entrada do PPL** e selecione maximização, duas variáveis e duas restrições. | A função objetivo mede o lucro e os LDs representam as disponibilidades. |
| 0:50–2:30 | Digite `Z = 40x1 + 30x2`, `2x1 + x2 ≤ 100` e `x1 + x2 ≤ 80`. Resolva e abra a explicação nos quadros 2 e 3. | As folgas formam a base inicial. X1 entra primeiro e F1 sai pela menor razão, 50. Em seguida X2 entra e F2 sai pela menor razão, 60. A linha Z mostra `Zj − Cj`. |
| 2:30–3:40 | Mostre o ótimo `x1=20`, `x2=60`, `Z=2600` e abra **Análise de sensibilidade**. | A linha Z nas colunas F1 e F2 dá os preços 10 e 20. As condições são `20 + Δ1 − Δ2 ≥ 0` e `60 − Δ1 + 2Δ2 ≥ 0`. |
| 3:40–5:00 | Selecione R1, escolha **Aumentar**, **%**, digite **10**, confira a prévia e aplique. | A prévia não altera a solução. Após aplicar, as condições dão 30 e 50, e `ZNOVO = 2600 + 10(10) = 2700`. A aplicação atualiza os valores do quadro por multiplicação matricial. |
| 5:00–6:00 | Aumente R2 em 10%, depois diminua R1 em 5 unidades. | Os LDs passam a `(110,88)` e `(105,88)`. Os lucros são 2860 e 2810. Cada mudança testa a base vigente e parte dos LDs confirmados. |
| 6:00–7:15 | Use **Restaurar problema original**. Em R1, defina LD final **1000**. Abra a explicação no quadro ótimo do Dual. | O básico x2 ensaiado fica −840. O preço-sombra anterior deixa de valer para a proposta. O Dual troca x2 por F1 e obtém `(80,0)`, lucro 3200 e novos preços 0 e 40. |
| 7:15–8:00 | Abra o **Histórico**, mostre uma mudança anterior e prepare o PDF completo. | A restauração reativa um quadro já armazenado. O PDF usa as mesmas contas e a mesma precisão selecionada na interface. |

## Perguntas e respostas

**Por que um preço-sombra vale apenas em uma região?**

A relação `yᵀ = CbᵀB⁻¹` depende da base. O preço mede o efeito de mudar um LD enquanto a mesma base fornece uma solução viável. `change_rhs` verifica isso antes de usar o preço.

**Como o programa sabe que a base continua ótima?**

Mudar apenas b preserva a matriz, os custos e os custos reduzidos. `change_rhs` testa `xB + B⁻¹Δb`. Se todos os básicos continuam não negativos e as linhas dependentes continuam em zero, o tableau ainda satisfaz as condições de ótimo.

**Por que duas mudanças juntas precisam de uma verificação própria?**

As faixas individuais assumem que os outros LDs ficam fixos. Duas mudanças podem somar contribuições negativas na mesma linha e ultrapassar a margem. `change_rhs` testa o vetor completo.

**O programa faz outra inversão ou resolve o Simplex de novo?**

`Tableau.pivot` transporta uma matriz auxiliar junto com os pivôs desde o primeiro quadro. Ela já é `B⁻¹` no ótimo. A mudança viável reutiliza essa matriz e atualiza os valores básicos, sem executar pivôs. O teste automatizado intercepta chamadas ao solver para confirmar isso.

**O que acontece se a base ficar inviável?**

`dual_reoptimize` começa no tableau armazenado, com os LDs básicos recalculados. A variável básica negativa sai e a coluna de entrada passa pelo teste da menor razão dual. Se nenhuma coluna puder corrigir a linha, o problema modificado é inviável e o último ótimo confirmado permanece disponível.

**Por que a resolução pode escolher pivôs diferentes da apostila?**

O modo padrão reproduz as escolhas observadas na NP2. A regra de Bland fica disponível como opção e como proteção contra repetição de bases. Em um problema degenerado, ela pode produzir outra base ótima com preços-sombra diferentes. A sensibilidade sempre corresponde à base exibida.

**Como são tratados minimização e sinais diferentes?**

O solver maximiza o negativo da função na minimização e reescala Z e os preços na saída. O sinal original da coluna de folga ou excesso entra na interpretação da linha Z. Uma igualdade sem folga usa `CbᵀB⁻¹`. As artificiais são tratadas na Fase 1.

**Quais são as limitações?**

O modelo aceita de 2 a 4 variáveis contínuas e não negativas. A análise disponível é de mudanças no LD. Coeficientes de escalas extremas podem exigir reescala. Os dados permanecem na sessão ativa e o relatório permite guardar os resultados fora dela.
