<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.md">English</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mcp-tool-shop-org/brand/main/logos/understanding-dataset-lab/readme.png" alt="understanding-dataset-lab" width="400">
</p>

<p align="center">
  <b>empathy</b>: "Identification with or understanding of the thoughts, feelings, or emotional state of another person."<br>
  <sub>Definition from <a href="https://en.wiktionary.org/wiki/empathy">Wiktionary</a>, <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>.</sub><br><br>
  Cognitive empathy, as udlab means it: an accurate model of what each party in a situation knows, believes, wants and needs.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-prototype-orange" alt="Prototype">
  <a href="https://github.com/mcp-tool-shop-org/understanding-dataset-lab/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/understanding-dataset-lab/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/understanding-dataset-lab/"><img src="https://img.shields.io/badge/docs-landing_page-blue" alt="Landing page"></a>
</p>

> **Protótipo.** udlab é uma pesquisa em andamento, de código aberto. O método e o código estão disponíveis para leitura e execução. Seus resultados até o momento são verificações de quão bem ele gera cenas, em bancos de dados de desenvolvimento. Não possui resultados finais ou definitivos, nem um modelo treinado, e não faz nenhuma afirmação sobre se alguma IA compreende alguém. As etapas ainda podem mudar a forma como as cenas são criadas.

**understanding-dataset-lab (udlab) cria cenários de teste para empatia cognitiva: se um modelo consegue manter uma imagem precisa do que cada participante em uma cena sabe, acredita, deseja e precisa.** Cenas curtas com várias pessoas são planejadas por meio de código e, em seguida, transformadas em prosa por um modelo de escrita local. A resposta para cada pergunta sobre uma cena vem do plano do código, nunca de um modelo.

É o complemento de [reasoning-dataset-lab](https://github.com/mcp-tool-shop-org/reasoning-dataset-lab), que faz o mesmo para quebra-cabeças e itens de verificação.

## Por que cenas com elementos criados por código?

Em uma cena planejada, a resposta correta segue mecanicamente dos eventos: quem se moveu para onde, quem podia ver, quem disse o quê e quem ouviu. Alguém que estava fora do alcance visual quando um objeto foi movido ainda acredita que ele está onde foi visto pela última vez. Como essas regras estão no código, a chave é calculada, não julgada.

Uma *chave criada por código* significa:
- a resposta vem do plano do código para a cena;
- ela é selada com uma semente privada;
- ela nunca é escrita ou avaliada por um modelo.

Cada versão em prosa é verificada pelo código em relação ao plano antes de ser considerada válida.

## Status: o que foi testado

Os resultados são contagens em bancos de dados de desenvolvimento de 160 versões. Os resultados negativos vêm primeiro, porque moldaram o projeto.

| Etapa | O que foi testado | Resultado | Adotado? |
|---|---|---|---|
| 3 | Uma lista de formas de frases aceitas no prompt do escritor | 111/160 versões foram aprovadas, em comparação com 127/160 para o prompt de referência. O escritor não usou as formas listadas. | **Não**, pela regra pré-registrada |
| 4 | Formas de frases atribuídas por código | 89/160, em comparação com 120/160 para o prompt de referência. O teste de desenvolvimento falhou, portanto, não houve execução confirmatória. 16 versões por grupo (10%) nunca foram geradas, porque o prompt mais o orçamento de saída excederam o limite de contexto do escritor. | **No** |
| 5 | Movimentos de código e discurso reescrito (revisão 2) | A execução de desenvolvimento passou por todos os 7 portões pré-registrados. Primeiros rascunhos: 126/160, em comparação com um mínimo de 112; 132/160 após o pipeline. As trocas de código mantiveram 2,501/2,501. As reescritas de discurso mantiveram 350/358. 0 versões excederam o limite de contexto. | Passou pelos portões de desenvolvimento |

**Ainda não demonstrado:**
- nenhuma execução confirmatória selada da Etapa 5;
- nenhum modelo treinado com dados do udlab;
- nada ainda sobre se algum modelo compreende as pessoas em uma cena.

## O que está aqui

- **`udlab-gen`**, o gerador. Ele planeja cenários, executa o pipeline da Etapa 5, sela as chaves de cada conjunto com uma semente privada e verifica um conjunto escrito renderizando-o novamente e comparando-o.
- **Sondas de atalho**, executadas fora do loop. Elas testam se as características superficiais de uma cena revelam a resposta.
- **O prompt do escritor** (`prompts/`). O escritor é `qwen3.6:35b`, executado localmente por meio do Ollama.
- **Um lançador do Colab** (`scripts/colab/`) para executar o escritor em uma GPU alugada.
- **753 testes.**

## Tente sem uma GPU

O escritor padrão é um simulacro de CPU, portanto, todo o pipeline é executado em qualquer máquina:

```bash
git clone https://github.com/mcp-tool-shop-org/understanding-dataset-lab
cd understanding-dataset-lab
uv run udlab-gen --count 3 --out-dir artifacts/demo
uv run udlab-gen --probe-only artifacts/demo/sealed_store/keys.sealed.jsonl
uv run pytest
```

- Python 3.15 ou mais recente; `uv` o instala, se necessário.
- O primeiro comando grava uma parte pública (`part_public/`) e um armazenamento selado (`sealed_store/`) em `artifacts/demo/`, uma pasta que o Git ignora.
- `uv run udlab-gen --help` lista todas as opções.

## O que permanece privado

Isso permanece privado, porque publicá-los revelaria as respostas:
- o banco de dados de testes selado e os bancos de sementes de desenvolvimento;
- cada armazenamento selado e registro de geração;
- os registros de execução com suas sementes de plano.

Nenhum deles está neste repositório ou em seu histórico. Os testes usam apenas sementes sintéticas e cenas inventadas.

## Segurança e confiança

- **Dados acessados:** arquivos na `--out-dir` que você escolher e o prompt do escritor. Com `--writer ollama`, também chama um servidor local do Ollama.
- **Dados não acessados:** nada fora da pasta de saída. Não há telemetria e nenhuma chamada de rede, exceto para o servidor local do Ollama e, se você o executar, o lançador do Colab.
- **Permissões:** nenhuma além de ler e gravar na pasta de saída. No modo GPU, ele interrompe o servidor Ollama que iniciou se a memória de vídeo exceder `--vram-halt-mib`.
- Relate problemas de segurança conforme descrito em [SECURITY.md](SECURITY.md).

## Licença

O código e o texto do próprio repositório estão sob a [licença MIT](LICENSE). A definição citada sob o logotipo é do Wiktionary e permanece sob [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Não é usado em nenhum conjunto de dados ou prompt.
