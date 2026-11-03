<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.md">English</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

> **Prototipo.** udlab è un progetto di ricerca aperto in corso. Il metodo e il codice sono disponibili per essere consultati ed eseguiti. Finora, i risultati consistono in verifiche sull'efficacia con cui genera scene, utilizzando database di sviluppo. Non ci sono risultati definitivi o conclusivi, né modelli addestrati, e non si fa alcuna affermazione sul fatto che un'intelligenza artificiale possa comprendere qualcun altro. Le fasi possono ancora cambiare il modo in cui vengono create le scene.

**understanding-dataset-lab (udlab) crea scenari di test per l'empatia cognitiva: se un modello è in grado di mantenere una rappresentazione accurata di ciò che ogni partecipante in una scena conosce, crede, desidera e di cui ha bisogno.** Scene brevi con diverse persone sono pianificate tramite codice, quindi trasformate in prosa da un modello di scrittura locale. La risposta a ogni domanda su una scena deriva dal piano del codice, mai da un modello.

È il complemento di [reasoning-dataset-lab](https://github.com/mcp-tool-shop-org/reasoning-dataset-lab), che fa la stessa cosa per enigmi e elementi di verifica.

## Perché scene con elementi creati tramite codice?

In una scena pianificata, la risposta corretta deriva meccanicamente dagli eventi: chi si è spostato dove, chi poteva vedere, chi ha detto cosa e chi l'ha sentito. Qualcuno che non era in vista quando un oggetto è stato spostato continua a credere che si trovi dove l'ha visto l'ultima volta. Poiché queste regole sono nel codice, la chiave viene calcolata, non giudicata.

Una *chiave creata tramite codice* significa:
- la risposta deriva dal piano della scena del codice;
- è sigillata con un seme privato;
- non viene mai scritta o valutata da un modello.

Ogni versione in prosa viene verificata dal codice rispetto al piano prima di essere considerata valida.

## Stato: cosa è stato provato

I risultati sono conteggi su database di sviluppo di 160 versioni. I risultati negativi vengono presentati per primi, perché hanno plasmato il progetto.

| Fase | Cosa è stato provato | Risultato | Adottato? |
|---|---|---|---|
| 3 | Un elenco di forme di frase accettate nel prompt dello scrittore | 111/160 versioni hanno superato il test, rispetto a 127/160 per il prompt di base. Lo scrittore non ha utilizzato le forme elencate. | **No**, in base alla regola preregistrata |
| 4 | Forme di frase assegnate tramite codice | 89/160, rispetto a 120/160 per il prompt di base. Il test di sviluppo è fallito, quindi non è stata eseguita alcuna verifica. 16 versioni per ramo (10%) non sono state generate, perché il prompt più il budget di output hanno superato il limite di contesto dello scrittore. | **No** |
| 5 | Modifiche al codice e riscrittura del discorso (revisione 2) | Il test di sviluppo ha superato tutti i 7 controlli preregistrati. Bozze iniziali: 126/160, rispetto a un minimo di 112; 132/160 dopo la pipeline. Le modifiche al codice hanno mantenuto 2,501/2,501. Le riscritture del discorso hanno mantenuto 350/358. 0 versioni hanno superato il limite di contesto. | Ha superato i controlli di sviluppo |

**Non ancora mostrato:**
- nessuna esecuzione di verifica definitiva della fase 5;
- nessun modello addestrato sui dati di udlab;
- al momento, nulla sul fatto che un modello possa comprendere le persone in una scena.

## Cosa c'è qui

- **`udlab-gen`**, il generatore. Pianifica gli scenari, esegue la pipeline della fase 5, sigilla le chiavi di ogni set con un seme privato e verifica un set scritto rielaborandolo e confrontandolo.
- **Sonde di scorciatoia**, eseguite in modo indipendente. Verificano se le caratteristiche superficiali di una scena rivelano la risposta.
- **Il prompt dello scrittore** (`prompts/`). Lo scrittore è `qwen3.6:35b`, eseguito localmente tramite Ollama.
- **Un lanciatore Colab** (`scripts/colab/`) per eseguire lo scrittore su una GPU in affitto.
- **753 test.**

## Provalo senza una GPU

Lo scrittore predefinito è un simulacro CPU, quindi l'intera pipeline viene eseguita su qualsiasi macchina:

```bash
git clone https://github.com/mcp-tool-shop-org/understanding-dataset-lab
cd understanding-dataset-lab
uv run udlab-gen --count 3 --out-dir artifacts/demo
uv run udlab-gen --probe-only artifacts/demo/sealed_store/keys.sealed.jsonl
uv run pytest
```

- Python 3.15 o successivo; `uv` lo installa se necessario.
- Il primo comando scrive una parte pubblica (`part_public/`) e un archivio sigillato (`sealed_store/`) in `artifacts/demo/`, una cartella che Git ignora.
- `uv run udlab-gen --help` elenca tutte le opzioni.

## Cosa rimane privato

Questi rimangono privati, perché pubblicarli rivelerebbe le risposte:
- il database di test sigillato e i database di semi di sviluppo;
- ogni archivio sigillato e registro di generazione;
- i registri di esecuzione con i relativi semi di piano.

Nessuno di questi si trova in questo repository o nella sua cronologia. I test utilizzano solo semi sintetici e scene inventate.

## Sicurezza e affidabilità

- **Dati interessati:** file nella cartella `--out-dir` che scegli e il prompt dello scrittore. Con `--writer ollama`, chiama anche un server Ollama locale.
- **Dati non interessati:** nulla al di fuori della cartella di output. Non c'è telemetria e non ci sono chiamate di rete, tranne al server Ollama locale e, se lo esegui, al lanciatore Colab.
- **Autorizzazioni:** nessuna, oltre alla lettura e alla scrittura della cartella di output. In modalità GPU, interrompe il server Ollama che ha avviato se la memoria video supera `--vram-halt-mib`.
- Segnala i problemi di sicurezza come descritto in [SECURITY.md](SECURITY.md).

## Licenza

Il codice e il testo del repository sono soggetti alla [licenza MIT](LICENSE). La definizione citata sotto il logo proviene da Wiktionary e rimane soggetta a [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Non viene utilizzata in nessun set di dati o prompt.
