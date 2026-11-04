<p align="center">
  <a href="README.md">English</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

**プロトタイプ。** udlabは、現在進行中のオープンな研究プロジェクトです。その方法とコードは、ここで確認・実行できます。これまでの結果は、開発環境で、シーンをどれだけうまく生成できるかの検証です。最終的な結果や確定された結果、トレーニングされたモデルは存在せず、AIが誰かを理解しているかどうかについても、断定的な主張はしていません。シーンの作成方法は、今後も変更される可能性があります。

**理解力データセットラボ（udlab）は、認知的な共感のためのテストシナリオを作成します。つまり、モデルが、シーン内の各登場人物が何を知っていて、何を信じ、何を望み、何が必要としているのかを正確に把握できるかどうかを検証します。** 複数の登場人物が登場する短いシーンをコードで計画し、その後、ローカルのライターモデルによって文章に変換します。シーンに関するすべての質問の答えは、コードの計画から得られ、モデルからは決して得られません。

これは、パズルや検証項目について同様の検証を行う[reasoning-dataset-lab](https://github.com/mcp-tool-shop-org/reasoning-dataset-lab)の関連プロジェクトです。

## コードで作成されたキーを持つシーンを使用する理由

計画されたシーンでは、正しい答えは、出来事から必然的に導き出されます。誰がどこへ移動したか、誰が見ることができたか、誰が何を言ったか、誰がそれを聞いたか、などです。ある物体が移動したときに視界から外れていた人物は、その物体が最後に見た場所にまだあると信じています。これらのルールはコードに記述されているため、キーは判断されるのではなく、計算されます。

*コードで作成されたキー*とは、次のことを意味します。
- 答えは、シーンのコードによる計画から得られます。
- 秘密のシードで保護されています。
- モデルによって作成または評価されることはありません。

すべての文章のレンダリングは、カウントされる前に、コードによって計画と比較して検証されます。

## ステータス：これまでに試したこと

結果は、160件のレンダリングの、開発環境でのカウントです。否定的な結果は、デザインに影響を与えたため、最初に示します。

| 段階 | 試したこと | 結果 | 採用されたか？ |
|---|---|---|---|
| 3 | ライターのプロンプトで受け入れられた文の形式のリスト | 111/160件のレンダリングが合格し、ベースラインプロンプトでは127/160件でした。ライターは、リストに記載されている形式を使用しませんでした。 | **いいえ**、事前に登録されたルールによる |
| 4 | コードによって割り当てられた文の形式 | 89/160件、ベースラインでは120/160件。開発ゲートに失敗したため、確認のための実行は行われませんでした。プロンプトと出力の予算の合計が、ライターのコンテキスト制限を超えたため、16件のレンダリング（10％）は生成されませんでした。 | **No** |
| 5 | コードによる移動と書き換えられた会話（改訂版2） | 開発実行では、事前に登録された7つのゲートすべてに合格しました。最初のドラフト：126/160件、最低値は112件。パイプライン後：132/160件。コードの入れ替えにより、2,501/2,501件が維持されました。会話の書き換えにより、350/358件が維持されました。コンテキスト制限を超えるレンダリングは0件でした。 | 開発ゲートに合格 |

**まだ示されていないこと：**
- 段階5の、秘密保持された確認実行は行われていません。
- udlabデータでトレーニングされたモデルはありません。
- シーン内の登場人物を、どのモデルが理解しているかについては、まだ何もわかりません。

## ここに何があるか

- **`udlab-gen`**、ジェネレーター。シナリオを計画し、段階5のパイプラインを実行し、各セットのキーを秘密のシードで保護し、書き換えられたセットを再レンダリングして比較することで、検証します。
- **ショートカットプローブ**。これは、シーンの表面的な特徴が答えを明らかにするかどうかをテストするために、独立して実行されます。
- **ライタープロンプト**（`prompts/`）。ライターは、ローカルでOllamaを通じて実行される`qwen3.6:35b`です。
- レンタルGPUでライターを実行するための**Colabランチャー**（`scripts/colab/`）。
- **759件のテスト**。

## GPUなしで試す

デフォルトのライターはCPUモックであるため、パイプライン全体はどのマシンでも実行できます。

```bash
git clone https://github.com/mcp-tool-shop-org/understanding-dataset-lab
cd understanding-dataset-lab
uv run udlab-gen --count 3 --out-dir artifacts/demo
uv run udlab-gen --probe-only artifacts/demo/sealed_store/keys.sealed.jsonl
uv run pytest
```

- Python 3.15以降。必要に応じて、`uv`がインストールします。
- 最初のコマンドは、公開部分（`part_public/`）と、秘密保持されたストア（`sealed_store/`）を、Gitによって無視されるフォルダである`artifacts/demo/`の下に書き込みます。
- `uv run udlab-gen --help`は、すべてのオプションをリストします。

## 秘密にされるもの

これらは秘密にされます。なぜなら、これらを公開すると、答えが明らかになってしまうからです。
- 秘密保持されたテストバンクと、開発用シードバンク。
- 秘密保持されたすべてのストアと、生成ログ。
- 計画シードを含む、実行記録。

これらはすべて、このリポジトリまたはその履歴に含まれていません。テストでは、合成シードと架空のシーンのみが使用されます。

## セキュリティと信頼

- **アクセスされるデータ：** 選択した`--out-dir`内のファイルと、ライタープロンプト。`--writer ollama`を使用すると、ローカルのOllamaサーバーも呼び出されます。
- **アクセスされないデータ：** 出力フォルダ外のものは何もありません。テレメトリはなく、ローカルのOllamaサーバーと、実行した場合のColabランチャーへのネットワーク呼び出しのみです。
- **権限：** 出力フォルダの読み書き以上の権限はありません。GPUモードでは、ビデオメモリが`--vram-halt-mib`を超えると、起動したOllamaサーバーを停止します。
- セキュリティの問題は、[SECURITY.md](SECURITY.md)に記載されている方法で報告してください。

## ライセンス

コードとリポジトリ自体のテキストは、[MITライセンス](LICENSE)の対象です。ロゴの下に引用されている定義は、Wiktionaryからのもので、[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)のままです。これは、データセットまたはプロンプトで使用されません。
