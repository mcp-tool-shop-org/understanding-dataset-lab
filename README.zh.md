<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.md">English</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

**原型。** udlab 是一项正在进行中的开放研究。方法和代码都可以在这里查看和运行。到目前为止，它的结果是对其生成场景的能力的检查，用于开发数据集。它没有最终或确定的结果，也没有经过训练的模型，并且没有声称任何人工智能是否能够理解任何人。各个阶段仍然可以改变场景的生成方式。

**理解数据集实验室 (udlab) 构建认知同理心的测试场景：即模型是否能够准确地掌握场景中每个参与者所知道、相信、想要和需要的信息。** 通过代码规划包含多个角色的简短场景，然后由本地的写作模型将其转化为文字。对于场景中的每个问题，答案都来自代码的规划，而不是来自模型。

它是 [reasoning-dataset-lab](https://github.com/mcp-tool-shop-org/reasoning-dataset-lab) 的配套项目，后者对谜题和验证项目执行相同的操作。

## 为什么使用代码生成的关键要素的场景？

在规划好的场景中，正确的答案会根据事件机械地得出：谁移动到哪里，谁能看到，谁说了什么，以及谁听到了。当一个物体被移动时，如果有人当时不在视野范围内，他们仍然会认为该物体位于他们上次看到的位置。由于这些规则都包含在代码中，因此关键要素是计算出来的，而不是判断出来的。

*代码生成的关键要素*意味着：
- 答案来自代码对场景的规划；
- 它使用私有种子进行密封；
- 它绝不会被模型写入或评分。

在将其计入结果之前，每个文字描述都会通过代码与规划进行检查。

## 状态：已经尝试过的内容

结果是基于 160 个文字描述的开发数据集的统计数据。负面结果首先呈现，因为它们塑造了设计。

| 阶段 | 尝试过的内容 | 结果 | 是否采用？ |
|---|---|---|---|
| 3 | 写作模型提示中接受的句子形式列表 | 111/160 个文字描述通过了测试，而基准提示的通过率为 127/160。写作模型没有使用列出的形式。 | **否**，根据预先注册的规则 |
| 4 | 由代码分配的句子形式 | 89/160，而基准提示的通过率为 120/160。开发测试失败，因此没有进行确认测试。由于提示加上输出预算超过了写作模型的上下文限制，因此从未生成 16 个文字描述（10%）。 | **No** |
| 5 | 代码移动和重写对话（修订版 2） | 开发测试通过了所有 7 个预先注册的测试。初稿：126/160（最低要求为 112）；流水线之后为 132/160。代码交换保持 2,501/2,501。对话重写保持 350/358。0 次渲染超过上下文限制。 | 通过了其开发测试 |

**尚未展示：**
- 没有对第 5 阶段进行密封的确认测试；
- 没有使用 udlab 数据训练的模型；
- 尚未研究任何模型是否能够理解场景中的人物。

## 这里有什么

- **`udlab-gen`**，生成器。它规划场景，运行第 5 阶段的流水线，使用私有种子密封每个数据集的关键要素，并通过重新渲染和比较来验证书面数据集。
- **快捷测试**，在独立环境中运行。它们测试场景的表面特征是否会泄露答案。
- **写作模型提示**（`prompts/`）。写作模型是 `qwen3.6:35b`，通过 Ollama 在本地运行。
- **Colab 启动器**（`scripts/colab/`），用于在租用的 GPU 上运行写作模型。
- **759 个测试。**

## 尝试在没有 GPU 的情况下运行

默认写作模型是一个 CPU 模拟器，因此整个流水线可以在任何机器上运行：

```bash
git clone https://github.com/mcp-tool-shop-org/understanding-dataset-lab
cd understanding-dataset-lab
uv run udlab-gen --count 3 --out-dir artifacts/demo
uv run udlab-gen --probe-only artifacts/demo/sealed_store/keys.sealed.jsonl
uv run pytest
```

- Python 3.15 或更高版本；如果需要，`uv` 会安装它。
- 第一个命令会在 `artifacts/demo/` 下写入公共部分（`part_public/`）和密封存储（`sealed_store/`），该文件夹会被 Git 忽略。
- `uv run udlab-gen --help` 列出了所有选项。

## 哪些内容保持私有

以下内容保持私有，因为发布它们会泄露答案：
- 密封的测试数据集和开发种子数据集；
- 每一个密封存储和生成日志；
- 包含其计划种子的运行记录。

这些内容都不在此仓库或其历史记录中。测试仅使用合成种子和虚构的场景。

## 安全性和信任

- **访问的数据：** 您选择的 `--out-dir` 下的文件以及写作模型提示。使用 `--writer ollama` 时，它还会调用本地 Ollama 服务器。
- **未访问的数据：** 输出文件夹之外的任何内容。没有遥测数据，也没有除了到本地 Ollama 服务器以及（如果运行）Colab 启动器的网络调用。
- **权限：** 除了读取和写入输出文件夹之外，没有其他权限。在 GPU 模式下，如果视频内存超过 `--vram-halt-mib`，它会停止启动的 Ollama 服务器。
- 按照 [SECURITY.md](SECURITY.md) 中描述的方式报告安全问题。

## 许可证

代码和仓库自身的文本采用 [MIT 许可证](LICENSE)。在徽标下引用的定义来自 Wiktionary，并采用 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)。它不用于任何数据集或提示。
