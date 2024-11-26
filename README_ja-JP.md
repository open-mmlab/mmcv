<div align="center">
  <img src="https://raw.githubusercontent.com/open-mmlab/mmcv/main/docs/en/mmcv-logo.png" width="300"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab ウェブサイト</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab プラットフォーム</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![platform](https://img.shields.io/badge/platform-Linux%7CWindows%7CmacOS-blue)](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmcv)](https://pypi.org/project/mmcv/)
[![pytorch](https://img.shields.io/badge/pytorch-1.8~2.0-orange)](https://pytorch.org/get-started/previous-versions/)
[![cuda](https://img.shields.io/badge/cuda-10.1~11.8-green)](https://developer.nvidia.com/cuda-downloads)
[![PyPI](https://img.shields.io/pypi/v/mmcv)](https://pypi.org/project/mmcv)
[![badge](https://github.com/open-mmlab/mmcv/workflows/build/badge.svg)](https://github.com/open-mmlab/mmcv/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmcv/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmcv)
[![license](https://img.shields.io/github/license/open-mmlab/mmcv.svg)](https://github.com/open-mmlab/mmcv/blob/master/LICENSE)

[📘ドキュメント](https://mmcv.readthedocs.io/en/latest/) |
[🛠️インストール](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) |
[🤔問題報告](https://github.com/open-mmlab/mmcv/issues/new/choose)

</div>

<div align="center">

[English](README.md) | [简体中文](README_zh-CN.md) | 日本語

</div>

## ハイライト

OpenMMLabチームは、2022年9月1日に世界人工知能会議で新世代のトレーニングエンジン[MMEngine](https://github.com/open-mmlab/mmengine)をリリースしました。これは、深層学習モデルのトレーニングのための基盤ライブラリです。MMCVと比較して、より汎用的で強力なランナー、より統一されたインターフェースを持つオープンアーキテクチャ、およびよりカスタマイズ可能なトレーニングプロセスを提供します。

MMCV v2.0.0の公式バージョンは2023年4月6日にリリースされました。バージョン2.xでは、トレーニングプロセスに関連するコンポーネントを削除し、データ変換モジュールを追加しました。また、2.x以降、パッケージ名を**mmcv**から**mmcv-lite**、**mmcv-full**から**mmcv**に変更しました。詳細は[互換性ドキュメント](docs/en/compatibility.md)を参照してください。

MMCVは、[1.x](https://github.com/open-mmlab/mmcv/tree/1.x)（元の[master](https://github.com/open-mmlab/mmcv/tree/master)ブランチに対応）と**2.x**（**main**ブランチに対応、現在のデフォルトブランチ）バージョンの両方を同時に維持します。詳細は[ブランチメンテナンス計画](README.md#branch-maintenance-plan)を参照してください。

## 紹介

MMCVは、コンピュータビジョン研究のための基盤ライブラリであり、以下の機能を提供します：

- [画像/ビデオ処理](https://mmcv.readthedocs.io/en/latest/understand_mmcv/data_process.html)
- [画像とアノテーションの可視化](https://mmcv.readthedocs.io/en/latest/understand_mmcv/visualization.html)
- [画像変換](https://mmcv.readthedocs.io/en/latest/understand_mmcv/data_transform.html)
- [さまざまなCNNアーキテクチャ](https://mmcv.readthedocs.io/en/latest/understand_mmcv/cnn.html)
- [一般的なCPUおよびCUDAオペレーションの高品質な実装](https://mmcv.readthedocs.io/en/latest/understand_mmcv/ops.html)

以下のシステムをサポートしています：

- Linux
- Windows
- macOS

詳細な機能と使用方法については、[ドキュメント](http://mmcv.readthedocs.io/en/latest)を参照してください。

注意：MMCVはPython 3.7以上が必要です。

## インストール

MMCVには2つのバージョンがあります：

- **mmcv**：包括的で、フル機能とさまざまなCUDAオペレーションを備えています。ビルドに時間がかかります。
- **mmcv-lite**：軽量で、CUDAオペレーションは含まれていませんが、他のすべての機能が含まれています。mmcv\<1.0.0に似ています。CUDAオペレーションが不要な場合に便利です。

**注意**：同じ環境に両方のバージョンをインストールしないでください。そうしないと、`ModuleNotFound`のようなエラーが発生する可能性があります。インストールする前に、もう一方をアンインストールする必要があります。`CUDAが利用可能な場合は、フルバージョンのインストールを強くお勧めします`。

### mmcvのインストール

mmcvをインストールする前に、[PyTorch公式インストールガイド](https://github.com/pytorch/pytorch#installation)に従ってPyTorchが正常にインストールされていることを確認してください。Apple Siliconユーザーの場合は、PyTorch 1.13+を使用してください。

mmcvをインストールするコマンド：

```bash
pip install -U openmim
mim install mmcv
```

mmcvのバージョンを指定する必要がある場合は、次のコマンドを使用できます：

```bash
mim install mmcv==2.0.0
```

上記のインストールコマンドが`.whl`で終わるプリビルドパッケージではなく、`.tar.gz`で終わるソースパッケージを使用している場合は、PyTorch、CUDA、またはmmcvバージョンに対応するプリビルドパッケージがない可能性があります。その場合は、[mmcvをソースからビルド](https://mmcv.readthedocs.io/en/latest/get_started/build.html)できます。

<details>
<summary>プリビルドパッケージを使用したインストールログ</summary>

リンクを検索中：https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html<br />
mmcvを収集中<br />
<b>ダウンロード中 https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/mmcv-2.0.0-cp38-cp38-manylinux1_x86_64.whl</b>

</details>

<details>
<summary>ソースパッケージを使用したインストールログ</summary>

リンクを検索中：https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html<br />
mmcv==2.0.0を収集中<br />
<b>ダウンロード中 mmcv-2.0.0.tar.gz</b>

</details>

詳細なインストール方法については、[インストールドキュメント](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)を参照してください。

### mmcv-liteのインストール

PyTorch関連のモジュールを使用する必要がある場合は、[PyTorch公式インストールガイド](https://github.com/pytorch/pytorch#installation)に従ってPyTorchが正常にインストールされていることを確認してください。

```bash
pip install -U openmim
mim install mmcv-lite
```

## FAQ

インストールの問題や実行時の問題が発生した場合は、[FAQ](https://mmcv.readthedocs.io/en/latest/faq.html)を参照して解決策があるかどうかを確認してください。問題が解決しない場合は、[issue](https://github.com/open-mmlab/mmcv/issues)を開いてください。

## 引用

このプロジェクトが研究に役立つ場合は、以下のように引用してください：

```latex
@misc{mmcv,
    title={{MMCV: OpenMMLab} Computer Vision Foundation},
    author={MMCV Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmcv}},
    year={2018}
}
```

## 貢献

MMCVの改善に貢献していただけることを感謝します。貢献ガイドラインについては、[CONTRIBUTING.md](CONTRIBUTING.md)を参照してください。

## ライセンス

MMCVはApache 2.0ライセンスの下でリリースされていますが、このライブラリの一部の特定の操作には他のライセンスが付与されています。商業的な目的でコードを使用する場合は、[LICENSES.md](LICENSES.md)を参照して注意深く確認してください。

## ブランチメンテナンス計画

MMCVには現在、main、1.x、master、および2.xの4つのブランチがあります。2.xはmainブランチのエイリアスであり、masterは1.xブランチのエイリアスです。将来的には2.xとmasterブランチは削除されます。MMCVのブランチは次の3つのフェーズを経ます：

| フェーズ                | 時間                  | ブランチ                                                                                                                              | 説明                                                                                                                                            |
| -------------------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| RC期間            | 2022.9.1 - 2023.4.5   | リリース候補コード（2.xバージョン）は2.xブランチでリリースされます。デフォルトのmasterブランチはまだ1.xバージョンです                     | Masterと2.xブランチは通常通り反復されます                                                                                                               |
| 互換性期間 | 2023.4.6 - 2023.12.31 | **2.xブランチはmainブランチに名前が変更され、デフォルトブランチとして設定されました**。1.xブランチは1.xバージョンに対応します | 古いバージョン1.xのメンテナンスを続け、ユーザーのニーズに応えますが、互換性を壊す変更はできるだけ導入しません。mainブランチは通常通り反復されます |
| メンテナンス期間   | 2024/1/1から         | デフォルトのmainブランチは2.xバージョンに対応し、1.xブランチは1.xバージョンに対応します                                                        | 1.xブランチはメンテナンスフェーズに入り、新機能のサポートは行いません。mainブランチは通常通り反復されます                                                     |

## OpenMMLabの他のプロジェクト

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLabの深層学習モデルトレーニング基盤ライブラリ。
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLabのコンピュータビジョン基盤ライブラリ。
- [MIM](https://github.com/open-mmlab/mim): MIMはOpenMMlabのパッケージをインストールします。
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLabの画像分類ツールボックスとベンチマーク。
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLabの検出ツールボックスとベンチマーク。
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLabの次世代汎用3Dオブジェクト検出プラットフォーム。
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLabの回転オブジェクト検出ツールボックスとベンチマーク。
- [MMYOLO](https://github.com/open-mmlab/mmyolo): OpenMMLabのYOLOシリーズツールボックスとベンチマーク。
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLabのセマンティックセグメンテーションツールボックスとベンチマーク。
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLabのテキスト検出、認識、および理解ツールボックス。
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLabのポーズ推定ツールボックスとベンチマーク。
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLabの3Dヒューマンパラメトリックモデルツールボックスとベンチマーク。
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLabの自己教師あり学習ツールボックスとベンチマーク。
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLabのモデル圧縮ツールボックスとベンチマーク。
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLabの少数ショット学習ツールボックスとベンチマーク。
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLabの次世代アクション理解ツールボックスとベンチマーク。
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLabのビデオ認識ツールボックスとベンチマーク。
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLabのオプティカルフローツールボックスとベンチマーク。
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLabの画像およびビデオ編集ツールボックス。
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLabの画像およびビデオ生成モデルツールボックス。
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLabのモデルデプロイメントフレームワーク。
